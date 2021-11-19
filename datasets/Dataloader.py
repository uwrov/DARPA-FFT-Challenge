import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import random
import torch
import json
import time
from datasets.Gribloader import get_field
#----------index of feature name ------------------
sampling_interval = 60*60

significantWaveHeight = 0
peakPeriod = 1
meanPeriod = 2
peakDirection = 3
peakDirectionalSpread = 4
meanDirection = 5
meanDirectionalSpread = 6
timestamp = 7
latitude = 8
longitude = 9 
epoch = 10
spotId = 11
scale_ratio = 50
#--------------------------------------------------


def read_json_file(name):
    with open(name,'r') as load_f:
        load_dict = json.load(load_f)
    print(load_dict["all_data"].keys())


def read_file():
    file = "./datasets/5eccca0e424247ae9cadb288470a1398.xlsx"
    data = pd.read_excel(file) # read the data from excel file
    index_name = [var for var in data.columns]  # index names in the first row
    raw_data = data.values
    print("Data loading successfully!")

    return index_name, raw_data

def get_path(data):
    all_path = []
    previous_id = data[0, spotId]
    timestamp = []
    path = []
    for i in range(0, data.shape[0]):
        current_id = data[i, spotId]
        if current_id != previous_id:
            all_path.append({"spot_id": previous_id, "timestamp": timestamp, "path": np.array(path)})
            path = []
            timestamp = []
        else:
            timestamp.append(data[i, epoch])
            path.append([data[i, latitude], data[i, longitude]])
        previous_id = current_id
    return all_path

def visualize_path(all_path):
    plt.figure(0)
    for path in all_path:
        id = path["spot_id"]
        route = path["path"]
        plt.plot(route[:,0], route[:,1], '--', markersize=2, label = "spot " + str(id))
    plt.legend()
    plt.show()

def feature_normalize(data, number_to_norm):
    for i in range(0, number_to_norm):
        max_value = np.max(data[:, i])
        min_value = np.min(data[:, i])
        data[:, i] =  (data[:, i] - min_value)/(max_value - min_value)

    return data

def prepare_test(points, previous_points, label, num_steps, pos, history, Config):
    assert history.shape[0] == pos+1
    if pos - num_steps + 1 >= 0:
        X = np.concatenate( (points[pos - num_steps + 1: pos + 1, 0:Config["feature_num"]], history[pos - num_steps + 1: pos + 1, :]), axis = 1)
        Y = label[pos]
        return torch.tensor([X], dtype=torch.float32), torch.tensor([Y], dtype=torch.float32)
    else:
        testing_part = np.concatenate( (points[0: pos + 1, 0:7] , history[0:pos + 1, :]), axis = 1)
        training_part = previous_points[-(num_steps - pos -1):, :]
        X = np.concatenate((training_part, testing_part), axis = 0) 
        Y = label[pos]
        return torch.tensor([X], dtype=torch.float32), torch.tensor([Y], dtype=torch.float32)
    

def test_iter(points, previous_points, label, batch_size, num_steps, device=None):
    num_examples = points.shape[0] 
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))

    # 返回从pos开始的长为num_steps的序�?
    def _data(pos):
        if pos - num_steps + 1 >= 0:
            return points[pos - num_steps + 1: pos + 1, :]
        else:
            return np.concatenate((previous_points[-(num_steps - pos -1):, :], points[0: pos + 1, :]), axis = 0)
    def _label(pos):
        return label[pos]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(epoch_size):
        # 每次读取batch_size个随机样�?
        j = i * batch_size
        if j + batch_size <= num_examples:  batch_indices = example_indices[j: j + batch_size]
        else: batch_indices = example_indices[j:]
        #print(batch_indices)
        X = [_data(index) for index in batch_indices]
        Y = [_label(index) for index in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)


def data_iter_random(points, label, batch_size, num_steps, slide_step, if_random, device=None):
    # 是因为输出的索引x是相应输入的索引y
    num_spots = len(points)
    example_indices = []
    num_examples = 0
    for id in range(0, num_spots):
        num_examples += (points[id].shape[0] - num_steps ) // slide_step + 1
        for num in range(0, (points[id].shape[0] - num_steps ) // slide_step + 1):
            example_indices.append([id, num*slide_step])
    assert len(example_indices) == num_examples
    epoch_size = num_examples // batch_size
    if if_random: random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序�?
    def _data(index, step):
        #pos = index[1]*step
        return points[index[0]][index[1]: index[1] + num_steps, :]
    def _label(index, step):
        #pos = index[1]*step
        return label[index[0]][index[1] + num_steps - 1]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(epoch_size):
        # 每次读取batch_size个随机样�?
        j = i * batch_size
        if j + batch_size <= num_examples:  batch_indices = example_indices[j: j + batch_size]
        else: batch_indices = example_indices[j:]
        #print(batch_indices)
        X = [_data(index, slide_step) for index in batch_indices]
        Y = [_label(index, slide_step) for index in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)

def convert_timestamp(time_string):
    #转换成时间数组
    time_s = time_string.split(".")[0]
    timeArray = time.strptime(time_s, "%Y-%m-%dT%H:%M:%S")
    #转换成时间戳
    timestamp = time.mktime(timeArray)
    return timestamp

def lstm_data_prepare_json(divide_factor, feature_number):
    data_extraction = []
    
    days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    spots_name = []
    current_time = []
    data_missing = []
    is_invalid = []
    for d in days:
        filename = "./datasets/darpa_data/challenge_1-day_sofar_202111%02d_day%dJSON.json"%(d+1, d)
        with open(filename,'r') as load_f:
            load_dict = json.load(load_f)
            for data in load_dict["all_data"]:
                spotid = data["data"]["spotterId"]
                if spotid in spots_name:
                    id = spots_name.index(spotid)
                else:
                    id = len(spots_name)
                    spots_name.append(spotid)
                    data_extraction.append([])
                    data_missing.append(0)
                    current_time.append(-1)
                    is_invalid.append(0)
                feature_data = data["data"]["waves"]

                for point in feature_data:
                    timestamp = convert_timestamp(point["timestamp"])
                    missing_hour = 0
                    if current_time[id] == -1: 
                        current_time[id] = timestamp
                    else: 
                        if abs(timestamp - current_time[id] - sampling_interval) > 600: 
                            missing_hour = (timestamp-current_time[id])/3600 - 1
                            #print(spotid, d, point["timestamp"], )
                            data_missing[id] += missing_hour
                        current_time[id] = timestamp
                    
                    current_feature = [
                            point["significantWaveHeight"],
                            point["peakPeriod"],
                            point["meanPeriod"],
                            point["peakDirection"],
                            point["peakDirectionalSpread"],
                            point["meanDirection"],
                            point["meanDirectionalSpread"],
                            point["latitude"],
                            point["longitude"] ]
                    result = get_field(point["latitude"], point["longitude"], timestamp)
                    print(point["latitude"], point["longitude"], timestamp)
                    print(result)
                    raise KeyboardInterrupt
                    if missing_hour ==0:
                        data_extraction[id].append(current_feature)
                    elif missing_hour == 1:
                        previous_feature = data_extraction[id][-1]
                        middle_feature = [1/2*previous_feature[i] + 1/2*current_feature[i] for i in range(0, len(current_feature))]
                        
                        data_extraction[id].append(middle_feature)
                        data_extraction[id].append(current_feature)
                    elif missing_hour == 2:
                        previous_feature = data_extraction[id][-1]
                        middle_feature1 = [2/3*previous_feature[i] + 1/3*current_feature[i] for i in range(0, len(current_feature))]
                        middle_feature2 = [1/3*previous_feature[i] + 2/3*current_feature[i] for i in range(0, len(current_feature))]
                        
                        data_extraction[id].append(middle_feature1)
                        data_extraction[id].append(middle_feature2)
                        data_extraction[id].append(current_feature)
                    elif missing_hour > 2:
                        is_invalid[id] = 1 
    total_number = []
    valid_data = []
    for i in range(0, len(spots_name)):
        if is_invalid[i] or len(data_extraction[i]) < 150:
            print("invalid data for: ", len(data_extraction[i]), spots_name[i], data_missing[i])
            #assert len(data_extraction[i]) == len(data_extraction[i+1])
        else: 
            valid_data.append(data_extraction[i])
            total_number.append(len(data_extraction[i]))
    print(len(total_number), total_number)
    print(data_missing)
    all_data = np.concatenate(valid_data, axis = 0)
    print(all_data.shape)
    all_data = feature_normalize(all_data, feature_number)

    train_data = ([], [], [])
    test_data = ([], [], [])

    index = 0
    for i in range(0, len(total_number)):
        raw_data = np.array(all_data[index : index + total_number[i]]) 
        index += total_number[i]
        delta_pos = raw_data[1:, feature_number:] - raw_data[:-1, feature_number:]
        training_num = int(divide_factor*total_number[i])

        inputs = np.concatenate((raw_data[1:-1, 0:feature_number], delta_pos[:-1, :]), axis = 1 )

        outputs = delta_pos[1:, :]
        ref = raw_data[1:-1, feature_number:]

        train_data[0].append(inputs[:training_num])
        train_data[1].append(outputs[:training_num])
        train_data[2].append(ref[:training_num])
        test_data[0].append(inputs[training_num:])
        test_data[1].append(outputs[training_num:])
        test_data[2].append(ref[training_num:])
    
    return train_data, test_data

                

def lstm_data_prepare(divide_factor, feature_number):
    train_data = ([], [], [])
    test_data = ([], [], [])
    index_name, data = read_file()
    #path = get_path(data)

    data = feature_normalize(data, feature_number) # normalize the feature to [0, 1]
    training_numbers = [] 
    testing_numbers = []

    previous_id = data[0, spotId]
    timestamp = []

    delta_movement = [] # output of model 
    features = [] # input of of model
    absolute_pos = []

    for i in range(0, data.shape[0]):
        current_id = data[i, spotId]
        if current_id != previous_id:
            # concatenate the features with history movement as the input features
            inputs = np.concatenate((np.array(features)[:-1, :], np.array(delta_movement)[:-1, :]), axis =1 )
            outputs = np.array(delta_movement)[1:, :]
            ref = np.array(absolute_pos)[:-1, :]
            assert inputs.shape[0] == outputs.shape[0]
            # divide the dataset into training dataset and testing dataset
            sample_num = inputs.shape[0]
            training_num = int(divide_factor*sample_num)
            training_numbers.append(training_num)
            testing_numbers.append(sample_num - training_num)

            train_data[0].append(inputs[:training_num])
            train_data[1].append(outputs[:training_num])
            train_data[2].append(ref[:training_num])
            test_data[0].append(inputs[training_num:])
            test_data[1].append(outputs[training_num:])
            test_data[2].append(ref[training_num:])

            # reset the tuple to save information for next spotid
            delta_movement = []
            absolute_pos = []
            timestamp = []
            features = []
        else:
            if(i == 0): continue
            timestamp.append(data[i, epoch])
            features.append(data[i, 0:feature_number])
            absolute_pos.append([data[i, latitude], data[i, longitude]])
            delta_movement.append([scale_ratio*(data[i, latitude] - data[i-1, latitude]), scale_ratio*(data[i, longitude] - data[i-1, longitude])   ])

        if i == data.shape[0] - 1:
            # concatenate the features with history movement as the input features
            inputs = np.concatenate((np.array(features)[:-1, :], np.array(delta_movement)[:-1, :]), axis =1 )
            outputs = np.array(delta_movement)[1:, :]
            ref = np.array(absolute_pos)[:-1, :]
            assert inputs.shape[0] == outputs.shape[0]
            # divide the dataset into training dataset and testing dataset
            sample_num = inputs.shape[0]
            training_num = int(divide_factor*sample_num)
            training_numbers.append(training_num)
            testing_numbers.append(sample_num - training_num)

            train_data[0].append(inputs[:training_num])
            train_data[1].append(outputs[:training_num])
            train_data[2].append(ref[:training_num])
            test_data[0].append(inputs[training_num:])
            test_data[1].append(outputs[training_num:])
            test_data[2].append(ref[training_num:])

        previous_id = current_id
    print("SpotID number: ", len(testing_numbers))
    #print("LSTM Train data:", training_numbers)
    #print("LSTM Test data:", testing_numbers)
    return train_data, test_data
        


if __name__ == "__main__":
    lstm_data_prepare_json(0.8, 7)
    #read_json_file("./darpa_data/challenge_1-day_sofar_20211102_day1JSON.json")
    #index_name, data = read_file()
    #path = get_path(data)
    #visualize_path(path)
