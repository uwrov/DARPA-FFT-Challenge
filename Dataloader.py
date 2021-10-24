import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#----------index of feature name ------------------
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
#--------------------------------------------------

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



if __name__ == "__main__":
    index_name, data = read_file()
    path = get_path(data)
    visualize_path(path)