from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# python imports
import argparse
import os
import time
import math
from pprint import pprint
import cv2
import pickle
# numpy imports
import numpy as np
import random
import time
# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from datasets import lstm_data_prepare, data_iter_random, scale_ratio, prepare_test, lstm_data_prepare_json, predict_data_prepare, prepare_pred, return_vector_field
from model import myLSTM, LSTM_MDN, sampling, return_expecation_value
from utils import Config, AverageMeter, visual_path, visual_path_pred
from geopy.distance import great_circle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Hand pose from mutliple views')
parser.add_argument('-o', '--output', default='temp', type=str,
                    help='the name of output file ')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to lstm checkpoint (default: none)')

MDN_USE = True

#lstm_data_prepare(0.8, 7, 24*4)
#raise KeyboardInterrupt

def calculate_score(trace_dis_day):
    threshold = [4, 8, 16, 32]
    #print(trace_dis_day.shape)
    total_num = trace_dis_day.shape[0]
    for d in range(0, trace_dis_day.shape[1]):
        day_err = trace_dis_day[:, d]
        num_valid = []
        for t in threshold:
            num_valid.append(np.sum(day_err <= t))
        print("Day %d, less than %d km: %d/%d; less than %d km: %d/%d; less than %d km: %d/%d; less than %d km: %d/%d;" 
        %(d+1, threshold[0], num_valid[0], total_num, 
            threshold[1], num_valid[1], total_num, 
            threshold[2], num_valid[2], total_num, 
            threshold[3], num_valid[3], total_num ))


# training the LSTM model
def train(epoch, num_epochs, model, train_data, optimizer, config, device, total_step, scheduler):
    model.train()
    points, label, _ = train_data
    data_iter = data_iter_random(points, label, Config["batch_size"], Config["sequence_length"], Config["slide_step"], 1, device)

    i = 0
    losses = AverageMeter()
    mae = AverageMeter()

    for points, labels in data_iter:
        labels = labels.cuda(device)
        points = points.cuda(device)
        # Forward pass
        outputs, loss = model(points, labels)
        loss = loss.mean()
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if MDN_USE:
            mae.update(0, points.size(0))
        else:
            pred, truth = outputs.data, labels.data
            err = nn.functional.mse_loss(pred, truth)
            mae.update(err.data.item(), points.size(0))

        losses.update(loss.data.item(), points.size(0))
        if (i+1) % 50 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Mean_error: {:.4f}, Mean_Loss: {:.4f}' 
                .format(epoch+1, num_epochs, i+1, total_step, mae.avg, losses.avg))
        i += 1 
        scheduler.step()


def validate(model, final_status, train_data, config, device, visual = 0):
    # Test the model
    model.eval()
    losses = AverageMeter()
    mae = AverageMeter()
    day_distance = []

    with torch.no_grad():
        correct = 0
        total = 0
        previous_points, previous_label, previous_path = train_data
        spots_num = len(final_status)

        result = {}

        for id in range(0, spots_num): 
            final_timestamp, final_feature, final_pos, spot_name = final_status[id]
            location_pred = [final_pos[0], final_pos[1]] 

            previous_trace = previous_path[id]
            trace_predict = [[final_timestamp, final_pos[0], final_pos[1] ]]

            current_timestamp = final_timestamp
            history_movement = [ final_feature ]

            for idx in range(0, 10*24):
                
                inputs = prepare_pred( previous_points[id], Config["sequence_length"], idx, np.array(history_movement), Config)

                current_timestamp += 3600
                inputs = inputs.cuda(device)
                # Forward pass
                outputs = model(inputs)
                if not MDN_USE:
                    pred = outputs.data.cpu()
                else:
                    pi, sigma, mu, pho = outputs[0][0, ...].data.cpu(), outputs[1][0, ...].data.cpu(), outputs[2][0, ...].data.cpu(), outputs[3][0, ...].data.cpu()
                    pred = return_expecation_value(pi, mu)#sampling(pi, sigma, mu, pho, n= 1000)
                
                pred = pred.numpy()

                pred_pos= pred[:, :2]
                for i in range(0, pred_pos.shape[0]):
                    location_pred[0] += pred_pos[i][0]/scale_ratio
                    location_pred[1] += pred_pos[i][1]/scale_ratio
                    trace_predict.append([current_timestamp, location_pred[0], location_pred[1]])
                    
                history_movement.append(return_vector_field(current_timestamp, location_pred, pred[0, :])) 
            result[spot_name] = trace_predict
            print("save data", len(trace_predict))
            if visual:
                visual_path_pred(previous_trace, np.array(trace_predict),id)

        return result

def main(args):
    output_file = args.output
    # read data from excel filtrain_data, test_data = lstm_data_prepare(divide_factor = Config['divide_factor'], feature_number = Config['feature_num'], Config['TEST_NUM'])
    train_data, final_status = predict_data_prepare(divide_factor = Config['divide_factor'], feature_number = Config['feature_num'], test_num = Config['test_num'], vector_field_use = Config['vector_field'] )
    # initial the deep learning model
    master_gpu = 0 # GPU you will use in training
    if not MDN_USE:
        model = myLSTM( Config, master_gpu)#LSTM_MDN( Config, master_gpu)
    else:
        model = LSTM_MDN( Config, master_gpu)
    model = model.cuda(master_gpu) # load model from CPU to GPU 

    if args.resume: # if have pre-trained model, load it!
        print("loading trained model.....")
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
            # only load the optimizer if necessary
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=Config["learning_rate"])

        points, label, _ = train_data
        num_examples = 0
        for id in range(0, len(points)):
            num_examples += (points[id].shape[0] - Config["sequence_length"] ) // Config["slide_step"] + 1
        total_step = num_examples // Config["batch_size"]

        if Config["scheduler"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, total_step * Config["num_epoch"])
        elif Config["scheduler"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_step, gamma = Config["lr_decay"])
        else: 
            raise TypeError("Unsupported scheduler")
        # Train the model
        min_err = 10000

        for epoch in range(Config['num_epoch']):
            train( epoch, Config['num_epoch'], model, train_data, optimizer, Config, master_gpu, total_step, scheduler)
            #if epoch %5 == 1:
            #    err, loss, trace_err = validate(model, final_status,train_data, Config,  master_gpu, 0)
            #    if trace_err < min_err:
            #        min_err = trace_err
            #        torch.save(model.state_dict(), './ckpt/' + output_file +'lstm_bestmodel.ckpt')
            torch.save(model.state_dict(), './ckpt/' + output_file +'lstm_checkpoint.ckpt')
        checkpoint = torch.load('./ckpt/' + output_file + 'lstm_checkpoint.ckpt')
        model.load_state_dict(checkpoint)
    result_dict = validate(model, final_status, train_data, Config, master_gpu, 1)
    f = open("prediction.pkl","wb")
    # write the python object (dict) to pickle file
    pickle.dump(result_dict,f)
    # close file
    f.close()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
