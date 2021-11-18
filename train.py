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
from datasets import lstm_data_prepare, data_iter_random, scale_ratio, prepare_test, lstm_data_prepare_json
from model import myLSTM, LSTM_MDN, sampling, return_expecation_value
from utils import Config, AverageMeter, visual_path
from geopy.distance import great_circle

parser = argparse.ArgumentParser(description='Hand pose from mutliple views')
parser.add_argument('-o', '--output', default='temp', type=str,
                    help='the name of output file ')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to lstm checkpoint (default: none)')

MDN_USE = True

lstm_data_prepare_json(0.8, 7)
raise KeyboardInterrupt

def calculate_score(trace_dis_day):
    threshold = [4, 8, 16, 32]
    total_num = trace_dis_day.shape[0]
    for d in range(0, 5):
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


def validate(model, test_data, train_data, config, device, visual = 0):
    # Test the model
    model.eval()
    losses = AverageMeter()
    mae = AverageMeter()
    day_distance = []

    with torch.no_grad():
        correct = 0
        total = 0
        points, label, absolute_pos = test_data
        previous_points, previous_label, previous_path = train_data
        spots_num = len(points)
        #test_iter = data_iter_random(points, label, Config["batch_size"], Config["sequence_length"], Config["slide_step"], 0, device)
        trace_dis_day = []
        path_gt = []
        path_truth = []
        for id in range(0, spots_num):
            location_pred = [absolute_pos[id][0][0], absolute_pos[id][0][1]] 
            location_gt = [absolute_pos[id][0][0], absolute_pos[id][0][1]] 
            trace_err = []

            previous_trace = previous_path[id]
            trace_predict = [[absolute_pos[id][0][0], absolute_pos[id][0][1]]]
            trace_gt = [[absolute_pos[id][0][0], absolute_pos[id][0][1]]]

            history_movement = [ [points[id][0][-2], points[id][0][-1]] ]

            for idx in range(0, points[id].shape[0]):
                inputs, labels = prepare_test(points[id], previous_points[id], label[id], Config["sequence_length"], idx, np.array(history_movement), Config)

                labels = labels.cuda(device)
                inputs = inputs.cuda(device)
                # Forward pass
                outputs, loss = model(inputs, labels)
                losses.update(loss.data.item(), inputs.size(0))
                if not MDN_USE:
                    pred = outputs.data.cpu()
                else:
                    pi, sigma, mu, pho = outputs[0][0, ...].data.cpu(), outputs[1][0, ...].data.cpu(), outputs[2][0, ...].data.cpu(), outputs[3][0, ...].data.cpu()
                    pred = return_expecation_value(pi, mu)#sampling(pi, sigma, mu, pho, n= 1000)
                truth = labels.data.cpu()
                err = nn.functional.mse_loss(pred, truth)
                mae.update(err.data.item(), inputs.size(0))
                
                # track the trace of spotid
                pred = pred.numpy()
                truth = truth.numpy()
                history_movement.append([pred[0,0], pred[0,1]]) 

                for i in range(0, pred.shape[0]):
                    location_pred[0] += pred[i][0]/scale_ratio
                    location_pred[1] += pred[i][1]/scale_ratio
                    location_gt[0]+= truth[i][0]/scale_ratio
                    location_gt[1]+= truth[i][1]/scale_ratio
                    trace_predict.append([location_pred[0], location_pred[1]])
                    trace_gt.append([location_gt[0], location_gt[1]])
                    circle_dis = great_circle(location_pred, location_gt).km
                    trace_err.append(circle_dis)
            day_num = (len(trace_err) - 12) // 24 + 1
            day_err = []
            for d in range(0, day_num):
                day_err.append(trace_err[11 + d*24])
            trace_dis_day.append(day_err)

            if visual:
                visual_path(previous_trace, np.array(trace_gt), np.array(trace_predict),id)
        
        calculate_score(np.array(trace_dis_day))
        mean_dis = np.mean(np.array(trace_dis_day), axis=0)
        print('Testing Mean_err: {:.4f}, Mean_Loss: {:.4f}'.format(mae.avg, losses.avg))
        print('Testing Great Circle Dis by days: ', mean_dis)

        return mae.avg, losses.avg, mean_dis[4]

def main(args):
    output_file = args.output
    # read data from excel file
    train_data, test_data = lstm_data_prepare(divide_factor = Config['divide_factor'], feature_number = Config['feature_num'] )

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
            if epoch %5 == 4:
                err, loss, trace_err = validate(model, test_data,train_data, Config,  master_gpu, 0)
                if trace_err < min_err:
                    min_err = trace_err
                    torch.save(model.state_dict(), './ckpt/' + output_file +'lstm_bestmodel.ckpt')
            torch.save(model.state_dict(), './ckpt/' + output_file +'lstm_checkpoint.ckpt')
        checkpoint = torch.load('./ckpt/' + output_file + 'lstm_bestmodel.ckpt')
        model.load_state_dict(checkpoint)
    err, loss, trace_err = validate(model, test_data, train_data, Config, master_gpu, 1)
    print("Final error: ", err, trace_err)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)