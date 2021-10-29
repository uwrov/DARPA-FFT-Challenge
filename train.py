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
from datasets import lstm_data_prepare, data_iter_random, scale_ratio, test_iter
from model import myLSTM
from utils import Config, AverageMeter
from geopy.distance import great_circle

parser = argparse.ArgumentParser(description='Hand pose from mutliple views')
parser.add_argument('-o', '--output', default='temp', type=str,
                    help='the name of output file ')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to lstm checkpoint (default: none)')

# training the LSTM model
def train(epoch, num_epochs, model, train_data, optimizer, config, device):
    model.train()
    points, label, _ = train_data
    data_iter = data_iter_random(points, label, Config["batch_size"], Config["sequence_length"], Config["slide_step"], 1, device)
    num_examples = 0
    for id in range(0, len(points)):
        num_examples += (points[id].shape[0] - Config["sequence_length"] ) // Config["slide_step"] + 1
    total_step = num_examples // Config["batch_size"]
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

        pred, truth = outputs.data, labels.data
        err = nn.functional.mse_loss(pred, truth)
        
        mae.update(err.data.item(), points.size(0))
        losses.update(loss.data.item(), points.size(0))

        if (i+1) % 150 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Mean_error: {:.4f}, Mean_Loss: {:.4f}' 
                .format(epoch+1, num_epochs, i+1, total_step, mae.avg, losses.avg))
        i += 1 



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
        previous_points, previous_label, _ = train_data
        spots_num = len(points)
        #test_iter = data_iter_random(points, label, Config["batch_size"], Config["sequence_length"], Config["slide_step"], 0, device)
        trace_dis_day = []
        for id in range(0, spots_num):
            location_pred = [absolute_pos[id][0][0], absolute_pos[id][0][1]] 
            location_gt = [absolute_pos[id][0][0], absolute_pos[id][0][1]] 
            data_iter = test_iter(points[id], previous_points[id], label[id], 12, Config["sequence_length"],device)
            trace_err = []
            for inputs, labels in data_iter:
                labels = labels.cuda(device)
                inputs = inputs.cuda(device)
                # Forward pass
                outputs, loss = model(inputs, labels)
                losses.update(loss.data.item(), inputs.size(0))
                pred = outputs.data
                truth = labels.data
                err = nn.functional.mse_loss(pred, truth)
                mae.update(err.data.item(), inputs.size(0))
                
                # track the trace of spotid
                pred = pred.cpu().numpy()
                truth = truth.cpu().numpy()
                for i in range(0, pred.shape[0]):
                    location_pred[0] += pred[i][0]/scale_ratio
                    location_pred[1] += pred[i][1]/scale_ratio
                    location_gt[0]+= truth[i][0]/scale_ratio
                    location_gt[1]+= truth[i][1]/scale_ratio
                    circle_dis = great_circle(location_pred, location_gt).km
                    trace_err.append(circle_dis)
            day_num = (len(trace_err) - 12) // 24 + 1
            day_err = []
            for d in range(0, day_num):
                day_err.append(trace_err[11 + d*24])
            trace_dis_day.append(day_err)

        mean_dis = np.mean(np.array(trace_dis_day), axis=0)
        print('Testing Mean_err: {:.4f}, Mean_Loss: {:.4f}'.format(mae.avg, losses.avg))
        print('Testing Great Circle Dis by days: ', mean_dis)

        return mae.avg, losses.avg

def main(args):
    output_file = args.output
    # read data from excel file
    train_data, test_data = lstm_data_prepare(divide_factor = Config['divide_factor'], feature_number = Config['feature_num'] )

    # initial the deep learning model
    master_gpu = 0 # GPU you will use in training
    model = myLSTM( Config, master_gpu)
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
        # Train the model
        min_err = 10000
        for epoch in range(Config['num_epoch']):
            train( epoch, Config['num_epoch'], model, train_data, optimizer, Config, master_gpu)
            err, loss = validate(model, test_data,train_data, Config,  master_gpu, 0)
            if err < min_err:
                min_err = err
                torch.save(model.state_dict(), './ckpt/' + output_file +'lstm_bestmodel.ckpt')
            torch.save(model.state_dict(), './ckpt/' + output_file +'lstm_checkpoint.ckpt')
        checkpoint = torch.load('./ckpt/' + output_file + '/lstm_bestmodel.ckpt')
        model.load_state_dict(checkpoint)
    err, loss = validate(model, test_data, train_data, Config, master_gpu, 1)
    print("Final error: ", err)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)