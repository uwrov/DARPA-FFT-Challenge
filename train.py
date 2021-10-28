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
from datasets import lstm_data_prepare, data_iter_random
from model import myLSTM
from utils import Config

parser = argparse.ArgumentParser(description='Hand pose from mutliple views')
parser.add_argument('-o', '--output', default='temp', type=str,
                    help='the name of output file ')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to lstm checkpoint (default: none)')

# training the LSTM model
def train(epoch, num_epochs, model, train_data, criterion, optimizer, config, device):
    model.train()
    points, label = train_data
    data_iter = data_iter_random(points, label, Config["batch_size"], Config["sequence_length"], Config["slide_step"], if_random=1)
    num_examples = (points.shape[0] - Config["sequence_length"] // Config["slide_step"]
    total_step = num_examples // Config["batch_size"]
    i = 0
    losses = AverageMeter()

    for points, labels in data_iter:
        labels = labels.cuda(device)
        points = points.cuda(device)
        # Forward pass
        outputs, loss = model(points, None)
        losses.update(loss.data.item(), points.size(0))
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Current_Loss: {:.4f}, Mean_Loss: {:.4f}' 
                .format(epoch+1, num_epochs, i+1, total_step, losses.val, losses.avg))
        i += 1 



def validate(model, test_data, criterion, config, device, visual = 0):
    # Test the model
    flag = 0
    model.eval()
    losses = AverageMeter()
    with torch.no_grad():
        correct = 0
        total = 0
        points, label = test_data
        test_iter = data_iter_random(points, label, Config.batch_size, Config.sequence_length, Config.slide_step, 0, device)
        for points, labels in test_iter:
            labels = labels.cuda(device)
            points = points.cuda(device)
            # Forward pass
            outputs, loss = model(points, None)
            losses.update(loss.data.item(), points.size(0))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if flag == 0:
                ground_truth = labels.cpu().numpy()
                result = predicted.cpu().numpy()
                flag = 1
            else:
                ground_truth = np.hstack((ground_truth,labels.cpu().numpy()))
                result = np.hstack((result,predicted.cpu().numpy()))
                
        print('Current_Loss: {:.4f}, Mean_Loss: {:.4f}'.format(losses.val, losses.avg))
        print('Test Accuracy of the model: {} %'.format(100 * correct / total)) 
        return correct / total, ground_truth, result

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
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Train the model
        max_acc = 0
        for epoch in range(Config['num_epoch']):
            train( epoch, Config['num_epoch'], model, train_data, optimizer, Config, master_gpu)
            acc, _, _ = validate(model, test_data, Config,  master_gpu, 0)
            if acc > max_acc:
                max_acc = acc
                torch.save(model.state_dict(), './ckpt/' + output_file +'lstm_bestmodel.ckpt')
            torch.save(model.state_dict(), './ckpt/' + output_file +'lstm_checkpoint.ckpt')
        checkpoint = torch.load('./ckpt/' + output_file + '/lstm_bestmodel.ckpt')
        model.load_state_dict(checkpoint)
    acc, truth, result = validate(model, test_data, Config, master_gpu, 1)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)