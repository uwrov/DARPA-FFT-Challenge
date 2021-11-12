import torch
import torch.nn as nn
from model.MDN import MDN, mdn_loss
from model.LSTM_model import myLSTM

class LSTM_MDN(nn.Module):
    def __init__(self, Config, device):
        super(LSTM_MDN, self).__init__()
        self.rnn = myLSTM(Config, device)
        self.mdn = MDN(Config["input_features"], Config["out_features"], Config["num_gaussians"])
    
    def forward(self, minibatch, targets = None):
        rnn_out = self.rnn(minibatch, targets = None)
        pi, sigma, mu, pho = self.mdn(rnn_out)
        out = pi, sigma, mu, pho 
        if (targets is not None):
            loss = mdn_loss(pi, mu, sigma, pho, targets)
            '''
            if loss <= 0: 
                print(loss)
                raise KeyboardInterrupt
            '''
            return out, loss
        else:
            return out