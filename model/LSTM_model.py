import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def build_loss(network_config):
    """Get the loss function
    """
    if network_config['loss_type'] == 'l1':
        criterion = nn.L1Loss()
    elif network_config['loss_type'] == 'l2':
        criterion = nn.MSELoss()
    elif network_config['loss_type'] == 'huber':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError('Unsupported loss type!')
    return criterion

# Recurrent neural network (many-to-one)
class myLSTM(nn.Module):
    def __init__(self, Config, device):
        super(myLSTM, self).__init__()
        self.device = device
        self.hidden_size = Config["hidden_size"]
        self.num_layers = Config["num_layers"]
        self.output_size = Config["output_size"]
        self.criterion = build_loss(Config)
        self.bidirectional = Config["bidirectional"]

        self.lstm = nn.LSTM(input_size=input_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               dropout=Config["dropout"],
                               batch_first=True,
                               bidirectional=Config["bidirectional"])

        if self.bidirectional: self.fc = nn.Linear(hidden_size *2, output_size)
        else: self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, state, targets = None):
        # Set initial hidden and cell states 
        h0 = torch.zeros((self.bidirectional + 1)*self.num_layers, x.size(0), self.hidden_size).cuda(self.device)
        c0 = torch.zeros((self.bidirectional + 1)*self.num_layers, x.size(0), self.hidden_size).cuda(self.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        if (targets is not None)
            loss = self.criterion(outputs, targets)
            return out, loss
        else
            return out
