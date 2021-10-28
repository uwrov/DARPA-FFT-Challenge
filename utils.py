Config = {
    "batch_size": 30, # batch size for training
    "sequence_length": 35, # the input sequence length
    "feature_num": 7, # the number of input features
    "output_size": 2, # the output size, [delta_x, delta_y] for next timestamp
    "hidden_size": 64, # hidden size in LSTM model
    "num_layers": 2, # the Layer number of LSTM model
    "learning_rate": 0.001, # learning rate
    "divide_factor": 0.85, # the proportion for training dataset to total dataset
    "slide_step": 1, #slide step for input sequence
    "loss_type": "huber",   # which loss to use (l1 / l2 / huber)
    "num_epoch": 50,
    "dropout": 0.1
}   

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0.0
    self.avg = 0.0
    self.sum = 0.0
    self.count = 0.0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
