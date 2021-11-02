import matplotlib.pyplot as plt

Config = {
    "batch_size": 50, # batch size for training
    "sequence_length": 30, # the input sequence length
    "feature_num": 7, # the number of input features
    "output_size": 2, # the output size, [delta_x, delta_y] for next timestamp
    "hidden_size": 64, # hidden size in LSTM model
    "num_layers": 2, # the Layer number of LSTM model
    "learning_rate": 1e-4, # learning rate
    "divide_factor": 0.8, # the proportion for training dataset to total dataset
    "slide_step": 1, #slide step for input sequence
    "loss_type": "huber",   # which loss to use (l1 / l2 / huber)
    "num_epoch": 50,
    "dropout": 0.1,
    "bidirectional": True
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


def visual_path(previous_path, trace_gt, trace_predict, id):
  plt.figure(0)
  plt.clf()
  print("visual for", id)
  plt.plot(previous_path[:,0], previous_path[:,1], '-', markersize=2, label = "training path")
  plt.plot(trace_gt[:,0], trace_gt[:,1], '-', markersize=2, label = "truth path")
  plt.plot(trace_predict[:,0], trace_predict[:,1], '--', markersize=2, label = "predict path")

  plt.legend()
  plt.savefig("./visual/Fig_" + str(id) + ".png")
