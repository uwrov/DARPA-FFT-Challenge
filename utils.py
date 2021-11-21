import matplotlib.pyplot as plt

Config = {
    # ---------------- para for training  --------------------
    "batch_size": 50, # batch size for training
    "learning_rate": 2e-4, # learning rate
    "num_epoch": 70,
    "divide_factor": 0.8, # the proportion for training dataset to total dataset
    "slide_step": 1, #slide step for input sequence
    "para_regu": False,
    "lambda_sigma": 1e-3,
    "lambda_pi": 1e-3,
    "lambda_mu": 1e-3,
    "lambda_pho": 1e-3,
    "scheduler": "cosine",
    "lr_decay": 0.98,
    "test_num": 85,
    "vector_field": True,
    # ---------------- para for LSTM --------------------
    "sequence_length": 30, # the input sequence length
    "feature_num": 7, # the number of input features
    "hidden_size": 64, # hidden size in LSTM model
    "num_layers": 2, # the Layer number of LSTM model
    "loss_type": "huber",   # which loss to use (l1 / l2 / huber)
    "dropout": 0.1,
    "bidirectional": True,
    "output_size": 128, # the output size, [delta_x, delta_y] for next timestamp

    # ---------------- para for MDN --------------------
    "input_features": 128,
    "num_gaussians": 10,
    "out_features": 1,


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
