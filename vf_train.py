import numpy as np
import matplotlib.pyplot as plt

from datasets import read_file, get_path
from model import VectorField

def get_velocity(p1, p2, t1, t2):
    (x1, y1) = p1
    (x2, y2) = p2
    dt = t2 - t1

    return np.array([x2 - x1, y2 - y1]) * dt


def train():
    index, data = read_file()
    paths = get_path(data)

    vf_data = dict()

    for i in range(10):
        path = paths[i]
        length = len(path["timestamp"])
        prev_p = path["path"][0]
        prev_t = path["timestamp"][0]
        for i in range(length):
            point = path["path"][i]
            t = path["timestamp"][i]
            v = get_velocity(prev_p, point, prev_t, t)
            (x, y) = point
            vf_data[(x, y, t)] = v
            prev_p = point
            prev_t = t

    vf_model = VectorField()
    vf_model.train(vf_data)

    print(vf_model.predict(26, 310, 0))



if __name__ == "__main__":
    train()
