import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from datasets import read_file, get_path
from model import VectorField

TIME_SCALE = 36
(MIN_LAT, MAX_LAT) = (20, 40)
(MIN_LONG, MAX_LONG) = (280, 320)

anim_index = 0

def get_velocity(p1, p2, t1, t2):
    (x1, y1) = p1
    (x2, y2) = p2
    dt = (t2 - t1) / TIME_SCALE

    return np.array([x2 - x1, y2 - y1]) * dt

def visualize_sample_model(vf_model):
    x, y = np.meshgrid(
            np.linspace(MIN_LAT, MAX_LAT, 21),
            np.linspace(MIN_LONG, MAX_LONG, 21))

    u = np.zeros(x.shape)
    v = np.zeros(y.shape)

    for i in range(21):
        for j in range(21):
            vel_list = vf_model.predict(x[i, j], y[i, j], 0)
            if len(vel_list) > 0:
                (temp_x, temp_y) = list(vel_list.values())[0]
                u[i, j] = temp_x
                v[i, j] = temp_y

    fig, ax = plt.subplots(1,1)

    Q = ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1)
    anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, x, y, vf_model),
                                    interval=50, blit=False)
    fig.tight_layout()
    plt.show()


def update_quiver(num, Q, X, Y, vf_model):
    U = np.zeros(X.shape)
    V = np.zeros(Y.shape)

    for i in range(21):
        for j in range(21):
            vel_list = vf_model.predict(X[i, j], Y[i, j], 0)
            if len(vel_list) > 0:
                (temp_x, temp_y) = list(vel_list.values())[num % (len(vel_list) - 1)]
                U[i, j] = temp_x
                V[i, j] = temp_y

    Q.set_UVC(U,V)

    return Q


def train():
    index, data = read_file()
    paths = get_path(data)

    vf_data = dict()

    for i in range(20): #len(paths)):
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

    visualize_sample_model(vf_model)


if __name__ == "__main__":
    train()
