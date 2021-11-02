import numpy as np
import matplotlib.pyplot as plt

from datasets import read_file, get_path


class SpatialHash:
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.data = dict()
        self.min_x = float('inf')
        self.max_x = float('-inf')
        self.min_y = float('inf')
        self.max_y = float('-inf')

    def update_bounds(self, coords):
        (x, y) = coords
        if x < self.min_x: self.min_x = x
        if x > self.max_x: self.max_x = x
        if y < self.min_y: self.min_y = y
        if y > self.max_y: self.max_y = y


def add_vector_to_hash(p1, p2, hash, weight=1):
    (x1, y1) = p1
    (x2, y2) = p2
    (lx, ux) = (int(min(x1, x2) // hash.width),
                int(max(x1, x2) // hash.width))
    (ly, uy) = (int(min(y1, y2) // hash.height),
                int(max(y1, y2) // hash.height))

    v = np.array([x2 - x1, y2 - y1])

    for temp_x in range(lx , ux + 1):
        for temp_y in range(ly , uy + 1):
            ind = (temp_x, temp_y)
            if ind in hash.data:
                hash.data[ind] = (hash.data[ind] + v) / 2
            else:
                hash.update_bounds(ind)
                hash.data[ind] = v


def visualize_spatial_hash(hash):
    (min_x, max_x) = (hash.min_x * hash.width, hash.max_x * hash.width)
    (min_y, max_y) = (hash.min_y * hash.height, hash.max_y * hash.height)

    num_col = hash.max_x - hash.min_x + 1
    num_row = hash.max_y - hash.min_y + 1
    x, y = np.meshgrid(
            np.linspace(min_x, max_x, num_col),
            np.linspace(min_y, max_y, num_row))

    u = np.zeros(x.shape)
    v = np.zeros(y.shape)
    for key in hash.data.keys():
        (tx, ty) = key
        u[ty - hash.min_y, tx - hash.min_x] = hash.data[key][0]
        v[ty - hash.min_y, tx - hash.min_x] = hash.data[key][1]

    plt.quiver(x, y, u, v)
    plt.show()


def test():
    index, data = read_file()
    paths = get_path(data)

    hash = SpatialHash(0.1, 0.1)

    prev_val = None

    for i in range(10):
        path = paths[i]
        for x in path['path']:
            if prev_val is not None:
                add_vector_to_hash(prev_val, x, hash, 0.5)
            else:
                prev_val = x

    visualize_spatial_hash(hash)


if __name__ == "__main__":
    test()
