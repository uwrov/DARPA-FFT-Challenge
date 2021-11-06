import numpy as np
'''
Process:
1. take an x_i and x_i-1
2. get parametric equation from t=0 to t=1
3. figure out force vector between the two
4. take points around the parametric in grid points
5. apply a rbf method to create
'''

TIME_ADJUSTMENT = 0

(MIN_LAT, MAX_LAT) = (10, 40)
(MIN_LONG, MAX_LONG) = (260, 350)

EPSILON = 0.1

class VectorField:
    def __init__(self, precision = 0, radius = 2):
        self.precision = precision
        self.radius = radius
        self.data = None
        self.divider = 5/10**self.precision

    def train(self, data):
        '''
            data should be formatted in the form of (x, y, t) = v
        '''
        xy_hash = dict()

        count = dict()
        for x in np.arange(MIN_LAT, MAX_LONG, self.divider):
            for y in np.arange(MIN_LONG, MAX_LONG, self.divider):
                xy_hash[(x, y)] = dict()
        for key in data:
            (x, y, t) = key
            v = data[key]
            x_round = (x // self.divider)
            y_round = (y // self.divider)
            (min_x, max_x) = ((x_round - self.radius) * self.divider,
                                (x_round + self.radius) * self.divider)
            (min_y, max_y) = ((y_round - self.radius) * self.divider,
                                (y_round + self.radius) * self.divider)
            for i in np.arange(min_x, max_x, self.divider):
                for j in np.arange(min_y, max_y, self.divider):
                    dist = ((x-i)**2 + (y-j)**2)**0.5
                    vel = v * (np.exp(-(EPSILON*dist)**2))      #gaussian rbf
                    if t in xy_hash[(i, j)]:
                        xy_hash[(i, j)][t] += vel
                        if (i,j,t) in count: count[(i,j,t)] += 1
                        else: count[(i,j,t)] = 2
                    else:
                        xy_hash[(i, j)][t] = vel

            for (i,j,t) in count:
                xy_hash[(i, j)][t] /= count[(i,j,t)]

        self.data = xy_hash

    def predict(self, x, y, t):
        if(self.data is None):
            raise Exception("model has not been trained yet")
        if(x >= MIN_LAT and x < MAX_LAT and
            y >= MIN_LONG and y < MAX_LONG):
            x_round = (x // self.divider) * self.divider
            y_round = (y // self.divider) * self.divider
            return self.data[(x_round, y_round)]
        else:
            raise ValueError("parameter out of bounds")
