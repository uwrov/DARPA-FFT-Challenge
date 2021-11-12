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

(MIN_LAT, MAX_LAT) = (10, 50)
(MIN_LONG, MAX_LONG) = (260, 350)

EPSILON = 0.1

#linear equation
def func(data, a1, b):
    t = data

    return a1*t + b

class VectorFunction:
    def __init__(self, x_param, y_param):
        self.x_param = x_param
        self.y_param = y_param

    def predict(self, t):
        return (func(t, x_param[0], x_param[1]),
                func(t, y_param[0], y_param[1]))


class VectorField:
    def __init__(self, precision = 0, radius = 10):
        self.precision = precision
        self.radius = radius
        self.data = None                #### data for velocity according to (x, y): {t:vel,...}
        self.funcs = None
        self.divider = 5/10**self.precision

    def train(self, data):
        '''
            data should be formatted in the form of (x, y, t) = v
        '''
        xy_hash = dict()

        count = dict()
        for x in np.arange(MIN_LAT, MAX_LONG + self.divider, self.divider):
            for y in np.arange(MIN_LONG, MAX_LONG + self.divider, self.divider):
                xy_hash[(x, y)] = dict()

        key_length = len(data)
        key_count = 0
        print("there are ", key_length, " datasets to train")
        for key in data:
            if key_count % 100 == 0:
                print(key_count, ":", "progress is at", (key_count/key_length)*100, "%.")
            key_count += 1

            (x, y, t) = key
            v = data[key]
            x_round = (x // self.divider)
            y_round = (y // self.divider)
            (min_x, max_x) = ((x_round - self.radius) * self.divider,
                                (x_round + self.radius) * self.divider)
            (min_y, max_y) = ((y_round - self.radius) * self.divider,
                                (y_round + self.radius) * self.divider)
            min_x = max(min_x, MIN_LAT)
            max_x = min(max_x, MAX_LAT)
            min_y = max(min_y, MIN_LONG)
            max_y = min(max_y, MAX_LONG)

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

        #self.funcs = self._fit_x_y(self.data)

    def _fit_x_y(self, data):
        funcs = dict()
        key_length = len(data)
        key_count = 0
        print("Fitting ", key_length, " functions")
        for (x, y) in data:
            if key_count % 100 == 0:
                print(key_count, ":", "progress is at", (key_count/key_length)*100, "%.")
            key_count += 1
            initParams = [1.0, 1.0] # these are the same as scipy default values in this example
            vel_dat = data[(x, y)]
            if len(vel_dat) > 1:
                t = vel_dat.keys()
                x = list(vel_dat.values())[:,0]
                y = list(vel_dat.values())[:,1]
                x_param, pcov = scipy.optimize.curve_fit(func, t, x, p0=initParams)
                y_param, pcov = scipy.optimize.curve_fit(func, t, y, p0=initParams)
                funcs[(x,y)] = VectorFunction(x_param, y_param)
            else:
                funcs[(x,y)] = VectorFunction([0, 0], [0, 0])

        return funcs


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


    def func_predict(self, x, y, t):
        if(self.data is None):
            raise Exception("model has not been trained yet")
        if(x >= MIN_LAT and x < MAX_LAT and
            y >= MIN_LONG and y < MAX_LONG):
            x_round = (x // self.divider) * self.divider
            y_round = (y // self.divider) * self.divider
            return self.funcs[(x_round, y_round)].predict(t)
        else:
            raise ValueError("parameter out of bounds")
