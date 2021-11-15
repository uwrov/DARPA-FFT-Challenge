import numpy as np


class GribVectorField:
    def __init__(self, grb):
        self.grb = grb
        size = self.grb.values.shape
        lat, long = grb.latlons()
        self.min_lat = lat.min()
        self.max_lat = lat.max()
        self.min_long = long.min()
        self.max_long = long.max()
        self.d_lat = (self.max_lat - self.min_lat) / size[0]
        self.d_long = (self.max_long - self.min_long) / size[1]

    def get_field(self, long, lat, size):
        field = list()
        shape = self.grb.values.shape
        lat_ind = int(((lat - self.min_lat) // self.d_lat) - size // 2)
        long_ind = int(((long - self.min_long) // self.d_long) - size // 2)
        for i in range(lat_ind, lat_ind + size):
            lat_list = list()
            for j in range(long_ind, long_ind + size):
                if i >= 0 and j >= 0 and i < shape[0] and j < shape[1]:
                    lat_list.append(self.grb.values[i, j])
                else:
                    lat_list.append(np.ma.masked)
            field.append(lat_list)
        return field
