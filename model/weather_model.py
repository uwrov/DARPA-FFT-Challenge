import numpy as np
import pygrib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import time

SCALE = 360

class GribVectorField:
    def __init__(self, grb):
        self.grb = grb
        self.shape = self.grb.values.shape
        self.lat, self.long = grb.latlons()
        self.min_lat = self.lat.min()
        self.max_lat = self.lat.max()
        self.min_long = self.long.min()
        self.max_long = self.long.max()
        self.d_lat = (self.max_lat - self.min_lat) / self.shape[0]
        self.d_long = (self.max_long - self.min_long) / self.shape[1]
        self.vals = np.ma.filled(self.grb.values, -SCALE)
        ''' Visualizing data
        data=grb.values
        lat,lon = grb.latlons()
        m = Basemap(projection='mill',lat_ts=10,llcrnrlon=lon.min(), \
              urcrnrlon=lon.max(),llcrnrlat=lat.min(),urcrnrlat=lat.max(), \
              resolution='c')
        x, y = m(lon,lat)
        cs = m.pcolormesh(x,y,data,shading='flat',cmap=plt.cm.jet)
        m.drawcoastlines()
        m.fillcontinents()
        m.drawmapboundary()
        m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
        m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])

        tx, ty = m(324, 28)
        m.plot(tx, ty, marker='D')

        plt.colorbar(cs,orientation='vertical')
        plt.title('Example 2: NWW3 Significant Wave Height from GRiB')
        plt.show()
        #'''


    def get_field(self, long, lat, size):
        field = list()
        l_list = list()
        lat_ind = int(((self.max_lat - lat) // self.d_lat) - size // 2)
        long_ind = int(((long - self.min_long) // self.d_long) - size // 2)
        for i in range(lat_ind, lat_ind + size):
            #lat_list = list()
            #i_list = list()
            for j in range(long_ind, long_ind + size):
                if i >= 0 and j >= 0 and i < self.shape[0] and j < self.shape[1]:
                    field.append(self.vals[i, j] / SCALE)
                    l_list.append((self.long[i, j], self.lat[i, j]))
                else:
                    field.append(np.ma.masked)
            #field.append(lat_list)
            #l_list.append(i_list)
        return field, l_list
