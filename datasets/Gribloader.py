import pygrib
from datetime import datetime
from model import GribVectorField

import os
import sys

FILE_DIR = "./datasets/grib"

FILES = {}
PARAM_NAME = "Primary wave direction" # "Direction of wind waves" # 

files_read = False

models = dict()

def read_file_paths():
    global files_read, FILES

    walk_dir = FILE_DIR

    for root, subdirs, files in os.walk(walk_dir):
        dirs = root.split('b')
        dir = dirs[-1][1:]
        if len(dir) == 8:
            year = int(dir[0:4])
            month = int(dir[4:6])
            day = int(dir[6:8])

            for filename in files:
                info = filename.split(".")
                s_t = int(info[1].strip("tz"))
                f_t = int(info[4].lstrip("f"))
                hour = s_t + f_t
                file_path = os.path.join(root, filename)
                dt = datetime(year, month, day, hour).timestamp()
                FILES[dt] = file_path
    files_read = True


def round_time(t):
    times = FILES.keys()
    return min(times, key=lambda x:abs(x-t))


def read_grib_file(t):
    try:
        grbs=pygrib.open(FILES[t])
    except FileNotFoundError:
        print("File for", t, "does not exist!")
        raise FileNotFoundError("Make sure to include file")

    grbs.seek(0)
    return grbs.select(name=PARAM_NAME)[0]


def get_field(long, lat, timestamp, size=5):
    if files_read is not True:
        read_file_paths()
    time = round_time(timestamp)
    grib_vf = None
    if time not in models.keys():
        models[time] = GribVectorField(read_grib_file(time))

    grib_vf = models[time]

    return grib_vf.get_field(long, lat, size)
