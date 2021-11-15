import pygrib
from datetime import datetime
from model import GribVectorField

FILES = {
    datetime(2021,11,5).timestamp(): './datasets/grib/20211105/gfswave.t00z.global.0p16.f000.grib2',
    datetime(2021,11,5,3).timestamp(): './datasets/grib/20211105/gfswave.t00z.global.0p16.f003.grib2',
    datetime(2021,11,5,6).timestamp(): './datasets/grib/20211105/gfswave.t00z.global.0p16.f006.grib2',
    datetime(2021,11,5,9).timestamp(): './datasets/grib/20211105/gfswave.t00z.global.0p16.f009.grib2',
    datetime(2021,11,5,12).timestamp(): './datasets/grib/20211105/gfswave.t00z.global.0p16.f012.grib2',
    datetime(2021,11,5,15).timestamp(): './datasets/grib/20211105/gfswave.t00z.global.0p16.f015.grib2',
    datetime(2021,11,5,18).timestamp(): './datasets/grib/20211105/gfswave.t00z.global.0p16.f018.grib2',
    datetime(2021,11,5,21).timestamp(): './datasets/grib/20211105/gfswave.t00z.global.0p16.f021.grib2',
    datetime(2021,11,6).timestamp(): './datasets/grib/20211106/gfswave.t00z.global.0p16.f000.grib2',
    datetime(2021,11,6,3).timestamp(): './datasets/grib/20211106/gfswave.t00z.global.0p16.f003.grib2',
    datetime(2021,11,6,6).timestamp(): './datasets/grib/20211106/gfswave.t00z.global.0p16.f006.grib2',
    datetime(2021,11,6,9).timestamp(): './datasets/grib/20211106/gfswave.t00z.global.0p16.f009.grib2',
    datetime(2021,11,6,12).timestamp(): './datasets/grib/20211106/gfswave.t00z.global.0p16.f012.grib2',
    datetime(2021,11,6,15).timestamp(): './datasets/grib/20211106/gfswave.t00z.global.0p16.f015.grib2',
    datetime(2021,11,6,18).timestamp(): './datasets/grib/20211106/gfswave.t00z.global.0p16.f018.grib2',
    datetime(2021,11,6,21).timestamp(): './datasets/grib/20211106/gfswave.t00z.global.0p16.f021.grib2',
    }

PARAM_NAME = "Direction of wind waves"

models = dict()

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
    time = round_time(t)
    grib_vf = None
    if time not in models.keys():
        models[time] = GribVectorField(read_grib_file(time))

    grib_vf = models[time]

    return grib_vf.get_field(long, lat, size)
