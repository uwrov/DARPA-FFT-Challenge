import pygrib

FILE = './datasets/exmpl.grib2'
PARAM_NAME = "Direction of wind waves"

def read_grib_file():
    grbs=pygrib.open(FILE)

    grbs.seek(0)
    return grbs.select(name=PARAM_NAME)[0]
