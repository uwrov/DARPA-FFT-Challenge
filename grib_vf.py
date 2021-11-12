from datasets import read_grib_file
from model import GribVectorField

grb = read_grib_file()

model = GribVectorField(grb)

print(model.min_lat, model.max_lat, model.min_long, model.max_long)
