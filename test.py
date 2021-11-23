from datasets import read_file, get_path, get_field, write_results
from datetime import datetime
import random
import time
import pickle

TOTAL = 10000
SPOT_COUNT = 100

PICKLE_FILE = 'prediction2.pkl'
def test_results():
    data = dict()
    start = time.perf_counter()
    for s in range(SPOT_COUNT):
        pred_list = list()
        for i in range(TOTAL):
            long = random.uniform(0, 360)
            lat = random.uniform(0, 50)
            day = random.randint(1, 30)
            hour = random.randint(0, 23)
            pred_list.append((datetime(2021,11,day,hour).timestamp(), long, lat))
        data[f'spot-{s}'] = pred_list

    write_results(data)


def generate_results():
    with open(PICKLE_FILE, 'rb') as f:
        data = pickle.load(f)
        write_results(data)


def main():
    #print(get_field(37.12, 40, datetime(2021,11,5,2).timestamp()))
    #print(get_field(37.12, 40, datetime(2021,11,5,6).timestamp()))
    #print(get_field(37.12, 40, datetime(2021,11,5,9).timestamp()))

    dat = list()
    start = time.perf_counter()
    for i in range(TOTAL):
        if (i % 100 == 0):
            print("Percent Finished: ", i/TOTAL*100, "% calculated.")
        long = random.randint(0, 360)
        lat = random.randint(0, 50)
        day = random.randint(20, 30)
        hour = random.randint(0, 23)
        f = get_field(long, lat, datetime(2021,11,day,hour).timestamp())
        dat.append(f)
    end = time.perf_counter()
    print(f"Queried 10000 results in {end - start:0.4f} seconds");


if __name__ == '__main__':
    #test_results()
    #main()
    generate_results()
