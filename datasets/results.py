import csv
from datetime import datetime

ACCURACY = 5

FIELD_NAMES = ['spotterId', 'Nov24lat', 'Nov24lon', 'Nov26lat',
    'Nov26lon', 'Nov28lat', 'Nov28lon', 'Nov30lat', 'Nov30lon', 'Dec2lat', 'Dec2lon']

DATES = ['Nov24','Nov26','Nov28','Nov30','Dec2']

TIMESTAMPS = {
    datetime(2021, 11, 24, 17).timestamp(): 'Nov24',
    datetime(2021, 11, 26, 17).timestamp(): 'Nov26',
    datetime(2021, 11, 28, 17).timestamp(): 'Nov28',
    datetime(2021, 11, 30, 17).timestamp(): 'Nov30',
    datetime(2021, 12, 2, 17).timestamp(): 'Dec2'
}

def write_results(data):
    '''
        dict[spotid] (size m) = list of size(N, 3)
            - m number of spots,
            - N number of predictions
            - 3 [timestamp, longitude, latitude]
    '''
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELD_NAMES)
        writer.writeheader()

        for spot in data.keys():
            closest_ts = {date: float('inf') for date in DATES}
            row = {field: None for field in FIELD_NAMES}
            row['spotterId'] = spot
            predictions = data[spot]
            for predict in predictions:
                (t_s, lat, lon) = predict
                for anchor_time in TIMESTAMPS.keys():
                    date = TIMESTAMPS[anchor_time]
                    dt = abs(anchor_time - t_s)
                    if dt < closest_ts[date]:
                        closest_ts[date] = dt
                        row[date+'lon'] = round(lon, ACCURACY)
                        row[date+'lat'] = round(lat, ACCURACY)
            writer.writerow(row)
