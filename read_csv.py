from asyncore import read
from datetime import time
from csv import reader
import json

# NOTE: Before using:
# Save 'data' & 'recording_codes' pages from excel sheet as individual csv files
# will output 'all_data.json' that can be used with 'CowsWater' class in model.ipynb

data_csv = 'data/data.csv'
rec_csv = 'data/recording_codes.csv'

# get all lines in a given csv except the first (header)
def read_data(filename=str):

    lines = []

    with open(filename, 'r') as f:
        # read the opened file with csv
        file_reader = reader(f)

        # skip the first line of the csv if the first line is a header
        head = next(file_reader)

        if file_reader is not None:
            for r in file_reader:
                lines.append(r)

    return lines

# fixes single digits in the times for conversion to datetime
def fix_single_digit(s=str):
    l = s.split(':')

    new_l = []
    for i in l:
        if len(i) < 2:
            i = '0'+i

        new_l.append(i)

    out = ':'.join(new_l)

    return out

# sorts the data from the given list of strings into a dict representing the cow labels
def get_cows_data(lines=list):

    cows_data = []
    for l in lines:
        
        if(len(l) > 9):
            struct = {
                'DATE': l[0],
                'PEN': l[1],
                'TAG': l[2],
                'START': fix_single_digit(l[3]),
                'END': fix_single_digit(l[4]),
                'TIME': fix_single_digit(l[5]),
                'NOSE_RUB': True if l[7] == '1' else False,
                'LAP': True if l[8] == '1' else False,
                'IDLE': True if l[9] == '1' else False,
            }

            cows_data.append(struct)

    return cows_data

# sorts the data from the given list of strings into a dict representing the recording codes
def get_rec_codes(lines=list):

    videos = []
    for l in lines:
        
        if(len(l) > 5):
            struct = {
                'DATE': l[0],
                'PENS': l[1].split('/'),
                'START': fix_single_digit(l[2]),
                'END': fix_single_digit(l[3]),
                'FILE_NAME': l[4],
                'IGNORE': []
            }

            for i in l[6].split('/'):
                if len(l[6]) > 0:
                    struct['IGNORE'].append(i.split('-'))

            videos.append(struct)

    return videos

# gets the necessary data from the csv files

def main():

    # read the lines from this file
    lines = read_data(data_csv)
    rec_lines = read_data(rec_csv)

    # sort the data into a list of dicts
    cows_data = get_cows_data(lines)

    # get the rec codes data
    rec_codes = get_rec_codes(rec_lines)

    # list of data points that have an associated file
    good_data = []

    # associating observations at certain times in certain pens with specifi videos
    for dp in cows_data:
        for v in rec_codes:
            if dp['DATE'] == v['DATE'] and dp['PEN'] in v['PENS'] and time.fromisoformat(dp['START']) > time.fromisoformat(v['START']) and time.fromisoformat(dp['END']) < time.fromisoformat(v['END']):
                dp['VIDEO'] = v
                good_data.append(dp)

    # write the final struct of data associated with videos to a json file to use later
    with open('all_data.json', 'w') as out:
        out.write(json.dumps(good_data, indent=4))

if __name__ == "__main__":
    main()