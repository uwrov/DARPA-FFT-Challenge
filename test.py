from datasets import read_file, get_path, get_field
from datetime import datetime

def main():
    print(get_field(37.12, 40, datetime(2021,11,5,2).timestamp()))


if __name__ == '__main__':
    main()
