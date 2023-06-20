import argparse
import pandas as pd 
import os

def main():
    parser = argparse.ArgumentParser(description="Please launch following command: \
        python dataCleaner.py ")
    parser.add_argument('filename_in', type = str, help="python dataCleaner.py 'xyz.csv'")

    args = parser.parse_args()
    print("Processing...")
    cleandata(args.filename_in)
    print("Complete...")



def cleandata(file_path):
    f = pd.read_csv(file_path)
    print(f)
    fname = os.path.basename(file_path)
    f_mod = f.dropna()
    print("Number of rows changed from " +  str(f.shape[0]) + "to "+ str(f_mod.shape[0]))
    f_mod.to_csv('Data Tables Step 1/' + fname)


if __name__ == '__main__':
    main()