import argparse
import os
import pandas as pd 

def main():
    parser = argparse.ArgumentParser(description="Please launch following command: \
        python reindex.py ")
    parser.add_argument('filename_in', type = str, help="python reindex.py 'xyz.csv'")
    parser.add_argument('start_location', type= int, help="from which index to start indexing")

    args = parser.parse_args()
    print("Processing...")
    next_loc = reindexdata(args.filename_in, args.start_location)
    print("Next assigned location: "+ str(next_loc))
    print("Complete...")

def reindexdata(fname, startlocation):
    id_old = []
    id_new = []
    name = os.path.basename(fname)

    df = pd.read_csv(fname)

    for k in df.index:
        id_old.append(df['ID'][k])
        id_new.append(startlocation)
        startlocation += 1

    mapping_dict = {'id_old': id_old, 'id_new': id_new}
    targetfname = 'Data Tables Step 2/' + 'index_map_' + name + '.csv'
    pd.DataFrame(mapping_dict).to_csv(targetfname)
    return startlocation


if __name__ == '__main__':
    main()