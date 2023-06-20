import argparse
import os
import pandas as pd 

def main():
    parser = argparse.ArgumentParser(description="Please launch following command: \
        python reindex_data.py ")

    parser.add_argument('file1_location', type= str, help="on which file to change the indexes, choose from Data Tables Step 1")
    parser.add_argument('index_location', type = str, help="the file holding the old and new index")
    parser.add_argument('--index2_location', type = str, help="only to be given when the file is legitimately a graph", default=None)

    args = parser.parse_args()
    print("Processing...")
    reindexdataset(args.file1_location, args.index_location, args.index2_location)
    print("Complete...")


def reindexdataset(f_loc, i1_loc, i2_loc):
    
    df = pd.read_csv(f_loc)
    target_fname = os.path.basename(f_loc)
    df_i1 = pd.read_csv(i1_loc)
    
    if i2_loc == None:  
        for k in df.index:
            if any(df_i1['id_old'] == df['ID'][k]) == True:
                df['ID'][k] = df_i1[ df_i1['id_old'] == df['ID'][k] ]['id_new'].values[0]
            else:
                df.drop([k])

    else:
        df_i2 = pd.read_csv(i2_loc)
        for k in df.index:
            if any(df_i1['id_old'] == df['id1'][k]) == True and any( df_i2['id_old'] == df['id2'][k] ) == True:
                df['id1'][k] = df_i1[ df_i1['id_old'] == df['id1'][k] ]['id_new'].values[0]
                df['id2'][k] = df_i2[ df_i2['id_old'] == df['id2'][k] ]['id_new'].values[0]
            else:
                df.drop([k])

    df.to_csv("Data Tables Step 3/" + target_fname)


if __name__ == '__main__':
    main()