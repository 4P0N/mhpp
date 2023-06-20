import pandas as pd 
import numpy as np
import argparse

from math import radians, sin, cos, acos


def main():
    
    print("Processing...")
    get_house_school()
    print("part 1 done!")
    get_school_train()
    print("part 2 done!")
    get_train_train()
    print("part 3 done!")


    print("Complete...")


def dist(slat, slon, elat, elon):
    #print("Input coordinates of two points:")
    slat = radians(slat)
    slon = radians(slon)
    elat = radians(elat)
    elon = radians(elon)

    return 6371.01 * acos(sin(slat)*sin(elat) + cos(slat)*cos(elat)*cos(slon - elon))
    #print("The distance is %.2fkm." % dist)

# i)Property- Train ii) Train-Train iii) Property-School iv) School-Train v)(done)Property-area

graph_house_file = 'Data Tables Step 4\\Property_full_table_raw.csv'
graph_train_file = 'Data Tables Step 4\\Train_Station.csv'
graph_train_time_file = 'Data Tables Step 4\\11 - Train_Time.csv'
graph_school_file = 'Data Tables Step 4\\7 - School .csv'



def get_house_school():
    LIMIT = 2
    df1 = pd.read_csv(graph_house_file)
    df2 = pd.read_csv(graph_school_file)

    id1 = df1['ID'].values
    lat1 = df1['Lat'].values
    lng1 = df1['Lng'].values


    id2 = df2['ID'].values
    lat2 = df2['Lat'].values
    lng2 = df2['Lng'].values


    final_csv = []

    for (k,l,m) in zip(id1, lat1, lng1):
        for (n,p,q) in zip (id2, lat2, lng2):
            d = dist(l,m,p,q)
            if d <= LIMIT:
                final_csv.append([k, n, d])

    np.set_printoptions(suppress=True)
    final_csv = np.array(final_csv)
    np.savetxt('Data Tables Step 5/property-school.csv', final_csv ,delimiter = ',', fmt= "%f", header="id1, id2, weight")


def get_school_train():
    
    df1 = pd.read_csv(graph_school_file)
    df2 = pd.read_csv(graph_train_file)

    id1 = df1['ID'].values
    lat1 = df1['Lat'].values
    lng1 = df1['Lng'].values


    id2 = df2['ID'].values
    lat2 = df2['Lat'].values
    lng2 = df2['Lng'].values


    final_csv = []

    for (k,l,m) in zip(id1, lat1, lng1):
        d_min = 999999999999999
        d_min_ind = -1
        for (n,p,q) in zip (id2, lat2, lng2):
            d = dist(l,m,p,q)
            if d < d_min:
                d_min = d
                d_min_ind = n

        final_csv.append([k, d_min_ind, d_min])

    np.set_printoptions(suppress=True)
    final_csv = np.array(final_csv)
    np.savetxt('Data Tables Step 5/school-train.csv', final_csv ,delimiter = ',', fmt= "%f", header="id1, id2, weight")


def get_train_train():
    
    df2 = pd.read_csv(graph_train_file)

    id2 = df2['ID'].values
    lat2 = df2['Lat'].values
    lng2 = df2['Lng'].values

    final_csv = []

    for (k,l,m) in zip(id2, lat2, lng2):
        lowest = 999999
        lowest_id = -1
        lowest_2 = 999999
        lowest_2_id = -1
        lowest_3 = 999999
        lowest_3_id = -1
        lowest_4 = 999999
        lowest_4_id = -1
        for (n,p,q) in zip (id2, lat2, lng2):
            if k == n:
                continue
            d = dist(l,m,p,q)

            if d < lowest:
                lowest_4, lowest_4_id = lowest_3, lowest_3_id
                lowest_3, lowest_3_id = lowest_2, lowest_2_id
                lowest_2, lowest_2_id = lowest, lowest_id
                lowest, lowest_id = d,n
            elif d<lowest_2:
                lowest_4, lowest_4_id = lowest_3, lowest_3_id
                lowest_3, lowest_3_id = lowest_2, lowest_2_id
                lowest_2, lowest_2_id = d,n
            elif d<lowest_3:
                lowest_4, lowest_4_id = lowest_3, lowest_3_id
                lowest_3, lowest_3_id = d,n
            elif d<lowest_4:
                lowest_4, lowest_4_id = d,n

        final_csv.append([k, lowest_id, lowest])
        final_csv.append([k, lowest_2_id, lowest_2])
        final_csv.append([k, lowest_3_id, lowest_3])
        final_csv.append([k, lowest_4_id, lowest_4])


    np.set_printoptions(suppress=True)
    final_csv = np.array(final_csv)
    np.savetxt('Data Tables Step 5/train-train.csv', final_csv ,delimiter = ',', fmt= "%f", header="id1, id2, weight")


if __name__ == '__main__':
    main()