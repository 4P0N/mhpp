import pandas as pd  
import numpy as np 
from math import radians, sin, cos, acos
import tqdm

def dist(slat, slon, elat, elon):
    #print("Input coordinates of two points:")
    slat = radians(slat)
    slon = radians(slon)
    elat = radians(elat)
    elon = radians(elon)

    return 6371.01 * acos(min(sin(slat)*sin(elat) + cos(slat)*cos(elat)*cos(slon - elon), 1))

f_house = 'Data Tables Step 5 - Attributes\property.csv'
f_region = 'Data Tables Step 5 - Attributes\\region.csv'
g_h_r = 'Data Tables Step 5 - Graph\house-region.csv'

f_region_target = 'Data Tables Step 5 - Attributes\\region_modified.csv'
g_h_r_target = 'Data Tables Step 5 - Graph\house-region_modified.csv'

df_house = pd.read_csv(f_house)
df_region = pd.read_csv(f_region)
df_graph = pd.read_csv(g_h_r)

regions = set(df_graph['id2'])
total_regions = len(regions)
region_count = 0

for region in regions:
    region_count += 1
    if region_count%300 == 0:
        print(str(region_count)+ " / "+ str(total_regions))
    #take all the rows with this region
    rows_with_region = df_graph[df_graph['id2'] == region]
    houses_this_region = rows_with_region['id1'].values

    #take those house rows
    houses_rows = df_house[np.isin(df_house['ID'].values, houses_this_region)]
    region_lat = np.mean(houses_rows['Lat'])
    region_lng = np.mean(houses_rows['Lng'])
    #change region attribute
    mask = (df_region['ID'] == region)
    df_region.loc[mask, 'Lat'] = region_lat
    df_region.loc[mask, 'Lng'] = region_lng
    #change the weights of the graph

    for k in range(len(rows_with_region)):
        prop = houses_rows[houses_rows['ID'] == rows_with_region.iat[k, 0]]
        rows_with_region.iat[k, 2] = dist(region_lat, region_lng, prop['Lat'].values, prop['Lng'].values) + 0.1
        mask1 = (df_graph['id1'] == rows_with_region.iat[k, 0])
        df_graph.loc[mask1, 'weight'] = rows_with_region.iat[k, 2]

df_region.to_csv(f_region_target)
df_graph.to_csv(g_h_r_target)
