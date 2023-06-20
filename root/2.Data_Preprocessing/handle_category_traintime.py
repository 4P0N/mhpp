import pandas as pd 
import os


house = 'Data Tables Step 3\Property_full_table_raw.csv'
school = 'Data Tables Step 3\\7 - School .csv'
train_basic = 'Data Tables Step 3\\10 - Train_station.csv'
train_times = 'Data Tables Step 3\\11 - Train_Time.csv'

def main():
    #handle_house()
    #handle_school()
    handle_train()


def handle_school():
    df = pd.read_csv(school)
    df_new = pd.get_dummies(df,columns=['gender','restrictedZone','type'])
    df_new.to_csv('Data Tables Step 4\\7 - School .csv')

def handle_house():
    df = pd.read_csv(house)

    df.drop('Locality', axis=1, inplace=True)
    df.drop('Postal Code', axis=1, inplace=True)

    df = pd.get_dummies(df, columns=['proType'])
    
    df.to_csv('Data Tables Step 4\Property_full_table_raw.csv')

def handle_train():
    df_main = pd.read_csv(train_basic)
    df_time = pd.read_csv(train_times)

    time = []

    for k in range(0, 218, 1):
        time.append(df_time[ df_time['id2'] == k ] ['avg_time'].mean())

    print(time)
    df_main = df_main.assign(avg_time=pd.Series(time))
    df_main.to_csv('Data Tables Step 4\\Train_station.csv')


if __name__ == '__main__':
    main()
