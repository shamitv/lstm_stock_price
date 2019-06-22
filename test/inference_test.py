import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.externals import joblib


def get_data_dir():
    return '../data/'


def merged_file_path():
    return get_data_dir()+'Merged_Data_Excel_Test.xlsx'


def train_file_path():
    return get_data_dir()+'train_data.pkl'


def scaler_file_path():
    return get_data_dir()+'scaler.pkl'


def model_file_path():
    return get_data_dir()+'model_unscaled_v1.h5'


df = pd.read_excel(merged_file_path(),index_col=0,parse_dates=True )
scaler = joblib.load(scaler_file_path())

#scaled = scaler.transform(df)

scaled = df.as_matrix()

print("scaled")

timesteps=600
rowcount = scaled.shape[0]
close_price_col_idx=3


X_list=[]
Y_list=[]


for i in range(rowcount-1,timesteps,-1):
    target_row=scaled[i]
    y_val = target_row[close_price_col_idx]
    start_row=(i-1)-(timesteps)
    end_row=i-1

    Y_list.append(y_val)
    x=scaled[start_row:end_row]
    X_list.append(x)


X=np.asarray(X_list)
Y_hat=np.asarray(Y_list)

print("Data prepared")

model = load_model(model_file_path())

