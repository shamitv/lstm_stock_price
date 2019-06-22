import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

def get_data_dir():
    return '../data/'


def merged_file_path():
    return get_data_dir()+'Merged_Data_Excel.xlsx'


def train_file_path():
    return get_data_dir()+'train_data.pkl'


def scaler_file_path():
    return get_data_dir()+'scaler.pkl'


def model_file_path():
    return get_data_dir()+'model-epoch-{epoch:02d}-val_loss-{val_loss:.2f}.h5'


df = pd.read_excel(merged_file_path(),index_col=0,parse_dates=True )
scaler = MinMaxScaler()
#scaled = scaler.fit_transform(df)

scaled = df.as_matrix()

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

X_train=np.asarray(X_list)
Y_train=np.asarray(Y_list)

print(X_train.shape)
print(Y_train.shape)


num_features=X_train.shape[2]

print(num_features)


model = Sequential()
model.add(LSTM(100, input_shape=(timesteps, num_features)))
model.add(Dropout(0.5))
model.add(Dense(20,activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

checkpoint = ModelCheckpoint(model_file_path(), monitor='val_loss',
                             verbose=1, save_best_only=True, mode='min')
tboard = TensorBoard('./logs')
model.fit(X_train, Y_train, epochs=500, batch_size=32,
          validation_split=0.2,callbacks=[checkpoint,tboard])


joblib.dump(scaler, scaler_file_path())
model.save(model_file_path())