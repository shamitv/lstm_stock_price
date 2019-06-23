import pandas as pd
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers  import LSTM
from tensorflow.keras.layers  import Dropout
from tensorflow.keras.layers  import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import TerminateOnNaN


def get_data_dir():
    return '../data/'


def merged_file_path():
    return get_data_dir()+'Merged_Data_Excel.xlsx'


def train_file_path():
    return get_data_dir()+'train_data.pkl'


def scaler_file_path():
    return get_data_dir()+'scaler.pkl'


def model_file_path():
    return get_data_dir()+'models/'+'model-epoch-{epoch:02d}-val_loss-{val_loss:.2f}.h5'


df = pd.read_excel(merged_file_path(),index_col=0,parse_dates=True )
#scaled = scaler.fit_transform(df)

scaled = df.as_matrix()

timesteps=600
rowcount = scaled.shape[0]
close_price_col_idx=3

X_list=[]
Y_list=[]


for i in range(timesteps,rowcount,1):
    target_row=scaled[i]
    y_val = target_row[close_price_col_idx]
    start_row=i-timesteps
    end_row=i

    Y_list.append(y_val)
    x=scaled[start_row:end_row]
    x_first_row=x[0]
    x_last_row = x[599]
    X_list.append(x)

X=np.asarray(X_list)
Y=np.asarray(Y_list)

print(X.shape)
print(Y.shape)


num_features=X.shape[2]

print(num_features)


model = Sequential()
model.add(LSTM(100, input_shape=(timesteps, num_features)))
model.add(Dropout(0.5))
model.add(Dense(20,activation='relu'))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

checkpoint = ModelCheckpoint(model_file_path(), monitor='val_loss',
                             verbose=1, save_best_only=True, mode='min')
tboard = TensorBoard('.\\logs\\')
nanTerm = TerminateOnNaN()
model.fit(X, Y, epochs=500, batch_size=32,
          validation_split=0.2,callbacks=[checkpoint,tboard,nanTerm])


model.save(model_file_path())