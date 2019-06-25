import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from tensorflow.keras import Sequential
from tensorflow.keras.layers  import LSTM
from tensorflow.keras.layers  import Dropout
from tensorflow.keras.layers  import Dense
from tensorflow.keras.layers  import Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import TerminateOnNaN
from tensorflow.keras.callbacks import Callback


def get_data_dir():
    return '../data/'


def merged_file_path():
    return get_data_dir()+'Merged_Data_Excel.xlsx'


def train_file_path():
    return get_data_dir()+'train_data.pkl'


def scaler_file_path():
    return get_data_dir()+'scaler.pkl'


def model_file_path():
    return get_data_dir()+'models/'+'model-v4-epoch-{epoch:02d}-val_loss-{val_loss:.2f}.h5'


class CustomCallback(Callback):
    def __init__(self,epoch_test):
        self.X_epoch_test=epoch_test
    def on_epoch_end(self, epoch, logs=None):
        print("epoch done")
        y=self.model.predict(self.X_epoch_test)
        with np.printoptions(precision=15, suppress=True):
            print(y)

df = pd.read_excel(merged_file_path(),index_col=0,parse_dates=True )

df.ffill(inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df)

#scaled = df.as_matrix()

timesteps=60
rowcount = scaled.shape[0]
close_price_col_idx=0

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
    x_last_row = x[timesteps-1]
    X_list.append(x)

X=np.asarray(X_list)
Y=np.asarray(Y_list)

print(X.shape)
print(Y.shape)

X_epoch_test = X[:20]
Y_epoch_test = Y[:20]

num_features=X.shape[2]

print(num_features)


model = Sequential()

model.add(LSTM(100, input_shape=(timesteps, num_features), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
#model.add(Dense(20,activation='relu'))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

checkpoint = ModelCheckpoint(model_file_path(), monitor='val_loss',
                             save_best_only=True, verbose=1, mode='min')
tboard = TensorBoard('.\\logs\\v4\\',histogram_freq=5)
nanTerm = TerminateOnNaN()
customCallback=CustomCallback(X_epoch_test)
joblib.dump(scaler, scaler_file_path())

model.fit(X, Y, epochs=500, batch_size=128,
          validation_split=0.2,verbose=1,
          callbacks=[checkpoint,nanTerm,tboard])
