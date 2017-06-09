from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from math import sqrt
from matplotlib import pyplot
import numpy as np
import keras

def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

series = read_csv('wine.data', header=None, index_col= False)
series.head()
raw_values = series.values

x = raw_values[:, 1:14]
y = keras.utils.to_categorical((raw_values[:, 0]-1), num_classes=3)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler, train_x_scaled, test_x_scaled = scale(x_train, x_test)
model = Sequential()  #²ã´ÎÄ£ÐÍ

model.add(Dense(10, activation='sigmoid', input_dim=13))
model.add(Dense(3, activation='softmax'))
model.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
hist=model.fit(train_x_scaled, y_train,
          epochs=20,
          batch_size=10,validation_data=(test_x_scaled, y_test))
score = model.evaluate(test_x_scaled, y_test, batch_size=10)
print(score)
