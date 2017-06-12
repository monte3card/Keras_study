from __future__ import print_function

import numpy as np
np.random.seed(1337)


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.optimizers import SGD

# 设置batch的大小
batch_size = 128
# 设置类别的个数
nb_classes = 10
# 设置迭代的次数
nb_epoch = 20

# keras中的mnist数据集已经被划分成了60,000个训练集，10,000个测试集的形式，按以下格式调用即可
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train原本是一个60000*28*28的三维向量，将其转换为60000*784的二维向量
X_train = X_train.reshape(60000, 784)
# X_test原本是一个10000*28*28的三维向量，将其转换为10000*784的二维向量
X_test = X_test.reshape(10000, 784)
# 将X_train, X_test的数据格式转为float32存储
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 归一化
X_train /= 255
X_test /= 255
# 打印出训练集和测试集的信息
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')



Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# 建立顺序型模型
model1 = Sequential()

model1.add(Dense(256, activation='relu', input_dim=784))
model1.add(Dropout(0.2))
model1.add(Dense(256, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(10, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model1.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history1 = model1.fit(X_train, Y_train,
                    batch_size = batch_size,
                    epochs = nb_epoch,
                    verbose = 2,
                    validation_data = (X_test, Y_test))

model2 = Sequential()

model2.add(Dense(256, activation='relu', input_dim=784))

model2.add(Dense(256, activation='relu'))

model2.add(Dense(10, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model2.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history2 = model2.fit(X_train, Y_train,
                    batch_size = batch_size,
                    epochs = nb_epoch,
                    verbose = 2,
                    validation_data = (X_test, Y_test))
model3 = Sequential()

model3.add(Dense(256, activation='relu', input_dim=784))

model3.add(Dense(10, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model3.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history3 = model3.fit(X_train, Y_train,
                    batch_size = batch_size,
                    epochs = nb_epoch,
                    verbose = 2,
                    validation_data = (X_test, Y_test))


import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history1.history['acc'])
plt.plot(history1.history['val_acc'])
plt.title('model1 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('model2 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history3.history['acc'])
plt.plot(history3.history['val_acc'])
plt.title('model3 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history1.history['val_acc'])
plt.plot(history2.history['val_acc'])
plt.plot(history3.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['model1', 'model2', 'model3'], loc='upper left')
plt.show()



# summarize history for loss
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model2 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.title('model3 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()