import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


dataset_path="C:/Users/Hiral/Desktop/DLNLP/A1/a1.1/housing.csv"

raw_dataset = pd.read_csv(dataset_path)
dataset=raw_dataset.copy()

y=dataset['medv']
x=dataset.loc[:, dataset.columns != 'medv']

xTrain, xTest, yTrain, yTest=train_test_split(x, y, test_size = 0.3,random_state=0)

#rain_dataset = dataset.sample(frac=0.8,random_state=0)
#sns.pairplot(xTrain[["crim", "zn", "indus", "chas"]], diag_kind="kde")
#plt.show()

statistics1=xTrain.describe().transpose()
statistics2=xTest.describe().transpose()

#print(statistics.loc[:,statistics.columns!='75%']) 
#print(xTrain)
norm_xTrain=(xTrain-statistics1['mean'])/statistics1['std']
norm_xTest=(xTest-statistics2['mean'])/statistics2['std']

def build_model1():
  model = keras.Sequential([
    layers.Dense(1, activation='linear', input_shape=[len(xTrain.keys())]),
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.1)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model1=build_model1()
#---------------------------------------------------------
def build_model2():
  model = keras.Sequential([
    layers.Dense(1, activation='linear', input_shape=[len(xTrain.keys())]),
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.01)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model2=build_model2()
#---------------------------------------------------------
def build_model3():
  model = keras.Sequential([
    layers.Dense(1, activation='linear', input_shape=[len(xTrain.keys())]),
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model3=build_model3()
#---------------------------------------------------------
def build_model4():
  model = keras.Sequential([
    layers.Dense(1, activation='linear', input_shape=[len(xTrain.keys())]),
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.0001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model4=build_model4()
#---------------------------------------------------------
def build_model5():
  model = keras.Sequential([
    layers.Dense(1, activation='linear', input_shape=[len(xTrain.keys())]),
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.00001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model5=build_model5()
#---------------------------------------------------------

#print(model.summary())

example_batch = norm_xTest[:10]
example_result = model1.predict(example_batch)
print(example_result)

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000
#---------------------------------------------------------
history1 = model1.fit(
    norm_xTrain, yTrain,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[PrintDot()])

hist1 = pd.DataFrame(history1.history)
hist1['epoch'] = history1.epoch
hist1.tail()
#---------------------------------------------------------
history2 = model2.fit(
    norm_xTrain, yTrain,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[PrintDot()])

hist2 = pd.DataFrame(history2.history)
hist2['epoch'] = history2.epoch
hist2.tail()
#---------------------------------------------------------
history3 = model3.fit(
    norm_xTrain, yTrain,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[PrintDot()])

hist3 = pd.DataFrame(history3.history)
hist3['epoch'] = history3.epoch
hist3.tail()
#---------------------------------------------------------
history4 = model4.fit(
    norm_xTrain, yTrain,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[PrintDot()])

hist4 = pd.DataFrame(history4.history)
hist4['epoch'] = history4.epoch
hist4.tail()
#---------------------------------------------------------
history5 = model5.fit(
    norm_xTrain, yTrain,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[PrintDot()])

hist5 = pd.DataFrame(history5.history)
hist5['epoch'] = history5.epoch
hist5.tail()
#---------------------------------------------------------
#print(hist)

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label = 'Val Error')
    plt.ylim([0,100])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label = 'Val Error')
    plt.ylim([0,1000])
    plt.legend()
    plt.show()


plot_history(history1)
plot_history(history2)
plot_history(history3)
plot_history(history4)
plot_history(history5)

test_predictions = model1.predict(norm_xTest).flatten()

plt.scatter(yTest, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

#print(yTrain)
#print(y)
