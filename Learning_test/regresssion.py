import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

raw_data = pd.read_csv('../lane_detection3/Labels/housing.csv')
# raw_data.info()

data = raw_data.copy()
# print(data.head())

# Sprawdzenie ile danych jest pustych
# print(data.isnull().sum() / len(data))

# Usunięcie pustych danych
data.dropna(inplace=True)
# print(data.isnull().sum() / len(data))

data.describe()

# Wyświetlenie elemnetu typu 'obiekt'
# print(data.describe(include=['object']))

# for col in data.columns:
#     print(col)
#
# print(data['median_house_value'])
# print(data.median_house_value)

# plt.hist(data.median_house_value, bins=100)
# plt.show()

# print(data.median_house_value.value_counts())

# Usunięcie za dużych wartości
index_to_drop = data[data.median_house_value == 500001.0].index
data = data.drop(index=index_to_drop)

# plt.hist(data.median_house_value, bins=100)
# plt.show()

data_dummies = pd.get_dummies(data, drop_first=True)
# print(data_dummies.head())

train = data_dummies.sample(frac=0.8, random_state=0)
test = data_dummies.drop(train.index)

print(f'Train lenght: {len(train)}')
print(f'Test lenght: {len(test)}')

train_stats = train.describe()
train_stats.pop('median_house_value')
train_stats = train_stats.transpose()

train_labels = train.pop('median_house_value')
test_labels = test.pop('median_house_value')

def norm(set):
    return (set - train_stats['mean']) / train_stats['std']

normed_train = norm(train)
normed_test = norm(test)

# print(normed_train.isnull().sum())
# print(normed_test.isnull().sum())

# DataFrame -> numpy array
normed_train = normed_train.values
normed_test = normed_test.values

# print(normed_train)
# print(normed_test)

print(len(train.keys()))
print(train.shape[1])

'''Model'''
def build_model():
    model = Sequential()
    model.add(Dense(units=1024, kernel_regularizer='l2', activation='relu', input_shape=[len(train.keys())]))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae', 'mse'])

    return model

model = build_model()
model.summary()

history = model.fit(normed_train, train_labels.values, epochs=150, validation_split=0.2, verbose=1, batch_size=12)
