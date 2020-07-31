
import numpy as np
import csv
from random import randrange

from naive_bayes_final import NaiveBayes


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def train_test_split(dataset, split=0.70):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy

x_data = []
y_data = []
dataset = []

file = open('weight-height.csv')
csv_reader = csv.reader(file)
next(csv_reader)

for row in csv_reader:
    dataset.append(row)
    x_data.append(row[1:])
    y_data.append(row[0])

x_data = np.array(x_data)
y_data = np.array(y_data)

y_data = np.where(y_data=='Male', 0, y_data)
y_data = np.where(y_data=='Female', 1, y_data)

x_data = x_data.astype(np.float64)
y_data = y_data.astype(np.int64)

train, test = train_test_split(dataset)

x_train = []
y_train = []
x_test = []
y_test = []

for row in train:
    x_train.append(row[1:])
    y_train.append(row[0])

for row in test:
    x_test.append(row[1:])
    y_test.append(row[0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

y_train = np.where(y_train=='Male', 0, y_train)
y_train = np.where(y_train=='Female', 1, y_train)
y_test = np.where(y_test=='Male', 0, y_test)
y_test = np.where(y_test=='Female', 1, y_test)

x_test = x_test.astype(np.float64)
y_test = y_test.astype(np.int64)
x_train = x_train.astype(np.float64)
y_train = y_train.astype(np.int64)

nb = NaiveBayes()
nb.fit(x_train, y_train)
predictions = nb.predict(x_test)

print("Naive Bayes classification accuracy", accuracy(y_test, predictions))






