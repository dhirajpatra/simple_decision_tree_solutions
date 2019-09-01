"""
Created on Wed Aug 28 18:58:27 2019

@author: dhirajpatra
"""
from sklearn import tree
import pandas as pd

# read the csv of gender weight height
df = pd.read_csv('weight-height.csv', header=None, skiprows=1)
df = df.sample(frac=1).reset_index(drop=True)

# [weight, height]
# x = df.iloc[1:, 1:3]
# y = df.iloc[1:, 0:1]

x = df[[1, 2]].copy()
y = df[[0]].copy()

# call model
clf = tree.DecisionTreeClassifier()
# fit with model and data
clf.fit(x, y)

# taking input from user height and weight
height = float(input('Enter height to find out gender: '))
weight = float(input('Enter weight to find out gender: '))

# prediction for gender
prediction = clf.predict([[weight, height]])

# print the prediction
print('Gender is: %s' % prediction)
