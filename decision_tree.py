#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:58:27 2019

@author: dhirajpatra
"""
from sklearn import tree

#[height, weight, shoe size]
x = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [190, 90, 47]]
y = ['male', 'female', 'female', 'male']

clf = tree.DecisionTreeClassifier()

clf.fit(x, y)

prediction = clf.predict([[160, 62, 38]])

print(prediction)
