#! /usr/local/bin/python3


import decisionTreeSynthetic
import syntheticDataParser as parser
import matplotlib.pyplot as plt
from anytree import Node, RenderTree, Resolver
import numpy as np

testData = parser.syntheticDataSet("synthetic-1.csv", 2, 5)
testData.discretizeFeatures()
myTree = decisionTreeSynthetic.decisionTree(2, 5, "synthetic-1.csv")

correct = 0
total = testData.data.shape[0]
for dataPoint in testData.data:
    if dataPoint[2] == myTree.decisionTreeModel[int(dataPoint[0])-1][int(dataPoint[1])-1]:
        correct = correct + 1
print("Accuracy: ")
print(float(correct/total))
myTree.printTree()
