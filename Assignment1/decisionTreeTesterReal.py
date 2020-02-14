#! /usr/local/bin/python3


import decisionTreeReal
import realDataParser as parser
import matplotlib.pyplot as plt
from anytree import Node, RenderTree, Resolver
import numpy as np

testData = parser.realDataSet("Video_Games_Sales_Subset.csv", 11, [0,0,0,0,1,1,1,1,1,0,0], 20)
testData.discretizeFeatures()
testData.discretizeClassLabels()
testData.findCategoricalValues()
myTree = decisionTreeReal.decisionTreeReal(11, 20, "Video_Games_Sales_Subset.csv")
myTree.printTree()
