#! /usr/local/bin/python3


import decisionTreeSynthetic
import syntheticDataParser as parser
from anytree import Node, RenderTree, Resolver

testData = parser.syntheticDataSet("synthetic-4.csv", 2, 10) #4, 10, 10, 10 [number of bins]
myTree = decisionTreeSynthetic.decisionTree(2, 10, "synthetic-4.csv") #4, 10, 10, 10

correct = 0
total = testData.data.shape[0]
print(testData.data)
for dataPoint in testData.data:
    r = Resolver()
    query = "Feature1/bin " + str(int(dataPoint[0])) + "/Feature2/bin " + str(int(dataPoint[1])) + "/*"
    try:
        prediction = r.glob(myTree.decisionTree, query)[0].name
    except:
        try:
            query = "Feature2/bin " + str(int(dataPoint[1])) + "/Feature1/bin " + str(int(dataPoint[0])) + "/*"
            prediction = r.glob(myTree.decisionTree, query)[0].name
        except:
            try:
                query = "Feature2/bin " + str(int(dataPoint[1])) + "/*"
                prediction = r.glob(myTree.decisionTree, query)[0].name
            except:
                query = "Feature1/bin " + str(int(dataPoint[0])) + "/*"
                prediction = r.glob(myTree.decisionTree, query)[0].name

    print(query)
    actual = "Label = " + str(int(dataPoint[2]))
    print(prediction, actual)
    if actual == prediction:
        correct = correct + 1
print(float(correct/total))
myTree.printTree()
