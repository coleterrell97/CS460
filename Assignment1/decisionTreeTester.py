#! /usr/local/bin/python3


import decisionTreeSynthetic
import syntheticDataParser as parser
from anytree import Node, RenderTree, Resolver

testData = parser.syntheticDataSet("synthetic-4.csv", 2, 5)
myTree = decisionTreeSynthetic.decisionTree(2, 5, "synthetic-4.csv")

correct = 0
total = testData.data.shape[0]
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
