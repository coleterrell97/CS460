#! /usr/local/bin/python3
import syntheticDataParser as parser
import math
from anytree import Node, RenderTree

def ID3(dataSet, target_attribute, attributes, depth):
        subTreeRoot = Node(target_attribute, label = -1)
        labelDistribution = determineClassLabels(dataSet)
        #base cases
        if labelDistribution[0] == 0:
            subTreeRoot.label = 1
            return subTreeRoot
        elif labelDistribution[1] == 0:
            subTreeRoot.label = 0
            return subTreeRoot
        elif attributes == [] or depth == 3:
            if(labelDistribution[0] > labelDistribution[1]):
                subTreeRoot.label = 1
                return subTreeRoot
            elif(labelDistribution[1] > labelDistribution[0]):
                subTreeRoot.label = 1
                return subTreeRoot
            elif(labelDistribution[1] == labelDistribution[0]):
                subTreeRoot.label = 1
                return subTreeRoot
        #end base cases
        #begin entropy calculation
        else:
            target_attribute = findBestSplit(dataSet, labelDistribution, attributes)
            return subTreeRoot

def determineClassLabels(dataSet):
    oneLabels = 0
    zeroLabels = 0
    labels = [0, 0]
    for dataPoint in dataSet:
        if dataPoint[2] == 1:
            oneLabels = oneLabels + 1
        elif dataPoint[2] == 0:
            zeroLabels = zeroLabels + 1
    labels[0] = zeroLabels
    labels[1] = oneLabels
    return labels

def calculateEntropy(dataSet, labelDistribution):
    entropy = 0
    for label in labelDistribution:
        entropy = entropy + (-((label/dataSet.shape[0]) * math.log(label/dataSet.shape[0],2)))
    return entropy

def findBestSplit(dataSet, labelDistribution, attributes):

    parentEntropy = calculateEntropy(dataSet, labelDistribution)
    return 0



trainingData = parser.syntheticDataSet("synthetic-1.csv", 2, 5)
decisionTree = ID3(trainingData.data, "ALL", [0,1], 0)
print(RenderTree(decisionTree))
