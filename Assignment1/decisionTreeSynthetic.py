#! /usr/local/bin/python3
import syntheticDataParser as parser
import math
import numpy as np
from anytree.exporter import DotExporter


from anytree import Node, RenderTree

def ID3(dataSet, target_attribute, attributes, depth, parentNode):
        labelDistribution = determineClassLabels(dataSet)
        #base cases
        if labelDistribution[0] == 0:
            leaf = Node("Label = 1", parent=parentNode)
        elif labelDistribution[1] == 0:
            leaf = Node("Label = 0", parent=parentNode)
        elif attributes == [] or depth == 3:
            if(labelDistribution[0] > labelDistribution[1]):
                leaf = Node("Label = 0", parent=parentNode)
            elif(labelDistribution[1] > labelDistribution[0]):
                leaf = Node("Label = 1", parent=parentNode)
            elif(labelDistribution[1] == labelDistribution[0]):
                leaf = Node("Label = 1", parent=parentNode)
        #end base cases
        #begin entropy calculation
        else:
            target_attribute = findBestSplit(dataSet, labelDistribution, attributes)
            newRoot = Node(attributes[target_attribute[0]], parent=parentNode)
            attributes.pop(target_attribute[0])
            print(attributes)
            childSets = findChildDataSets(dataSet,target_attribute)
            for bin in range(1, 1 + numBins):
                currentChildSet = childSets[bin-1]
                if(currentChildSet.shape[0] == 0):
                    if(labelDistribution[0] > labelDistribution[1]):
                        leaf = Node("Label = 0", parent=newRoot)
                    elif(labelDistribution[1] > labelDistribution[0]):
                        leaf = Node("Label = 1", parent=newRoot)
                    elif(labelDistribution[1] == labelDistribution[0]):
                        leaf = Node("Label = 1", parent=newRoot)
                else:
                    branch = Node(bin, parent=newRoot);
                    ID3(currentChildSet, target_attribute, attributes, depth+1, branch)

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
    splitInfoGain = np.zeros([1,len(attributes)])
    parentEntropy = calculateEntropy(dataSet, labelDistribution)
    for attribute in range(0,len(attributes)): #for each feature
        splitInfoGain[0][attribute] = calculateChildAverageEntropy(dataSet, attribute)
        splitInfoGain[0][attribute] = parentEntropy - splitInfoGain[0][attribute]
    bestSplit = np.where(splitInfoGain[0] == np.amax(splitInfoGain[0]))
    return bestSplit[0]


def calculateChildAverageEntropy(dataSet, attribute):
    totalExamples = dataSet.shape[0]
    childAverageEntropy = 0
    childSets = findChildDataSets(dataSet,attribute)
    for set in childSets:
        labelDistribution = determineClassLabels(set)
        if labelDistribution[0] == 0 or labelDistribution[1] == 0:
            childAverageEntropy = childAverageEntropy
        else:
            childAverageEntropy = childAverageEntropy + (set.shape[0]/totalExamples*(calculateEntropy(set, labelDistribution)))
    return childAverageEntropy

def findChildDataSets(dataSet, attribute):
    childDataSets = []
    for bin in range(1,numBins+1):
        childDataSet = []
        for dataPoint in dataSet:
            if dataPoint[attribute] == bin:
                childDataSet.append(dataPoint)
        childDataSet = np.array(childDataSet, float)
        childDataSets.append(childDataSet)
    return childDataSets

numBins = 5
numFeatures = 2
trainingData = parser.syntheticDataSet("synthetic-4.csv", numFeatures, numBins)
decisionTree = Node('TOP')
ID3(trainingData.data, "", ["Column1", "Column2"], 0, decisionTree)
for pre, fill, node in RenderTree(decisionTree):
    print("%s%s" % (pre, node.name))
