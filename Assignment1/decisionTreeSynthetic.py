#! /usr/local/bin/python3
import syntheticDataParser as parser
import math
import numpy as np
import copy
from anytree import Node, RenderTree
from anytree.exporter import DotExporter


class decisionTree:
    def __init__(self, numFeatures, numBins, dataFileName):
        self.numFeatures = numFeatures
        self.numBins = numBins
        self.dataFileName = dataFileName
        self.trainingData = parser.syntheticDataSet(self.dataFileName, self.numFeatures, self.numBins)
        self.decisionTree = Node('TOP')
        self.ID3(self.trainingData.data, "", ["Feature1", "Feature2"], 0, self.decisionTree)

    def ID3(self, dataSet, target_attribute, attributes, depth, parentNode):
            labelDistribution = self.determineClassLabels(dataSet)
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
                target_attribute = self.findBestSplit(dataSet, labelDistribution, attributes)
                newRoot = Node(attributes[target_attribute[0]], parent=parentNode)
                newAttributes = copy.deepcopy(attributes)
                newAttributes.pop(target_attribute[0])
                childSets = self.findChildDataSets(dataSet,target_attribute)
                for bin in range(1, 1 + self.numBins):
                    branch = Node("bin %s" % bin, parent=newRoot);
                    currentChildSet = childSets[bin-1]
                    if(currentChildSet.shape[0] == 0):
                        if(labelDistribution[0] > labelDistribution[1]):
                            leaf = Node("Label = 0", parent=branch)
                        elif(labelDistribution[1] > labelDistribution[0]):
                            leaf = Node("Label = 1", parent=branch)
                        elif(labelDistribution[1] == labelDistribution[0]):
                            leaf = Node("Label = 1", parent=branch)
                    else:
                        self.ID3(currentChildSet, target_attribute, newAttributes, depth+1, branch)

    def determineClassLabels(self, dataSet):
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

    def calculateEntropy(self, dataSet, labelDistribution):
        entropy = 0
        for label in labelDistribution:
            entropy = entropy + (-((label/dataSet.shape[0]) * math.log(label/dataSet.shape[0],2)))
        return entropy

    def findBestSplit(self, dataSet, labelDistribution, attributes):
        splitInfoGain = np.zeros([1,len(attributes)])
        parentEntropy = self.calculateEntropy(dataSet, labelDistribution)
        for attribute in range(0,len(attributes)): #for each feature
            splitInfoGain[0][attribute] = self.calculateChildAverageEntropy(dataSet, attribute)
            splitInfoGain[0][attribute] = parentEntropy - splitInfoGain[0][attribute]
        bestSplit = np.where(splitInfoGain[0] == np.amax(splitInfoGain[0]))
        return bestSplit[0]


    def calculateChildAverageEntropy(self, dataSet, attribute):
        totalExamples = dataSet.shape[0]
        childAverageEntropy = 0
        childSets = self.findChildDataSets(dataSet,attribute)
        for set in childSets:
            labelDistribution = self.determineClassLabels(set)
            if labelDistribution[0] == 0 or labelDistribution[1] == 0:
                childAverageEntropy = childAverageEntropy
            else:
                childAverageEntropy = childAverageEntropy + (set.shape[0]/totalExamples*(self.calculateEntropy(set, labelDistribution)))
        return childAverageEntropy

    def findChildDataSets(self, dataSet, attribute):
        childDataSets = []
        for bin in range(1,self.numBins+1):
            childDataSet = []
            for dataPoint in dataSet:
                if dataPoint[attribute] == bin:
                    childDataSet.append(dataPoint)
            childDataSet = np.array(childDataSet, float)
            childDataSets.append(childDataSet)
        return childDataSets

    def printTree(self):
        for pre, fill, node in RenderTree(self.decisionTree):
            print("%s%s" % (pre, node.name))
