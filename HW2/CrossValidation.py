import CollaborativeFiltering as CF
import FilmCollection
import Viewer
import numpy as np
import copy

allFilms = FilmCollection.FilmCollection("./data/u-item.item")
viewers = []
trainData = "./data/u1-base.base"
testData = "./data/u1-base.base"
k = 5
allViewers = CF.createListOfViewers(trainData, allFilms, 1, 0)

def createValidationSet(k, startIndex):
    viewers = CF.createListOfViewers(trainData, allFilms, k, startIndex)
    return viewers


def createTrainingSet(validationSet):
    trainingSet = {}
    for viewerID in allViewers:
        trainingSet[viewerID] = copy.deepcopy(allViewers[viewerID])
        for review in allViewers[viewerID].reviews:
            try:
                if review in validationSet[viewerID].reviews:
                    del trainingSet[viewerID].reviews[review]
            except:
                continue
    return trainingSet

for j in range(1,6):
    averageMSE = 0
    for fold in range(1,k+1):
        validationSet = createValidationSet(5, fold)
        trainingSet = createTrainingSet(validationSet)
        testResults = []
        for viewer in trainingSet:
            trainingSet[viewer].createProfile()

        for validationTestCase in validationSet:
            for review in validationSet[validationTestCase].reviews:
                prediction = CF.runTest(j, validationSet[validationTestCase].viewerID, review, int(validationSet[validationTestCase].reviews[review]), trainingSet)
                testResults.append([prediction, int(validationSet[validationTestCase].reviews[review]), ((prediction-int(validationSet[validationTestCase].reviews[review])) ** 2)])
        testResults = np.array(testResults, float)
        averageMSE += np.sum(testResults, axis=0)[2]/(36770/5)
    averageMSE = averageMSE/k
    print(averageMSE)
