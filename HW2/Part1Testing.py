import CollaborativeFiltering as CF
import FilmCollection
import Viewer
import numpy as np

allFilms = FilmCollection.FilmCollection("./data/u-item.item")
trainData = "./data/u1-base.base"
testData = "./data/u1-test.test"
viewers = CF.createListOfViewers(trainData, allFilms, 1, 0)

for viewer in viewers:
    viewers[viewer].createProfile()
file = open(testData, "r")
testCases = []
testResults = []
testDataLines = file.readlines()
for line in testDataLines:
    line = line.rstrip("\n")
    line = line.split()
    testCases.append(line)

for testCase in testCases:
    prediction = CF.runTest(3, int(testCase[0]), int(testCase[1]), int(testCase[2]), viewers)
    testResults.append([prediction, int(testCase[2]), ((prediction-int(testCase[2])) ** 2)])
testResults = np.array(testResults, float)
print(np.sum(testResults, axis=0)[2]/len(testCases))
