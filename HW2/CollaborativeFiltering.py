import FilmCollection
import Viewer
import numpy as np

def createListOfViewers(fileName, allFilms, k, startIndex):
    viewers = {}
    file = open(fileName, "r")
    currentViewer = 1
    viewers[currentViewer] = Viewer.Viewer(currentViewer)
    lines = file.readlines()
    for i in range(startIndex, len(lines), k):
        lines[i] = lines[i].rstrip("\n")
        lines[i] = lines[i].split()
        if int(lines[i][0]) != currentViewer:
            currentViewer = int(lines[i][0])
            viewers[currentViewer] = Viewer.Viewer(currentViewer)
        elif int(lines[i][0]) == currentViewer:
            viewers[currentViewer].addReview(lines[i], allFilms)
    return viewers

def calculateEuclideanDistance(testViewer, trainingViewer, viewers):
    return np.sqrt(np.sum((testViewer.profile - trainingViewer.profile) ** 2))

def runTest(k, testViewerID, testFilm, actualValue, viewers):
    nearestNeighbors = np.empty(k, int)
    distances = []
    overallPrediction = 0
    for trainingViewer in viewers:
        if viewers[trainingViewer].viewerID == testViewerID:
            continue
        else:
            distances.append(calculateEuclideanDistance(viewers[testViewerID], viewers[trainingViewer], viewers))
    distances = np.array(distances, float)
    i = 0

    #get rid of neighbors that haven't seen the movie
    while i < k:
        if distances[np.argmin(distances)] == 99999999:
            break
        nearestNeighbors[i] = np.argmin(distances)
        distances[np.argmin(distances)] = 99999999
        if testFilm in viewers[nearestNeighbors[i] + 1].reviews:
            neighborPrediciton = viewers[nearestNeighbors[i] + 1].reviews[testFilm]
            i += 1
            overallPrediction += neighborPrediciton
        else:
            continue
    return overallPrediction / k
