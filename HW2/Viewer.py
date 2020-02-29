import numpy as np
class Viewer:
    def __init__(self, viewerID):
        self.viewerID = viewerID
        self.reviews = {}
        self.augmentedFilmsList = []
        self.profile = np.zeros(19)

    def addReview(self, line, allFilms):
        self.reviews[int(line[1])] = int(line[2])
        self.augmentedFilmsList.append(int(line[2])*allFilms.films[int(line[1])])


    def createProfile(self):
        for film in self.augmentedFilmsList:
            self.profile = np.add(self.profile, film)


    def printReviews(self):
        print(self.reviews)
