import numpy as np
genres = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western"
]
class FilmCollection:

    def __init__(self, fileName):
        self.fileName = fileName
        self.films = self.populateFilmCollection()

    def populateFilmCollection(self):
        file = open(self.fileName, "r")
        films = []
        lines = file.readlines()
        for i in lines:
        	i = i.rstrip("\n")
        	i = i.split("|")
        	films.append(i[5:])
        films = np.array(films, int)
        return films
