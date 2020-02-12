#! /usr/local/bin/python3
import csv


class syntheticDataRaw:
    def __init__(self, fileName):
        self.dataSource = "./data/" + fileName
        self.data = []
        self.__readCSVFile__()

    def __readCSVFile__(self):
        with open(self.dataSource) as dataCSV:
            dataReader = csv.reader(dataCSV)
            for row in dataReader:
                self.data.append(row)

    def printData(self):
        for row in self.data:
            print(row)


class syntheticDataFormatted:
    def __init__(self, rawData)
test = syntheticDataRaw("synthetic-1.csv")
test.printData()
