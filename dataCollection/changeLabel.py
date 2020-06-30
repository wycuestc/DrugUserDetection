import os
from numpy import array, asarray
import pandas as pd
import numpy as np
import csv

scriptDir = os.path.dirname(__file__)
inputFile = os.path.join(scriptDir, "dataset1000/dataOpiumStreet1020.csv")
outputFile = os.path.join(scriptDir, "dataset1000/bidataOpiumStreet1020.csv")

if os.path.exists(outputFile):
    os.remove(outputFile)
else:
    print("The fileTemp does not exist")

out = open(outputFile, 'a')

with open(inputFile) as csvfile:
    readCSV = csv.reader(csvfile, delimiter = ',')
    for row in readCSV:
        if str(row[2]) == "3":
            row[2] = 1
        elif str(row[2]) == "4" or str(row[2]) == "5":
            row[2] = 0
        out.write(str(row[0])+','+str(row[1])+','+str(row[2])+','+str(row[3])+','+str(row[5])+','+str(row[4])+'\n')


