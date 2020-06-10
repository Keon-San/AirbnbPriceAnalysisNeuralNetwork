import csv
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torch.optim as optim
import datetime
import torch
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np

def make1HotVector(labels, value):
    outArray = []
    for x in labels:
        if value == x:
            outArray.append(1)
        else:
            outArray.append(0)
    return outArray

class Listing:
    def __init__(self, borough, neighborhood, roomType, reviews, latestReview, reviewsPerMonth, numListings, numDays, price):
        self.reviews = float(reviews)
        if (self.reviews == 0):
            self.reviewsPerMonth = 0
        else:
            self.reviewsPerMonth = float(reviewsPerMonth)
        self.numListings = float(numListings)
        self.numDays = float(numDays)

        self.borough = make1HotVector(boroughLabels, borough)

        self.roomType = make1HotVector(roomLabels, roomType)

        self.neighborhood = make1HotVector(neighborhoodLabels, neighborhood)
        
        if latestReview == '':
            self.latestReview = datetime.timedelta(0)
        elif latestReview.find('/') != -1:
            self.latestReview = datetime.date(2019, 8, 13) - datetime.datetime.strptime(latestReview, '%m/%d/%Y')
        else:
            self.latestReview = datetime.datetime(2019, 8, 13) - datetime.datetime.strptime(latestReview, '%Y-%m-%d')
        
        self.price = int(price)

data = open('AB_NYC_2019.csv', "r", encoding="utf-8")
data_reader=csv.reader(data, delimiter=',')

prices = []
reviewsPerMonth = []
totalData = []

roomLabels = set()
boroughLabels = set()
neighborhoodLabels = set()

for row in data_reader:
    roomLabels.add(row[8])
    boroughLabels.add(row[4])
    neighborhoodLabels.add(row[5])
data.seek(0)

for row in data_reader:
    if row[0] == "id":  
        continue
    prices.append(row[9])
    reviewsPerMonth.append(row[13])
    totalData.append(Listing(row[4], row[5], row[8], row[11], row[12], row[13], row[14], row[15], row[9]))

plt.scatter(reviewsPerMonth, prices)
plt.show()
