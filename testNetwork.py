import csv
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torch.optim as optim
import datetime
import torch
import random

class Listing:
    def __init__(self, borough, roomType, reviews, latestReview, reviewsPerMonth, numListings, numDays, price):
        self.reviews = float(reviews)
        if (self.reviews == 0):
            self.reviewsPerMonth = 0
        else:
            self.reviewsPerMonth = float(reviewsPerMonth)
        self.numListings = float(numListings)
        self.numDays = float(numDays)

        self.boroughEnum = {
            "Bronx": 1,
            "Manhattan": 2,
            "Queens": 3,
            "Brooklyn": 4,
            "Staten Island": 5
        }
        self.borough = self.boroughEnum.get(borough)
        self.roomEnum = {
            "Shared room": 1,
            "Private room": 2,
            "Entire home/apt": 3
        }
        self.roomType = self.roomEnum.get(roomType)
        if latestReview == '':
            self.latestReview =     datetime.timedelta(-1)
        elif latestReview.find('/') != -1:
            self.latestReview = datetime.date(2019, 8, 12) - datetime.datetime.strptime(latestReview, '%m/%d/%Y')
        else:
            self.latestReview = datetime.datetime(2019, 8, 12) - datetime.datetime.strptime(latestReview, '%Y-%m-%d')
        
        self.price = int(price)

data = open('AB_NYC_2019.csv', "r", encoding="utf-8")
data_reader=csv.reader(data, delimiter=',')

totalData = []

for row in data_reader:
    if row[0] == "id":  
        continue
    totalData.append(Listing(row[4], row[8], row[11], row[12], row[13], row[14], row[15], row[9]))
print(len(totalData))
device = "cuda" if torch.cuda.is_available() else "cpu"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(7, 4)
        self.lin2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

net = Net()
net.to(device)
learningRate = 0.0001

optimizer = optim.SGD(net.parameters(), lr=learningRate)

random.shuffle(totalData)

trainData = totalData[:34226]
trainParamTensor = torch.tensor([[[i.reviews, i.reviewsPerMonth, i.numListings, i.numDays, i.borough, i.roomType, i.latestReview.days]] for i in trainData], dtype=torch.double).to(device).float()
trainLabelTensor = torch.tensor([[[i.price]] for i in trainData], dtype=torch.double).to(device).float()

validationData = totalData[34226:41561]
validationParamTensor = torch.tensor([[[i.reviews, i.reviewsPerMonth, i.numListings, i.numDays, i.borough, i.roomType, i.latestReview.days]] for i in validationData], dtype=torch.double).to(device).float()
validationLabelTensor = torch.tensor([[[i.price]] for i in validationData], dtype=torch.double).to(device).float()

testData = totalData[41561:]
testParamTensor = torch.tensor([[[i.reviews, i.reviewsPerMonth, i.numListings, i.numDays, i.borough, i.roomType, i.latestReview.days]] for i in testData], dtype=torch.double).to(device).float()
testLabelTensor = torch.tensor([[[i.price]] for i in testData], dtype=torch.double).to(device).float()

lossArray = []

while True:
    optimizer.zero_grad()
    outVal = net(trainParamTensor.float())
    lossfn = nn.MSELoss()
    loss = lossfn(outVal, trainLabelTensor)
    print(loss.item())
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        out = net(validationParamTensor)
        loss = lossfn(out, validationLabelTensor)
        print("Val loss " + str(loss.item()))
        lossArray.append(loss.item())

    if (len(lossArray) > 10):
        if lossArray[len(lossArray) - 1]/lossArray[len(lossArray) - 11] >= 1:
            print("Done")
            break
