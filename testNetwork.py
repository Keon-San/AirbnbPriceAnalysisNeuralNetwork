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

        self.bronx = 1 if borough == "Bronx" else 0
        self.manhattan = 1 if borough == "Manhattan" else 0
        self.queens = 1 if borough == "Queens" else 0
        self.brooklyn = 1 if borough == "Brooklyn" else 0
        self.staten = 1 if borough == "Staten Island" else 0

        self.shared = 1 if roomType == "Shared room" else 0
        self.private = 1 if roomType == "Private room" else 0
        self.entire = 1 if roomType == "Entire home/apt" else 0
        
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
        self.lin1 = nn.Linear(13, 7)
        self.lin2 = nn.Linear(7, 1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

net = Net()
net.to(device)
learningRate = 0.00001

optimizer = optim.SGD(net.parameters(), lr=learningRate)

random.shuffle(totalData)

trainData = totalData[:34226]
trainParamTensor = torch.tensor([[[i.reviews, i.reviewsPerMonth, i.numListings, i.numDays, i.bronx, i.manhattan, i.queens, i.brooklyn, i.staten, i.shared, i.private, i.entire, i.latestReview.days]] for i in trainData], dtype=torch.double).to(device).float()
trainLabelTensor = torch.tensor([[[i.price]] for i in trainData], dtype=torch.double).to(device).float()

validationData = totalData[34226:41561]
validationParamTensor = torch.tensor([[[i.reviews, i.reviewsPerMonth, i.numListings, i.numDays, i.bronx, i.manhattan, i.queens, i.brooklyn, i.staten, i.shared, i.private, i.entire, i.latestReview.days]] for i in validationData], dtype=torch.double).to(device).float()
validationLabelTensor = torch.tensor([[[i.price]] for i in validationData], dtype=torch.double).to(device).float()

testData = totalData[41561:]
testParamTensor = torch.tensor([[[i.reviews, i.reviewsPerMonth, i.numListings, i.numDays, i.bronx, i.manhattan, i.queens, i.brooklyn, i.staten, i.shared, i.private, i.entire, i.latestReview.days]] for i in testData], dtype=torch.double).to(device).float()
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
        if lossArray[len(lossArray) - 1]> lossArray[len(lossArray) - 11]:
            print("Done")
            break


with torch.no_grad():
        out = net(testParamTensor)
        lossfn = nn.MSELoss()
        loss = lossfn(out, testLabelTensor)
        print("test loss " + str(loss.item()))
