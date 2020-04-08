import csv
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torch.optim as optim
import datetime
import torch
import random

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
            self.latestReview =     datetime.timedelta(-1)
        elif latestReview.find('/') != -1:
            self.latestReview = datetime.date(2019, 8, 12) - datetime.datetime.strptime(latestReview, '%m/%d/%Y')
        else:
            self.latestReview = datetime.datetime(2019, 8, 12) - datetime.datetime.strptime(latestReview, '%Y-%m-%d')
        
        self.price = int(price)

data = open('AB_NYC_2019.csv', "r", encoding="utf-8")
data_reader=csv.reader(data, delimiter=',')

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
    totalData.append(Listing(row[4], row[5], row[8], row[11], row[12], row[13], row[14], row[15], row[9]))
print(len(totalData))
device = "cuda" if torch.cuda.is_available() else "cpu"

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(input_size, 1000)
        self.lin2 = nn.Linear(1000, 500)
        self.lin3 = nn.Linear(500, 250)
        self.lin4 = nn.Linear(250, 1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return x



random.shuffle(totalData)

trainData = totalData[:34226]
trainParamTensor = torch.tensor([[[i.reviews, i.reviewsPerMonth, i.numListings, i.numDays, i.latestReview.days] + i.borough + i.roomType + i.neighborhood] for i in trainData], dtype=torch.double).to(device).float()
trainLabelTensor = torch.tensor([[[i.price]] for i in trainData], dtype=torch.double).to(device).float()

validationData = totalData[34226:41561]
validationParamTensor = torch.tensor([[[i.reviews, i.reviewsPerMonth, i.numListings, i.numDays, i.latestReview.days] + i.borough + i.roomType + i.neighborhood] for i in validationData], dtype=torch.double).to(device).float()
validationLabelTensor = torch.tensor([[[i.price]] for i in validationData], dtype=torch.double).to(device).float()

testData = totalData[41561:]
testParamTensor = torch.tensor([[[i.reviews, i.reviewsPerMonth, i.numListings, i.numDays, i.latestReview.days] + i.borough + i.roomType + i.neighborhood] for i in testData], dtype=torch.double).to(device).float()
testLabelTensor = torch.tensor([[[i.price]] for i in testData], dtype=torch.double).to(device).float()

net = Net(trainParamTensor.shape[2])
net.to(device)
learningRate = 0.00001

optimizer = optim.SGD(net.parameters(), lr=learningRate)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

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
        scheduler.step(loss)


with torch.no_grad():
        out = net(testParamTensor)
        lossfn = nn.MSELoss()
        loss = lossfn(out, testLabelTensor)
        print("test loss " + str(loss.item()))
