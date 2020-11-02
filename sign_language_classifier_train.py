# -*- coding: utf-8 -*-
"""torchTestin(1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jphlcfFt26G588QZ0NF7jgiDNf4YLKD_
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms
import helper
import numpy as np

print(torch.cuda.is_available())
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

transform = transforms.Compose([transforms.Resize((64,64)),transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

trainDataset = datasets.ImageFolder("Dataset/train",transform)
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=32, shuffle=True)


validDataset = datasets.ImageFolder("Dataset/valid",transform)
validLoader = torch.utils.data.DataLoader(validDataset, batch_size=len(validDataset), shuffle=True)

testDataset = datasets.ImageFolder("Dataset/test",transform)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size=len(testDataset), shuffle=True)

print(len(trainDataset), len(validDataset), len(testDataset))

images, labels = next(iter(trainLoader))
print(images[0].detach().numpy().shape)
plt.title(trainDataset.classes[labels[0]])
plt.imshow(images[0].detach().numpy().reshape(64,64),cmap="gray")

Xval , Yval = next(iter(validLoader))
Xtest, Ytest = next(iter(testLoader))

model = torch.nn.Sequential(
      nn.Conv2d(1,32,3,1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(32,64,3,1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(64,128,3,1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(128,256,3,1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Flatten(),
      nn.Linear(1024,100),
      nn.Dropout(0.5),
      nn.ReLU(),
      nn.Linear(100,32)
)
optim = torch.optim.Adam(model.parameters(),lr =0.001)
crit = torch.nn.CrossEntropyLoss()
epochs = 100
model.to(dev)

#optim = torch.optim.Adam(model.parameters(),lr =0.00001)
for epoch in range(epochs):
  trainAcc = []
  print('Epoch ' + str(epoch+1) + '/' + str(epochs))
  for x, y in trainLoader:
    model.train()
    yhat = model(x.to(dev))
    optim.zero_grad()
    loss = crit(yhat, y.to(dev))
    loss.backward()
    optim.step()
    print('\r', loss.item(), end='')
    trainAcc.append(np.squeeze((np.argmax(torch.softmax(yhat.to("cpu").detach(),dim=1),1) == y.to("cpu")).float().mean()))
  with torch.no_grad():
    model.eval()
    print("\ntrain accuracy : ",np.array(trainAcc).mean()*100)
    yhat = model(Xval.to(dev))
    print("validation accuracy : ",(np.squeeze((np.argmax(torch.softmax(yhat.to("cpu").detach(),dim=1),1) == Yval.to("cpu")).float().mean())).item()*100)
  if epoch % 5 ==0:
    torch.save(model,"{} epoch model .pt".format(epoch))

acc = []
for da,la in dataset.items():
  yhat = model(torch.tensor(da))
  yhat = np.argmax(torch.softmax(yhat.detach(),1),axis = 1)
  if yhat == la:
    acc.append(1)
  else:
    acc.append(0)

print("accuracy : {}".format(np.array(acc).mean()))

f = (np.random.rand(2,2).reshape(1,2*2) )
print(f)
out = torch.softmax(model(torch.tensor(f,dtype=torch.float32)),dim=1).detach().numpy()

print("class : ",np.argmax(out,axis=1) , "\nprobability : ",np.amax(out, axis=1))

#torch.save(model,"99t-97v model.pt")
pred = model.to("cpu")(torch.tensor(Xtest[22].detach().numpy()).view(1,1,64,64))
confidence = torch.softmax(pred.detach(),1)
print(confidence)
print(np.amax(confidence.numpy(),1))
print(trainDataset.classes[Ytest[22]],Ytest[22])

with torch.no_grad():
    yhat = model.to("cuda")(Xtest.to(dev))
    print("test accuracy : ",(np.squeeze((np.argmax(torch.softmax(yhat.to("cpu").detach(),dim=1),1) == Ytest.to("cpu")).float().mean())).item()*100)

print(trainDataset.classes)

print(model)
