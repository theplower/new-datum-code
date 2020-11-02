import cv2 as cv
import torch
import torchvision
import numpy as np
import sys
from torch import nn
from datetime import datetime

a = datetime.now()


PATH = "hand_sign_model.pt"

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

model.load_state_dict(torch.load(PATH,torch.device("cpu")))

#model = torch.load("pytorchClassifierModel.pt",torch.device("cpu"))
b = datetime.now()
model.eval()
cap=cv.VideoCapture(sys.argv[1])

print('Model loaded in ', (b - a))

class_names = ['ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf', 'ghain', 'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 'laam', 'meem', 'nun', 'ra', 'saad', 'seen', 'sheen', 'ta', 'taa', 'thaa', 'thal', 'toot', 'waw', 'ya', 'yaa', 'zay']

arabicnames = ["ع","ال","ا","ب","د","ظ","ض","ف","ق","غ","ح","ه","ج","ك","خ","لا","ل","م","ن","ر","ص","س","ش","ت","ط","ذ","ة","و","ئ","ي","ز"]

cv.namedWindow('Window')
font = cv.FONT_HERSHEY_SIMPLEX
count = 0

while True:
	count += 1
	if count < 10000:
		continue
	_,frame = cap.read()
	frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	origFrame = cv.resize(frame, (480, 360))
	frame = cv.resize(origFrame, (64, 64))
	#converting it to the form that is known to the model
	test=np.array(frame,dtype=np.float32)
	test = test/255
	test = (test - 0.5) / 0.5
	test = torch.tensor(test)
	
	#getting the sign meaning 
	prediction = model(test.view(1,1,64,64))
	confidences = torch.softmax(prediction.detach(),1)
	confidencePercent = np.amax(confidences.numpy(),1).item()
	if confidencePercent > 0.7:
		classIndex = np.argmax(prediction.detach().numpy())
#		print(arabicnames[classIndex])
#		print(class_names[classIndex])
		letter = class_names[classIndex]
		message = 'Letter: %s   Prob: %.2f' % (letter, confidencePercent * 100) + '%'
#		print(message)
	else:
		message = 'Not sure enough'

	cv.putText(origFrame, message, (20, 25), font, 0.75, (255, 0, 0), 2, cv.LINE_AA)
	cv.imshow('Window', origFrame)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break
