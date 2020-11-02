import cv2 as cv
import torch
import torchvision
import numpy as np
import sys
import pyautogui
from torch import nn
from datetime import datetime

a = datetime.now()


PATH = "Keymodel.pt" #"80 epoch weight with dropout.pt"#"81-87transfered.pt"
model = torch.load(PATH,torch.device("cpu"))

b = datetime.now()
model.eval()
cap=cv.VideoCapture(sys.argv[1])
print('Model loaded in ', (b - a))

class_names = ['down', 'left', " ", 'right', 'up']
frameCount = 0
frameStep = 24
cv.namedWindow('Window')
font = cv.FONT_HERSHEY_SIMPLEX
placeholder = None
x2 = int(cap.get(3) / 2) + 20
y2 = int(cap.get(4) / 2) + 20
while True:

	_, origFrame = cap.read()
#	frame = cv.flip(frame,1)
#	origFrame = cv.resize(frame, (480, 360))
#	frame = cv.resize(frame, (480, 360))
	x1, y1 = (0, 0)
	
#	if frameCount % frameStep == 0:	
	frame = origFrame[0: x2, 0: y2]
	cut = frame = cv.resize(frame, (64,64))
	frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	#converting it to the form that is known to the model
	frame = np.array(frame,dtype=np.float32)
	frame = frame / 255
	frame = (frame - 0.5) / 0.5
	frame = torch.tensor(frame)
	
	#getting the sign meaning 
	prediction = model(frame.view(1,1,64,64))
	confidences = torch.softmax(prediction.detach(),1)
	confidencePercent = np.amax(confidences.numpy(),1).item()
	if confidencePercent > 0.2:
		classIndex = np.argmax(prediction.detach().numpy())
		letter = class_names[classIndex]
		if letter == 'up':
			pyautogui.press('UP')
		elif letter == 'down':
			pyautogui.press('DOWN')
		elif letter == 'right':
			pyautogui.press('RIGHT')
		elif letter == 'left':
			pyautogui.press('LEFT')
		message = letter
	else:
		message = ' '
#		placeholder = message
#else:
#	message = placeholder
	cv.putText(origFrame, message, (20, 25), font, 0.75, (255, 0, 0), 2, cv.LINE_AA)
	cv.rectangle(origFrame, (0, 0), (x2, y2), (0, 255, 0), 3)
#	origFrame = cv.flip(origFrame, 1)
	cv.imshow('Window', origFrame)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break
