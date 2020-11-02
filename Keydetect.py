# Import the required libraries
import cv2 as cv
import torch
import torchvision
import numpy as np
import sys
import pyautogui
from torch import nn
from datetime import datetime

# Record the time taken to load the model
a = datetime.now()
# Define the path to the trained model and load it 
PATH = "Keymodel.pt" #"80 epoch weight with dropout.pt"#"81-87transfered.pt"
model = torch.load(PATH,torch.device("cpu"))
b = datetime.now()

# Set the model to the prediction mode
model.eval()

# Define the video stream input
cap = cv.VideoCapture(sys.argv[1])

print('Model loaded in ', (b - a))

# Define the classes
class_names = ['down', 'left', " ", 'right', 'up']

# Open the window used to show the frames
cv.namedWindow('Window')
font = cv.FONT_HERSHEY_SIMPLEX

# Define the coords for the green rectangle that must enclose the hand direction
x2 = int(cap.get(3) / 2) + 20
y2 = int(cap.get(4) / 2) + 20

while True:
	# Read the video frame by frame
	success, origFrame = cap.read()
	# Assert that the stream isn't cut
	if not success:
		break
	# Extract the hand	
	frame = origFrame[0: x2, 0: y2]
	
	# Set the extracted part of the image to the form known to the model
	frame = cv.resize(frame, (64,64))
	frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	#converting it to the form that is known to the model
	frame = np.array(frame,dtype=np.float32)
	frame = frame / 255
	frame = (frame - 0.5) / 0.5
	frame = torch.tensor(frame)
	
	# Get the sign meaning 
	prediction = model(frame.view(1,1,64,64))
	confidences = torch.softmax(prediction.detach(),1)
	confidencePercent = np.amax(confidences.numpy(),1).item()
	if confidencePercent > 0.2:
		classIndex = np.argmax(prediction.detach().numpy())
		letter = class_names[classIndex]
		# Decide which key to press
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
	# Show the recognized direction on the video (or Live Camera stream)
	cv.putText(origFrame, message, (20, 25), font, 0.75, (255, 0, 0), 2, cv.LINE_AA)
	cv.rectangle(origFrame, (0, 0), (x2, y2), (0, 255, 0), 3)
	# Uncomment the next line if the image is flipped
	# origFrame = cv.flip(origFrame, 1)
	cv.imshow('Window', origFrame)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break
