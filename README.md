# Sign Language Classifier
To run the arabic sign languange classifier, type: 
*python hand_sign_classifier.py path-to-video-of-hand-signs* in the terminal.
It has a very good accuracy to gestures that it is similar to the data that it was trained on and to gestures that have a similar lighting conditions.
but when used to detect hand signs of a random conditions it has a barely working accuracy.

# Key pressing with hand sign
Similar to the sign language classifier, To run the hand-sign-to-keypress program, type: *python Keydetect.py path-to-video-of-recorded-hand-directions* in the terminal.
It has a very good accuracy provided that it has a decent lighting conditions.
If it detects a hand directed upwards it presses UP-key, and similar to that all the other three directions.
Specifically, the hand must be in the top left quarter of the video as the program is suited to detect hands in this position.


In order to make any of the programs work in realtime, change the line of *cap = cv.VideoCapture(sys.argv[1])* to *cap = cv.VideoCapture(0)*
