import cv2
import sys

# no Need for sys.argv[1] 
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
	# Captures frame by frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect face
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor = 1.1,
		minNeighbors = 5,
		minSize = (30, 50),
		flags = cv2.CASCADE_SCALE_IMAGE
	)

	# Detect eyes
	eyes = eyeCascade.detectMultiScale(
		gray,
		scaleFactor = 1.1,
		minNeighbors = 5,
		minSize = (30, 50),
		flags = cv2.CASCADE_SCALE_IMAGE
	)

	# draw rectangle around face
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)

	# draw rectangles around eyes
	for (x, y, w, h) in eyes:
		cv2.rectangle(frame, (x,y), (x+y+w+h), (255,0,0), 2)

	# display resulting frame
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# release capture once everything is done
cap.release()
cv2.destroyAllWindows()
