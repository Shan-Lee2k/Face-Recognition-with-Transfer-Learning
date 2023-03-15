import cv2

# Load the Haar Cascade Classifier XML file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the input image
img = cv2.imread('venv/Image_Test/ShanGB.jpg')

#Resize image 400x400
#img = cv2.resize(img, (600,600))

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x-10, y-5), (x+w-10, y+h), (0, 255, 0), 2)

# Display the output image
cv2.imshow('Output', img)

# Wait for a key press to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
