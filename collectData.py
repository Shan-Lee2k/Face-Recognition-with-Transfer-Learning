import cv2
import os
import time

# Set up the camera
cap = cv2.VideoCapture(0)

# Set up the face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a directory to save the face images
if not os.path.exists('Datasets'):
    os.makedirs('Datasets')

# Set the initial count of face images to 0
count = 0
label = input('Enter the label for this individual: ')

# Loop until 100 face images are collected
while count < 100:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces and save the cropped face images
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Save the cropped face image to the faces directory with a label for each individual
        face = frame[y:y+h, x:x+w]
        cv2.imwrite('Datasets/{}_{}.jpg'.format(label, count), face)
        print(label + str(count))
        # Increment the count of face images
        count += 1

        # Exit the loop if 100 face images have been collected
        if count == 100:
            break
        # Add time sleep delay 0.5 s
        time.sleep(0.5)
    # Display the frame
    cv2.imshow('Collecting Faces', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
