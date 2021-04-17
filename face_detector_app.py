import cv2
from random import randrange

# load pretrained data of face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose image to detect faces in
# img = cv2.imread('TH RDJ.png')
# or capture video from webcam
# 0 for default webcam and 'video_name.mp4' for video 
webcam = cv2.VideoCapture(0)

# iterate forever over frames
while True:
    # read current frame
    successful_frame_read, frame = webcam.read()

    # must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and give coordinates of face, no matter what size
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # draw rectangle
    # use randrange() for different rectangle color in every frame
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.imshow('Face Detector App', frame)
    key = cv2.waitKey(1)

    # Stop if q/Q key is pressed
    if key == 81 or key == 113:
        break

# release webcam
webcam.release()
cv2.destroyWindow('Face Detector App')

print('Code Completed')
