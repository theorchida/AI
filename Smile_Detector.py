import cv2

# pre trainted face/smile detector
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
trained_smile_data = cv2.CascadeClassifier('haarcascade_smile.xml')

# read img
#img = cv2.imread('pic.jpg')
# open webcam
webcam = cv2.VideoCapture(0)

# iterate forever
while True:
    (successful_frame_read, frame) = webcam.read()

    if successful_frame_read:
        # convert to grayscale
        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    #detect face
    face_coordinates = trained_face_data.detectMultiScale(frame)
        
    # draw rectangle around face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # since frame is numpy array as well
        # get subframe containing face (using numpy slicing)
        the_face = frame[y:y+h, x:x+w]

        # to run smile on face, cvt face to gray
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # find smile within face
        smile_coordinates = trained_smile_data.detectMultiScale(face_grayscale, scaleFactor = 1.8, minNeighbors = 30)

        # draw rec around smiles
        # for (x_, y_, w_, h_) in smile_coordinates:
            # cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (0, 255, 255), 2)

        #label face as smiling
        if len(smile_coordinates) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=2,
            fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    # show image
    cv2.imshow('Smile Detector', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()
cv2.destroyWindow('Smile Detector')



print('code completed')
