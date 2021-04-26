import cv2

# our image
# img_file = 'car image.jpg'
video = cv2.VideoCapture('car pedestrian.mp4')

# our pretrained car classifier
car_tracker_file = 'cars.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

# create car/pedestrian classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# run forever until true
while True:
    # read current frame
    (read_successful, frame) = video.read()

    #safe coding
    if read_successful:
        #cvt to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    # detect cars and pedestrian
    car_coordinates = car_tracker.detectMultiScale(frame)
    pedestrian_coordinates = pedestrian_tracker.detectMultiScale(frame)
    
    # draw rectangle around cars
    for (x, y, w, h) in car_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # draw rectangle around pedestrian
    for (x, y, w, h) in pedestrian_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # display image with cars spotted
    cv2.imshow('Car and Pedestrian Detector', frame)

    # wait
    key = cv2.waitKey(1)

    # stop if q key is presses
    if key == 81 or key == 113:
        break

video.release()
cv2.destroyWindow('Car and Pedestrian Detector')

# create opencv image
# img = cv2.imread(img_file)

# create car classifier
# car_tracker = cv2.CascadeClassifier(classifier_file)

# convert to grayscale
# grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect cars
# car_coordinates = car_tracker.detectMultiScale(grayscaled_img)

# draw rectangle around cars
# for (x, y, w, h) in car_coordinates:
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)

# display image with cars spotted
# cv2.imshow('Car Detector', img)

# wait
# cv2.waitKey(1)


print("Code completed")
