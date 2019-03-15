# utility file for emotion recognition from realtime webcam feed
import cv2
import sys
from keras.models import load_model
import time
import numpy as np
from decimal import Decimal
from model_utils import define_model, model_weights

# loads and resizes an image
def resize_img(image_path):
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (48, 48))
    return img
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
# runs the realtime emotion detection 
def realtime_emotions():
    # load keras model
    model = define_model()
    model = model_weights(model)
    print('Model loaded')

    # save location for image
    save_loc = 'save_loc/1.jpg'
    # numpy matrix for stroing prediction
    result = np.array((1,7))
    # for knowing whether prediction has started or not
    once = False
    # load haar cascade for face
    faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    # list of given emotions
    EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    # store the emoji coreesponding to different emotions
    emoji_faces = []
    for index, emotion in enumerate(EMOTIONS):

        emoji_faces.append(image_resize(cv2.imread('emojis/' + emotion + '.png', -1),height=200))

    # set video capture device , webcam in this case
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 640)  # WIDTH
    video_capture.set(4, 480)  # HEIGHT

    # save current time
    prev_time = time.time()

    # start webcam feed
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        # mirror the frame
        frame = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find face in the frame
        faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
        
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            # required region for the face
            roi_color = frame[y-10:y+h+10, x-10:x+w+10]

            # save the detected face
            cv2.imwrite(save_loc, roi_color)
            # draw a rectangle bounding the face
            cv2.rectangle(frame, (x-10, y-10),
                            (x+w+10, y+h+10), (15, 175, 61), 2)
            
            # keeps track of waiting time for emotion recognition
            curr_time = time.time()
            # do prediction only when the required elapsed time has passed
            if curr_time - prev_time >=1:
                # read the saved image
                img = cv2.imread(save_loc, 0)
                once = True
                if img is None:
                    continue
                # resize image for the model
                else:

                   img = cv2.resize(img, (48, 48))
                   img = np.reshape(img, (1, 48, 48, 1))
                   # do prediction
                   result = model.predict(img)
                   print(EMOTIONS[np.argmax(result[0])])
                    
                #save the time when the last face recognition task was done
                prev_time = time.time()

            if once == True:
                total_sum = np.sum(result[0])
                # select the emoji face with highest confidence
                emoji_face = emoji_faces[np.argmax(result[0])]
                for index, emotion in enumerate(EMOTIONS):
                    text = str(
                        round(Decimal(result[0][index]/total_sum*100), 2) ) + "%"
                    # for drawing progress bar
                    cv2.rectangle(frame, (100, index * 20 + 10), (100 +int(result[0][index] * 100), (index + 1) * 20 + 4),
                                    (255, 0, 0), -1)
                    # for putting emotion labels
                    cv2.putText(frame, emotion, (10, index * 20 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (7, 109, 16), 2)
                    # for putting percentage confidence
                    cv2.putText(frame, text, (105 + int(result[0][index] * 100), index * 20 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
               # im = cv2.imread(emoji_face)
                #ims=cv2.resize(im, (960, 540))
                cv2.imshow('video1', emoji_face)

                # overlay emoji on the frame for all the channels

            break

        # Display the resulting frame
        #cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
