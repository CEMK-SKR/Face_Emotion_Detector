import cv2
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model

new_model = load_model('./face_detector_model.h5')

emotions = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

# Load the Cascade Classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#startt  web cam
cap = cv2.VideoCapture(0)

while True:
    
    #read image from webcam
    respose, color_img = cap.read()

    # Convert to grayscale
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray_img, 1.4, 7)
    #display rectrangle
    for (x, y, w, h) in faces:
        cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        roi_color = color_img[y:y+h, x:x+w]
        final_image = cv2.resize(roi_color, (48, 48))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image/255.0
        prediction = new_model.predict(final_image)
        print(np.argmax(prediction))
        text = emotions[np.argmax(prediction)]
        org = (x, y-10)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0, 0, 255)
        lineType = cv2.LINE_4

        #text, org, fontFace, fontScale, color[, thickness

        img_text = cv2.putText(color_img, text, org, fontFace,
                            fontScale, color, lineType)

    # display image
    cv2.imshow('img', color_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
