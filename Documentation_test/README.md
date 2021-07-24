
## Facial Emotion Detector

Install my-project with npm

```bash
!pip install opencv-python
!pip install tensorflow
!pip install imblearn
```
importing libraries
```bash
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
``` 
```bash
df = pd.read_csv('icml_face_data.csv')
df.head(5)
```

<div align="center">
  <img height="60%" src="https://github.com/CEMK-SKR/Face_Emotion_Detector/blob/main/Documentation/screenshots/1.jpg?raw=true"/>
</div>

```bash 
df.dtypes
```
```bash
df.drop(['usage'],inplace=True, axis=1)
df.head(5)
```
<div align="center">
  <img height="60%" src="https://github.com/CEMK-SKR/Face_Emotion_Detector/blob/main/Documentation/screenshots/2.jpg?raw=true"/>
</div>

```bash
df.shape
```
```bash
df.isnull().sum()
```
```bash
df.drop_duplicates(inplace=True)
df.head(5)
```
<div align="center">
  <img height="60%" src="https://github.com/CEMK-SKR/Face_Emotion_Detector/blob/main/Documentation/screenshots/3.jpg?raw=true"/>
</div>

```bash
df.shape
```
```bash
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
df1 = df['emotion']
df1.head(10)
```
```bash
df2 = df.drop('emotion', axis=1)
df2.shape
```
```bash
len(df2)
```
```bash
image_array = np.zeros(shape=(len(df2), 48, 48))
```

```bash
 for i, row in enumerate(df2.index):
        image = np.fromstring(df2.loc[row, ' pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image
```
```bash
for i in range(10):
    plt.matshow(image_array[i])
```
```bash
from sklearn.model_selection import train_test_split
(X_train,X_test,y_train,y_test) = train_test_split(image_array,df1,test_size=0.2,random_state=10)

```
```bash
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
X_train=X_train.reshape(-1,48,48,1)
X_test = X_test.reshape(-1,48,48,1)
X_train.shape
X_train_rescaled = X_train/255
```
```bash
from tensorflow import keras
data_augmentation = keras.Sequential(
    [
        keras.layers.experimental.preprocessing.RandomRotation(0.1),
        keras.layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)
```

```bash
model = keras.Sequential([
    data_augmentation,
    keras.layers.Conv2D(32, (3,3),activation='relu', padding='same',input_shape=(48, 48,1)),
    keras.layers.Dropout(0.2),
    
    keras.layers.Conv2D(32, (5,5),activation='relu',padding='same'),
    keras.layers.Dropout(0.2),
    
    keras.layers.Conv2D(32, (3,3),activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(emotions), activation='softmax')
])

model.compile(optimizer='adam',
             loss = 'sparse_categorical_crossentropy',
             metrics='accuracy')

model.fit(X_train_rescaled,y_train,epochs=20)
```

<div align="center">
  <img height="60%" src="https://github.com/CEMK-SKR/Face_Emotion_Detector/blob/main/Documentation/screenshots/4.jpg?raw=true"/>
</div>

```bash
y_pred = model.predict(X_test_rescaled)
```
```bash
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
```
```bash
plt.matshow(X_test_rescaled[10])
```
```bash
y_pred_final= []
for i in range(len(y_pred)):
    y_pred_final.append(np.argmax(y_pred[i]))
```
```bash
emotions[(y_pred_final[10])]
```
```bash
print(classification_report(y_test,y_pred_final))
```

<div align="center">
  <img height="60%" src="https://github.com/CEMK-SKR/Face_Emotion_Detector/blob/main/Documentation/screenshots/5.jpg?raw=true"/>
</div>

```bash
import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_final)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
```

<div align="center">
  <img height="60%" src="https://github.com/CEMK-SKR/Face_Emotion_Detector/blob/main/Documentation/screenshots/result.jpg?raw=true"/>
</div>
