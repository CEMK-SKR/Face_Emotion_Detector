
## Facial Emotion Detector

Install my-project with npm

```bash
!pip install opencv-python
!pip install tensorflow
```
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
```bash
df.dtypes
```
```bash
df.drop(['usage'],inplace=True, axis=1)
df.head(5)
```
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

model = keras.Sequential([
    keras.layers.Conv2D(32, (8,8),activation='relu', input_shape=(48, 48,1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),

    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
             loss = 'sparse_categorical_crossentropy',
             metrics='accuracy')

model.fit(X_train_rescaled,y_train,epochs=20)
```
```bash
y_pred = model.predict(X_test)
```
```bash
from sklearn.metrics import classification_report
```
```bash
y_test.head(20)
```
```bash
plt.matshow(X_test[6])
```
```bash
plt.matshow(X_test[6])
```