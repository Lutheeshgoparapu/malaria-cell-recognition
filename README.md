# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
The problem at hand is the automatic classification of red blood cell images into two categories: parasitized and uninfected.
Malaria-infected red blood cells, known as parasitized cells, contain the Plasmodium parasite
uninfected cells are healthy and free from the parasite.
The goal is to build a convolutional neural network (CNN) model capable of accurately distinguishing between these two classes based on cell images.
Traditional methods of malaria diagnosis involve manual inspection of blood smears by trained professionals, which can be time-consuming and error-prone.
Automating this process using deep learning can significantly speed up diagnosis, reduce the workload on healthcare professionals, and improve the accuracy of detection.
Our dataset comprises 27,558 cell images, evenly split between parasitized and uninfected cells.
These images have been meticulously collected and annotated by medical experts, making them a reliable source for training and testing our deep neural network.

![image](https://github.com/Adithya-Siddam/malaria-cell-recognition/assets/93427248/d627edea-10a6-452b-a15e-e846d84a396b)

## Neural Network Model

![image](https://github.com/Adithya-Siddam/malaria-cell-recognition/assets/93427248/ecf5a2cb-0e7d-4a22-816a-a3bff1008502)

## DESIGN STEPS
### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Read the dataset

### STEP 3:
Create an ImageDataGenerator to flow image data

### STEP 4:
Build the convolutional neural network model and train the model

### STEP 5:
Fit the model

### STEP 6:
Evaluate the model with the testing data

### STEP 7:
Fit the model

### STEP 8:
Plot the performance plot

## PROGRAM

### Name:
```
Name : G.Lutheesh

Reg No : 212221230029
```
```
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf

my_data_dir = 'dataset/cell_images'

os.listdir(my_data_dir)

test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'

os.listdir(train_path)

len(os.listdir(train_path+'/uninfected/'))

len(os.listdir(train_path+'/parasitized/'))

os.listdir(train_path+'/parasitized')[7]

para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[7])

plt.imshow(para_img)

dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

sns.jointplot(x=dim1,y=dim2)

image_shape = (130,130,3)

image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )

image_gen.flow_from_directory(train_path)

image_gen.flow_from_directory(test_path)

model = models.Sequential()
model.add(layers.Input(shape=(130,130,3))) 
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),padding="same",activation='relu'))
model.add(layers.AvgPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),padding="same",activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu')) 
model.add(layers.Dense(1, activation ='sigmoid'))
model.summary()
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

batch_size = 16

train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')

train_image_gen.batch_size

len(train_image_gen.classes)

train_image_gen.total_batches_seen

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)

train_image_gen.class_indices

results = model.fit(train_image_gen,epochs=2,
validation_data=test_image_gen )

losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot()

model.metrics_names

model.evaluate(test_image_gen)

pred_probabilities = model.predict(test_image_gen)

test_image_gen.classes

predictions = pred_probabilities > 0.5

print(classification_report(test_image_gen.classes,predictions))

confusion_matrix(test_image_gen.classes,predictions)

plt.title("Model prediction: "+("Parasitized" if pred  else "Uninfected")+"\nActual Value: "+str(dir_))
plt.axis("off")
plt.imshow(img)
plt.show()
from tensorflow.keras.preprocessing import image
img = image.load_img('mui.jpg')
img=tf.convert_to_tensor(np.asarray(img))
img=tf.image.resize(img,(130,130))
img=img.numpy()
type(img)
plt.imshow(img)
x_single_prediction = bool(model.predict(img.reshape(1,130,130,3))>0.6)
print(x_single_prediction)
if(x_single_prediction==1):
    print("Cell is UNINFECTED")
else:
    print("Cell is PARASITIZED")
```



## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![331594574-f720e436-1f40-424e-8a55-1f354412dd9e](https://github.com/Lutheeshgoparapu/malaria-cell-recognition/assets/94154531/089a51d9-196a-4e7b-84a0-a300c8ac305a)



### Classification Report
![331594658-ab9392fc-cb77-4878-828d-54975e3a27fb](https://github.com/Lutheeshgoparapu/malaria-cell-recognition/assets/94154531/e534b5e6-5453-4719-8631-043f0a1219b8)



### Confusion Matrix
![331594768-93228326-ddc5-4c6d-863f-99e8091f297d](https://github.com/Lutheeshgoparapu/malaria-cell-recognition/assets/94154531/961f6574-aaee-4d0f-90ad-fa357af9ec0d)



### New Sample Data Prediction
![331595387-6e2ab739-fbec-4944-a0e7-c729e22ed388](https://github.com/Lutheeshgoparapu/malaria-cell-recognition/assets/94154531/d318690b-7de7-4e18-8abc-7756cedcfd65)

## RESULT:
The model's performance is evaluated through training and testing, and it shows potential for assisting healthcare professionals in diagnosing malaria more efficiently and accurately.




