#!/usr/bin/env python
# coding: utf-8

# In[48]:


from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm


# In[49]:


HEIGHT = 150
WIDTH = 150

TRAIN_DIR = "C:\\Users\\admin\\Desktop\\anotherdata\\train"
TEST_DIR = "C:\\Users\\admin\\Desktop\\anotherdata\\testagain"

BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
        horizontal_flip=True, 
        vertical_flip=False, 
        rotation_range=10,
        rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                    target_size=(HEIGHT, WIDTH), 
                                                    batch_size=BATCH_SIZE)

test_generator = train_datagen.flow_from_directory(TEST_DIR, 
                                                    target_size=(HEIGHT, WIDTH), 
                                                    batch_size=BATCH_SIZE, 
                                                    shuffle = False)


# In[50]:


# Building a Sequential Model using Sequential function

#create model
model = Sequential()

#add model layers
model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(HEIGHT, WIDTH, 3)))
model.add(Conv2D(16, kernel_size=3, activation='relu'))

model.add(MaxPooling2D())

model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(MaxPooling2D())

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Print the snapshot of the model
model.summary()


# In[51]:


# Compiling your CNN Model

NUM_EPOCHS = 25
num_train_images = train_generator.samples

adam = Adam(lr=0.0001)
model.compile(adam, loss='squared_hinge', metrics=['accuracy'])


# In[52]:


# Adding a Model Checkpoint (saving our model after every epoch)

filepath="model_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
#checkpoint = ModelCheckpoint(filepath, monitor=["val_acc"], verbose=1, mode='max')


# In[53]:


# Training your CNN Model using the generator that we created earlier

history = model.fit(train_generator, 
                              validation_data=test_generator,
                              epochs=NUM_EPOCHS,
                              steps_per_epoch=num_train_images // BATCH_SIZE, 
                              shuffle=True, 
                              callbacks=[checkpoint])


# In[54]:


# plot the evolution of Loss and Acuracy on the train and validation sets

import matplotlib.pyplot as plt

plt.figure(figsize=(22, 8))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()


# In[55]:


test_generator.class_indices.keys()


# In[56]:


from mlxtend.plotting import plot_confusion_matrix


# In[57]:


#Confution Matrix and Classification Report
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)


# In[58]:


print('Confusion Matrix')
target_names = test_generator.class_indices.keys()
plot_confusion_matrix(confusion_matrix(test_generator.classes, y_pred), target_names)


# In[59]:


print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=target_names))


# In[60]:


np.set_printoptions(suppress=True)


# In[61]:


import pandas as pd

target_names = test_generator.class_indices.keys()

image_path="C:\\Users\\admin\\Desktop\\anotherdata\\test\\cancer\\000108.png"
img = image.load_img(image_path, target_size=(HEIGHT, WIDTH))

img = np.expand_dims(img, axis=0)
img = img/255

preds = model.predict(img).flatten()

pd.DataFrame(preds.reshape(1, 2), columns=target_names)


# In[ ]:





# In[ ]:




