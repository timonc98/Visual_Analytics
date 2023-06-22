import pandas as pd
import numpy as np
import cv2
import os
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from captum.attr import Saliency
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Laden der Daten
viral_pneumonia_metadata = pd.read_excel('virtual/Dataset/Viral_Pneumonia.metadata.xlsx')
normal_metadata = pd.read_excel('virtual/Dataset/Normal.metadata.xlsx')
lung_opacity_metadata = pd.read_excel('virtual/Dataset/Lung_Opacity.metadata.xlsx')
covid_metadata = pd.read_excel('virtual/Dataset/COVID.metadata.xlsx')

# Größe der Bilder festlegen
image_size = (256, 256)

# Daten in eine Liste mit Dictionaries umwandeln
data = []
for folder, label in [('virtual/Dataset/Viral Pneumonia', 0),('virtual/Dataset/Normal', 1),('virtual/Dataset/Lung_Opacity', 2), ('virtual/Dataset/COVID', 3)]:
    for subfolder in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, subfolder)):
            images = []
            masks = []
            for filename in os.listdir(os.path.join(folder, subfolder)):
                image = cv2.imread(os.path.join(folder, subfolder, filename), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, image_size)
                images.append(image)
                if 'mask' in filename:
                    mask = cv2.imread(os.path.join(folder, subfolder, filename), cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, image_size)
                    masks.append(mask)
            data.append({'images': np.array(images), 
                         'masks': np.array(masks),
                         'label': label})

# Labels extrahieren
train_labels = []
for d in data:
    train_labels.extend([d['label']] * len(d['images']))
train_labels = np.array(train_labels)

# Daten in das erforderliche Format bringen
all_data = []
for d in data:
    all_data.extend(d['images'])

train_data, test_data, train_labels, test_labels = train_test_split(all_data, train_labels, test_size=0.2, random_state=42)

train_images = np.array(train_data).reshape(-1, 256, 256, 1)
test_images = np.array(test_data).reshape(-1, 256, 256, 1)

# CNN-Modell definieren
model = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(256, 256, 1)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4)
])

# Modell kompilieren
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Modell trainieren
batch_size = 32
epochs = 9
history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_labels), verbose=1)


# Plot Trainings- und Testgenauigkeiten
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

image_size = (256, 256)

# Laden des COVID-Bildes
covid_image = cv2.imread('virtual/Dataset/Testbilder/covid.png', cv2.IMREAD_GRAYSCALE)
covid_image = cv2.resize(covid_image, image_size)
covid_image = np.array(covid_image).reshape(-1, 256, 256, 1)

#Überprüfen ob die Bilder geladen werden
plt.imshow(covid_image.reshape(256,256), cmap='gray')
plt.show()

# Vorhersage für das Bild treffen
prediction = model.predict(covid_image)
predicted_class = np.argmax(prediction)

# Ausgabe der Vorhersage
if predicted_class == 3:
    print('Das Bild zeigt eine COVID-19-Lunge')
    print('Die Vorhersage-Wahrscheinlichkeit beträgt:', prediction[0][predicted_class])
else:
    print('Das Bild zeigt kein COVID-19')

image_size = (256, 256)

non_covid_image = cv2.imread('virtual\Dataset\Testbilder\Viral Pneumonia-25.png', cv2.IMREAD_GRAYSCALE)
non_covid_image = cv2.resize(non_covid_image, image_size)
non_covid_image = np.array(non_covid_image).reshape(-1, 256, 256, 1)

#Überprüfen ob die Bilder geladen werden
plt.imshow(non_covid_image.reshape(256,256), cmap='gray')
plt.show()


# Vorhersage für das Bild treffen
prediction = model.predict(non_covid_image)
predicted_class = np.argmax(prediction)

# Ausgabe der Vorhersage
if predicted_class == 3:
    print('Das Bild zeigt eine COVID-19-Lunge')
    print('Die Vorhersage-Wahrscheinlichkeit beträgt:', prediction[0][predicted_class])
else:
    print('Das Bild zeigt kein COVID-19')
    #aa