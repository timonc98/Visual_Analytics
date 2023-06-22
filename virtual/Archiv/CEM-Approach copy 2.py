import tensorflow as tf
tf.get_logger().setLevel(40) 
tf.compat.v1.disable_v2_behavior() 
import keras as keras
from keras import backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D
from keras.models import Model, load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import cv2
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

import os
from glob import glob
from PIL import Image

import matplotlib 
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os
from alibi.explainers import CEM

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) 

# Get Data
levels = ['Normal', 'COVID']

#Timon Dateipfad
path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset"
data_dir = os.path.join(path)

data = []
for id, level in enumerate(levels):
    for file in os.listdir(os.path.join(data_dir, level+'/'+'images')):
        data.append([level +'/' +'images'+ '/'+file, level])

data = pd.DataFrame(data, columns = ['image_file', 'corona_result'])

data['path'] = path + '/' + data['image_file']
data['corona_result'] = data['corona_result'].map({'Normal': 'Negative', 'COVID': 'Positive'})

data.head()

# Count number of Samples
print('Number of Duplicated Samples: %d'%(data.duplicated().sum()))
print('Number of Total Samples: %d'%(data.isnull().value_counts()))

# Show Image samples
data['image'] = data['path'].map(lambda x: np.asarray(Image.open(x).resize((75,75))))

data.head()

n_samples = 3


fig, m_axs = plt.subplots(2, n_samples, figsize = (6*n_samples, 3*4))

for n_axs, (type_name, type_rows) in zip(m_axs, data.sort_values(['corona_result']).groupby('corona_result')):
    n_axs[1].set_title(type_name, fontsize = 15)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state = 1234).iterrows()):       
        picture = c_row['path']
        image = cv2.imread(picture)
        c_ax.imshow(image)
        c_ax.axis('off')

# Multiplot Visualization
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title = ""):
    
    fig, myaxes = plt.subplots(figsize = (15, 8), nrows = 2, ncols = ncols, squeeze = False)
    fig.suptitle(main_title, fontsize = 18)
    fig.subplots_adjust(wspace = 0.3)
    fig.subplots_adjust(hspace = 0.3)
    
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize = 15)
        
    plt.show()


all_data = []

# Storing images and their labels into a list for further Train Test split

for i in range(len(data)):
    image = cv2.imread(data['path'][i])
    image = cv2.resize(image, (70, 70)) / 255.0
    label = 1 if data['corona_result'][i] == "Positive" else 0
    all_data.append([image, label])

x = []
y = []

for image, label in all_data:
    x.append(image)
    y.append(label)

# Converting to Numpy Array    
x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42)

print(x_train.shape, x_test.shape, x_val.shape, y_train.shape, y_test.shape, y_val.shape)

# CNN-Model Definition
def cnn_model():
    x_in = Input(shape=(70, 70, 3))
    x = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(x_in)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv2D(filters=16, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv2D(filters=4, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x_out = Dense(10, activation='softmax')(x)

    cnn = Model(inputs=x_in, outputs=x_out)
    cnn.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), optimizer='adam', metrics=['accuracy'])

    return cnn

cnn = cnn_model()
cnn.summary()
cnn.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1)
cnn.save('covid_cnn.h5', save_format='h5')

cnn = load_model('covid_cnn.h5')
score = cnn.evaluate(x_test, y_test, verbose=0)
print('Test accuracy: ', score[1])

# Autoencoder Model Definition
def ae_model():
    x_in = Input(shape=(70, 70, 3))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x_in)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(3, (3, 3), activation=None, padding='same')(x)

    autoencoder = Model(x_in, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

ae = ae_model()
ae.summary()
ae.fit(x_train, x_train, batch_size=128, epochs=1, validation_data=(x_test, x_test), verbose=0)
ae.save('covid_ae.h5', save_format='h5')

ae = load_model('covid_ae.h5')

from keras.utils import load_img, img_to_array, array_to_img

img_path = ["C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Viral Pneumonia/images/Viral Pneumonia-1003.png",
                      "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Normal/images/Normal-10004.png",
                      "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Lung_Opacity/images/Lung_Opacity-1000.png",
                      "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-1008.png"]


model_builder = keras.applications.xception.Xception
img_size = (299, 299)
preprocess_input = keras.applications.xception.preprocess_input
imag = []


def get_img_array(img_path, size):
    img = load_img(img_path, target_size = size) 
    array = img_to_array(img)
    array = np.expand_dims(array, axis = 0)
    return array

def save_and_display_images(img_path, cam_path = "cam.jpg"):
    img = load_img(img_path)
    img = img_to_array(img)

    superimposed_img = img
    superimposed_img = array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    
    imag.append(cv2.imread(img_path))
    imag.append(cv2.imread("./cam.jpg"))

for i in range(len(img_path)):
    save_and_display_images(img_path[i])

for i in img_path:
    img_array = preprocess_input(get_img_array(i, size = img_size))
    model = model_builder(weights = "imagenet")
    model.layers[-1].activation = None
    decoded_imgs = model.predict(img_array)

n = 4
plt.figure(figsize=(70, 3))
for i in range(1, n+1):
    # display original
    plt.subplot(1, 2, 1)
    plt.imshow(img_array.reshape(70, 70, 3))
    plt.axis('off')

    # display reconstruction
    plt.subplot(1, 2, 2)
    plt.imshow(decoded_imgs.reshape(70, 70, 3))
    plt.axis('off')

plt.show()




idx = 0

X = img_array[idx].reshape((1,) + img_array[idx].shape)


plt.imshow(X.reshape(70, 70, 3))

cnn.predict(X).argmax(), cnn.predict(X).max()

mode = 'PN'  
shape = (1,) + x_train.shape[1:]  
kappa = 0. 
beta = .1 
gamma = 100  
c_init = 1.  
              
c_steps = 1 
max_iterations = 1000  
feature_range = (x_train.min(),x_train.max())  
clip = (-1000.,1000.)  
lr = 1e-2  
no_info_val = -1. 
cem = CEM(cnn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range,
          gamma=gamma, ae_model=ae, max_iterations=max_iterations,
          c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)

explanation = cem.explain(X, verbose=False)

print('Pertinent negative prediction: {}'.format(explanation.PN_pred))
plt.imshow(explanation.PN.reshape(70, 70, 3))
plt.show()

mode = 'PP'

cem = CEM(cnn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range,
          gamma=gamma, ae_model=ae, max_iterations=max_iterations,
          c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)

explanation = cem.explain(X)

print('Pertinent positive prediction: {}'.format(explanation.PP_pred))
plt.imshow(explanation.PP.reshape(70, 70, 3))
plt.show()


