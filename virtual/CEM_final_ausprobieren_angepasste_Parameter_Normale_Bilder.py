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
import tensorflow as tf
tf.keras.backend.clear_session()

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

# Laden Sie die Modelle
ae = keras.models.load_model('covid_ae.h5')
cnn = keras.models.load_model('covid_cnn.h5')

decoded_imgs = ae.predict(x_test)


n = 5
plt.figure(figsize=(70, 3))
for i in range(1, n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(70, 70, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(70, 70, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



# Laden Sie das gewünschte Bild
custom_image = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Normal/images/Normal-210.png"

# Laden und Skalieren des Bildes auf 70x70 Pixel
X = cv2.imread(custom_image)
X = cv2.resize(X, (70, 70))

# Reshape und Normalisierung des Bildes
X = X.reshape((1, 70, 70, 3))
X = X.astype(np.float32) / 255.


cnn.predict(X).argmax(), cnn.predict(X).max()

shape = X.shape
mode = 'PN' 
kappa = .2 
beta = 0.1 
gamma = 100  
c_init = 10.                
c_steps = 10 
max_iterations = 1000  
feature_range = (x_train.min(axis=0).reshape(shape)-.1,  # feature range for the perturbed instance
                 x_train.max(axis=0).reshape(shape)+.1)
clip = (-1000.,1000.)  
lr = 1e-2  
no_info_val = -1. 


cem = CEM(cnn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range,
          gamma=gamma, ae_model=ae, max_iterations=max_iterations,
          c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)


explanation = cem.explain(X, verbose=False)

if explanation.PN_pred is not None:
    index_pn = explanation.PN_pred.argmax()
    image_path_pn = data['path'].iloc[index_pn]
else:
    print("CEM-Erklärung für Pertinent Negative war nicht erfolgreich.")

if explanation.PN is not None:
    plt.imshow(explanation.PN.reshape(70, 70, 3))
    plt.show()
else:
    print("Kein Pertinent Negative gefunden.")

mode = 'PP'
shape = X.shape
kappa = .2  # minimum difference needed between the prediction probability for the perturbed instance on the
            # class predicted by the original instance and the max probability on the other classes 
            # in order for the first loss term to be minimized
beta = .1  # weight of the L1 loss term
c_init = 10.  # initial weight c of the loss term encouraging to predict a different class (PN) or 
              # the same class (PP) for the perturbed instance compared to the original instance to be explained
c_steps = 10  # nb of updates for c
max_iterations = 2000  # nb of iterations per value of c
feature_range = (x_train.min(axis=0).reshape(shape)-.1,  # feature range for the perturbed instance
                 x_train.max(axis=0).reshape(shape)+.1)  # can be either a float or array of shape (1xfeatures)
clip = (-1000.,1000.)  # gradient clipping
lr = 1e-2  # initial learning rate
no_info_val = -1.  # Example value for no_info_val

cem = CEM(cnn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range, 
          max_iterations=max_iterations, c_init=c_init, c_steps=c_steps, 
          learning_rate_init=lr, clip=clip, no_info_val=no_info_val)

explanation = cem.explain(X)

if explanation.PP_pred is not None:
    index_pp = explanation.PP_pred.argmax()
    image_path_pp = data['path'].iloc[index_pp]
else:
    print("CEM-Erklärung für Pertinent Positive war nicht erfolgreich.")

if explanation.PP is not None:
    plt.imshow(explanation.PP.reshape(70, 70, 3))
    plt.show()
else:
    print("Kein Pertinent Positive gefunden.")
