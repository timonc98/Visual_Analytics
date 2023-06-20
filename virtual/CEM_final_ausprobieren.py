import tensorflow as tf
import keras as keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
from alibi.explainers import CEM

tf.get_logger().setLevel(40) 
tf.compat.v1.disable_v2_behavior() 

# Laden Sie die Modelle
ae = keras.models.load_model('covid_ae.h5')
cnn = keras.models.load_model('covid_cnn.h5')

# Laden Sie das gewünschte Bild
custom_image = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-38.png"

# Laden und Skalieren des Bildes auf 70x70 Pixel
X = cv2.imread(custom_image)
X = cv2.resize(X, (70, 70))

# Reshape und Normalisierung des Bildes
X = X.reshape((1, 70, 70, 3))
X = X.astype(np.float32) / 255.

shape = X.shape

# Definieren Sie die Hyperparameter für CEM
mode = 'PP'
kappa = 0.
beta = 0.1
gamma = 100
c_init = 1.
c_steps = 1
max_iterations = 1000
feature_range = (0., 1.)
clip = (-1000., 1000.)
lr = 1e-2
no_info_val = -1.

# Erzeugen Sie die CEM-Erklärung für Pertinent Positive
cem = CEM(cnn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range,
          gamma=gamma, ae_model=ae, max_iterations=max_iterations,
          c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)

explanation = cem.explain(X)

if explanation.PP is not None:
    plt.imshow(explanation.PP.reshape(70, 70, 3))
    plt.show()
else:
    print("Kein Pertinent Positive gefunden.")
