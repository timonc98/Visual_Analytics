from PIL import Image

import cv2

custom_image = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID-828_PN_neu.png"
# custom_image = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual\Dataset/COVID/images/COVID-3.png"
image = cv2.imread(custom_image)
height, width, channels = image.shape

print("Bildgröße: {}x{}x{}".format(width, height, channels))
