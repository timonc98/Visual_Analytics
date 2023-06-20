from PIL import Image

import cv2

custom_image = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-1008.png"
image = cv2.imread(custom_image)
height, width, channels = image.shape

print("Bildgröße: {}x{}x{}".format(width, height, channels))
