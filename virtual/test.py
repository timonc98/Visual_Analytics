import cv2
image_size = (256, 256) 
# Laden des Testbildes
lung_image = cv2.imread('virtual\Dataset\Viral Pneumonia\images\Viral Pneumonia-78.png', cv2.IMREAD_GRAYSCALE)
lung_image = cv2.resize(lung_image, image_size)

# Bild anzeigen
cv2.imshow('Lung Image', lung_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
