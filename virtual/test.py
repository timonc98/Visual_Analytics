import tensorflow as tf
model = tf.keras.models.load_model('mein_modell.h5')
model.get_layer('dropout_2').output
