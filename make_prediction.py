import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf

# Load the trained model
model = load_model('model.h5')
graph = tf.compat.v1.get_default_graph()

# Prepare the image for prediction
test_image = image.load_img('Rose.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Predict the class
with graph.as_default():
    result = model.predict(test_image)

# Define class indices
class_indices = {
    'daisy': 0,
    'dandelion': 1,
    'rose': 2,
    'sunflower': 3,
    'tulip': 4
}

# Get the index of the predicted class
predicted_class_index = np.argmax(result, axis=1)[0]

# Get the class label from the index
prediction = list(class_indices.keys())[list(class_indices.values()).index(predicted_class_index)]

# Print the prediction
print(prediction)
