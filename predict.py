import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

class flowers:
    def __init__(self, filename):
        self.filename = filename
        self.model = load_model('model.h5')  # Load the model once during initialization
        self.graph = tf.compat.v1.get_default_graph()  # Use this if you're using TensorFlow 1.x
        self.sess = tf.compat.v1.Session()  # Create a session
        with self.sess.as_default():
            with self.graph.as_default():
                # Ensure that variables are initialized
                self.sess.run(tf.compat.v1.global_variables_initializer())
        self.class_indices = {
            'daisy': 0,
            'dandelion': 1,
            'rose': 2,
            'sunflower': 3,
            'tulip': 4
        }  # Define class indices if they are fixed

    def predictionflowers(self):
        with self.graph.as_default():
            with self.sess.as_default():
                imagename = self.filename
                test_image = image.load_img(imagename, target_size=(64, 64))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis=0)
                result = self.model.predict(test_image)

                # Get the index of the predicted class
                predicted_class_index = np.argmax(result, axis=1)[0]

                # Get the class label from the index
                prediction = list(self.class_indices.keys())[list(self.class_indices.values()).index(predicted_class_index)]

                # Return the prediction
                return [{"image": prediction}]
