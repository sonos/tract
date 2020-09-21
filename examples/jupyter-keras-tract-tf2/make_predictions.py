import tensorflow as tf
import numpy as np
import warnings

warnings.filterwarnings("ignore")
X = np.random.random((10, 100))
new_model = tf.keras.models.load_model('my_model')
    
    
predictions = new_model.predict(X)

print(f"result: {predictions}")