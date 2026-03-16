import tensorflow as tf
from PIL import Image
import numpy as np

# Load trained model
model = tf.keras.models.load_model("digit_model.h5")

# Load image
img = Image.open("digits.png").convert("L")

# Resize to 28x28 (same as MNIST dataset)
img = img.resize((28, 28))

# Convert to numpy array
img_array = np.array(img)

# Normalize pixel values
img_array = img_array / 255.0

# Reshape for model
img_array = img_array.reshape(1, 28, 28)

# Predict digit
prediction = model.predict(img_array)

digit = np.argmax(prediction)

print("Predicted Digit:", digit)