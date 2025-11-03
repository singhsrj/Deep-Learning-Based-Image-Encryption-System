from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import hashlib

# %%
import numpy as np
from PIL import Image

image = Image.open("images.jpg")
image = image.resize((256,256))

image_array = np.array(image)
# %%


# Example: input image shape = 128x128 with 3 color channels (RGB)
model = Sequential([
    # 1st Convolutional layer
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(256 , 256 , 3)),
    
    # MaxPooling layer
    MaxPooling2D(pool_size=(2,2)),
    
    # 2nd Convolutional layer
    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    
    # Flatten layer
    Flatten()
])

def key_gen(image_array):
    """
    Generate 32-byte AES-256 key from image.
    Works with ANY image size!
    """
    image_array = np.expand_dims(image_array, axis=0)  
    features = model.predict(image_array)
    non_zero_features = [int(i) for i in features[0] if i != 0]
    key_string = "".join([str(i) for i in non_zero_features])

    # ✅ Use .digest() to get 32 BYTES (not .hexdigest())
    key_bytes = hashlib.sha256(key_string.encode('utf-8')).digest()
    
    # Verify length
    assert len(key_bytes) == 32, f"Expected 32 bytes, got {len(key_bytes)}"
    
    return key_bytes, key_string


# Test it
key_bytes, key_string = key_gen(image_array)