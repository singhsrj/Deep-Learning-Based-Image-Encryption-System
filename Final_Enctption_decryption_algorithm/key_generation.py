import hashlib
import galois
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten


# Define BCH parameters (standard + strong code)
GF = galois.GF(2)
bch = galois.BCH(255, 131)

def fuzzy_key_gen(biometric_array):
    """
    Input: biometric feature array (noisy source)
    Output: AES key K, helper data P
    """

    # Step 1: Convert features → binary vector
    # (flatten & threshold)
    bio_bits = (biometric_array.flatten() % 2).astype(np.uint8)

    # Fit to BCH length
    bio_bits = bio_bits[:bch.n] if bio_bits.size >= bch.n else \
               np.pad(bio_bits, (0, bch.n - bio_bits.size))

    # Step 2: Encode to BCH codeword
    codeword = bch.encode(bio_bits[:bch.k])

    # Step 3: Helper Data = noisy XOR between bio and codeword
    P = (bio_bits ^ codeword).astype(np.uint8)

    # Step 4: Hash codeword → AES 256-bit key
    K = hashlib.sha256(codeword.tobytes()).digest()

    return K, P

def fuzzy_key_rep(bio_again_array, P):
    """
    Reconstruct AES key during decryption
    """
    bio_bits = (bio_again_array.flatten() % 2).astype(np.uint8)
    bio_bits = bio_bits[:bch.n] if bio_bits.size >= bch.n else \
               np.pad(bio_bits, (0, bch.n - bio_bits.size))

    # Recover corrected codeword
    noisy_codeword = (bio_bits ^ P).astype(np.uint8)
    codeword = bch.decode(noisy_codeword)

    K = hashlib.sha256(codeword.tobytes()).digest()
    return K

def extract_stable_features(image):
    """
    Use your CNN model to produce stable features
    Output: binary vector (length >= 255)
    """
    img = np.expand_dims(image, axis=0)
    features = model.predict(img)[0]  # shape ~ (some float vector)

    # Quantize + Binarize
    feat_norm = (features - features.min()) / (features.max() - features.min())
    bin_bits = (feat_norm > 0.5).astype(np.uint8)

    # Fit BCH length by padding/truncation
    bin_bits = bin_bits[:bch.n] if bin_bits.size >= bch.n else \
               np.pad(bin_bits, (0, bch.n - bin_bits.size))

    return bin_bits

image = Image.open("images.jpg").resize((256,256))
image_array = np.array(image)

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