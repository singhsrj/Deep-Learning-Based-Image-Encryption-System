import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64

# Pad data to be multiple of 16 bytes for AES block size
def pad(arr_bytes):
    padding_len = 16 - len(arr_bytes) % 16
    return arr_bytes + bytes([padding_len]) * padding_len

def unpad(arr_bytes):
    padding_len = arr_bytes[-1]
    return arr_bytes[:-padding_len]

# Encrypt NumPy array
def encrypt_array(data, key):
    arr_bytes = data.tobytes()
    arr_bytes = pad(arr_bytes)

    iv = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    encrypted = cipher.encrypt(arr_bytes)
    return encrypted, iv, data.shape, data.dtype

# Decrypt back to NumPy array
def decrypt_array(encrypted, iv, key, shape, dtype):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = cipher.decrypt(encrypted)
    decrypted = unpad(decrypted)
    return np.frombuffer(decrypted, dtype=dtype).reshape(shape)


# -------------------- TEST --------------------

# Generate random array
array = np.random.randint(0, 256, (4, 4), dtype=np.uint8)
print("Original Array:\n", array)

# 32-byte key = AES-256
key = get_random_bytes(32)

encrypted, iv, shape, dtype = encrypt_array(array, key)
print("\nEncrypted (base64):", base64.b64encode(encrypted).decode())

# Decrypting
decrypted_array = decrypt_array(encrypted, iv, key, shape, dtype)
print("\nDecrypted Array:\n", decrypted_array)

print("\nMatch:", np.array_equal(array, decrypted_array))
