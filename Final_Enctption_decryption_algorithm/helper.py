import os
import hashlib
import numpy as np
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import galois
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle
import json


# ==================== CONFIGURATION ====================
# BCH parameters
bch = galois.BCH(255, 131)
print(f"BCH parameters: n={bch.n}, k={bch.k}, t={bch.t} (correctable errors)")

# VGG16 model for biometric feature extraction
VGG_INPUT_SHAPE = (224, 224, 3)
vgg_model = VGG16(include_top=False, pooling="avg", input_shape=VGG_INPUT_SHAPE)
print("VGG16 model loaded")

AES_BLOCK_SIZE = 16  # bytes


# ==================== BIOMETRIC FUZZY EXTRACTOR ====================

def load_and_preprocess_image(path, target_size=(224, 224)):
    """Load image and preprocess for VGG16"""
    img = Image.open(path).convert("RGB").resize(target_size)
    arr = np.asarray(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def features_from_image_path(path):
    """Extract VGG16 features from biometric image"""
    preproc_batch = load_and_preprocess_image(path, target_size=(224, 224))
    feats = vgg_model.predict(preproc_batch, verbose=0)[0]
    return feats


def float_features_to_bits(feats, n_bits):
    """Convert float feature vector to binary vector"""
    feats = np.asarray(feats).astype(np.float32)
    if feats.max() == feats.min():
        norm = np.zeros_like(feats)
    else:
        norm = (feats - feats.min()) / (feats.max() - feats.min())
    bits = (norm > 0.5).astype(np.uint8).flatten()
    if bits.size >= n_bits:
        bits = bits[:n_bits]
    else:
        bits = np.pad(bits, (0, n_bits - bits.size), constant_values=0)
    return bits


def bits_to_bytes(bits):
    """Pack bit array into bytes"""
    return np.packbits(bits).tobytes()


def hamming_distance(bits1, bits2):
    """Calculate Hamming distance between two bit arrays"""
    return np.sum(np.bitwise_xor(bits1, bits2))


def enroll_biometric(biometric_image_path):
    """
    Enrollment: Generate key from biometric
    Returns: (key_bytes, helper_data_dict)
    """
    print(f"[Enrollment] Processing: {biometric_image_path}")
    
    # Extract and binarize features -  from vgg16 model pre-trained weights
    feats = features_from_image_path(biometric_image_path)
    bio_bits = float_features_to_bits(feats, n_bits=bch.n)
    
    # Generate random message and encode with BCH
    message_bits = np.random.randint(0, 2, size=bch.k, dtype=np.uint8)
    message_gf = galois.GF2(message_bits)
    codeword_gf = bch.encode(message_gf)
    codeword = np.array(codeword_gf, dtype=np.uint8)
    
    # Helper data: P = bio_bits XOR codeword
    P = np.bitwise_xor(bio_bits, codeword).astype(np.uint8)
    
    # Derive AES key from codeword
    codeword_bytes = bits_to_bytes(codeword)
    key = hashlib.sha256(codeword_bytes).digest()  # 32 bytes for AES-256
    
    helper_data = {
        "P": P.tolist(),
        "n": int(bch.n),
        "k": int(bch.k),
        "t": int(bch.t)
    }
    
    print(f"[Enrollment] Key generated (32 bytes), Helper data created")
    return key, helper_data


def reproduce_biometric_key(biometric_image_path, helper_data):
    """
    Reproduction: Regenerate key from probe biometric
    Returns: key_bytes (32 bytes)
    """
    print(f"[Reproduction] Processing: {biometric_image_path}")
    
    # Extract and binarize features
    feats = features_from_image_path(biometric_image_path)
    bio_bits_probe = float_features_to_bits(feats, n_bits=helper_data["n"])
    
    # Recover noisy codeword
    P = np.array(helper_data["P"], dtype=np.uint8)
    noisy_codeword = np.bitwise_xor(bio_bits_probe, P).astype(np.uint8)
    noisy_codeword_gf = galois.GF2(noisy_codeword)
    
    # BCH decode and re-encode to get corrected codeword
    try:
        decoded_message_gf = bch.decode(noisy_codeword_gf)  # Returns message (k bits)
        corrected_codeword_gf = bch.encode(decoded_message_gf)  # Re-encode to get codeword (n bits)
        corrected_codeword = np.array(corrected_codeword_gf, dtype=np.uint8)
        print(f"[Reproduction] BCH decoding successful")
    except Exception as e:
        raise ValueError(f"BCH decode failed (too many errors > {helper_data['t']}): {e}")
    
    # Derive key from corrected codeword
    corrected_bytes = bits_to_bytes(corrected_codeword)
    key = hashlib.sha256(corrected_bytes).digest()
    
    print(f"[Reproduction] Key reproduced successfully (32 bytes)")
    return key


# ==================== AES ENCRYPTION/DECRYPTION ====================

def aes_encrypt_numpy_array(np_array, key_bytes):
    """
    Encrypt numpy array with AES-256-CBC
    Args:
        np_array: numpy array (any dtype, any shape)
        key_bytes: 32-byte AES key
    Returns:
        (ciphertext_bytes, iv_bytes, metadata_dict)
    """
    # Store metadata for reconstruction
    metadata = {
        "shape": list(np_array.shape),
        "dtype": str(np_array.dtype)
    }
    
    # Convert to bytes
    plaintext = np_array.tobytes()
    
    # Encrypt with AES-CBC
    cipher = AES.new(key_bytes, AES.MODE_CBC)
    iv = cipher.iv
    ciphertext = cipher.encrypt(pad(plaintext, AES_BLOCK_SIZE))
    
    print(f"[AES Encrypt] Array shape {np_array.shape}, dtype {np_array.dtype}")
    print(f"[AES Encrypt] Plaintext: {len(plaintext)} bytes → Ciphertext: {len(ciphertext)} bytes")
    
    return ciphertext, iv, metadata


def aes_decrypt_to_numpy_array(ciphertext, iv, key_bytes, metadata):
    """
    Decrypt AES ciphertext back to numpy array
    Args:
        ciphertext: encrypted bytes
        iv: initialization vector
        key_bytes: 32-byte AES key
        metadata: dict with 'shape' and 'dtype'
    Returns:
        numpy array (original shape and dtype)
    """
    # Decrypt with AES-CBC
    cipher = AES.new(key_bytes, AES.MODE_CBC, iv=iv)
    plaintext_padded = cipher.decrypt(ciphertext)
    plaintext = unpad(plaintext_padded, AES_BLOCK_SIZE)
    
    # Reconstruct numpy array
    arr = np.frombuffer(plaintext, dtype=np.dtype(metadata["dtype"]))
    arr = arr.reshape(tuple(metadata["shape"]))
    
    print(f"[AES Decrypt] Ciphertext: {len(ciphertext)} bytes → Plaintext: {len(plaintext)} bytes")
    print(f"[AES Decrypt] Reconstructed array shape {arr.shape}, dtype {arr.dtype}")
    
    return arr


# ==================== SUBSTITUTION & PERTURBATION (PLACEHOLDERS) ====================

def substitution(image_array):
    """
    Apply substitution transformation to image
    Args:
        image_array: numpy array (H, W, C) uint8
    Returns:
        substituted_array: numpy array (H, W, C) uint8
    """
    print(f"[Substitution] Input shape: {image_array.shape}")
    # TODO: Implement your substitution logic here
    # Example: Simple pixel value transformation
    substituted = (image_array.astype(np.int16) + 50) % 256
    substituted = substituted.astype(np.uint8)
    print(f"[Substitution] Output shape: {substituted.shape}")
    return substituted


def inverse_substitution(substituted_array):
    """
    Apply inverse substitution transformation
    Args:
        substituted_array: numpy array (H, W, C) uint8
    Returns:
        original_array: numpy array (H, W, C) uint8
    """
    print(f"[Inverse Substitution] Input shape: {substituted_array.shape}")
    # TODO: Implement inverse of your substitution
    # Example: Inverse of simple transformation
    original = (substituted_array.astype(np.int16) - 50) % 256
    original = original.astype(np.uint8)
    print(f"[Inverse Substitution] Output shape: {original.shape}")
    return original


def perturbation(substituted_array, key_or_seed=None):
    """
    Apply perturbation (permutation/diffusion) to array
    Args:
        substituted_array: numpy array (H, W, C) uint8
        key_or_seed: optional seed for deterministic perturbation
    Returns:
        perturbed_array: numpy array (same shape) uint8
    """
    print(f"[Perturbation] Input shape: {substituted_array.shape}")
    # TODO: Implement your perturbation logic here
    # Example: Simple shuffle (deterministic with seed)
    if key_or_seed is not None:
        np.random.seed(key_or_seed)
    
    shape = substituted_array.shape
    flat = substituted_array.flatten()
    indices = np.arange(len(flat))
    np.random.shuffle(indices)
    
    perturbed = flat[indices].reshape(shape)
    print(f"[Perturbation] Output shape: {perturbed.shape}")
    return perturbed, indices  # Return indices for inverse


def inverse_perturbation(perturbed_array, indices):
    """
    Apply inverse perturbation to restore original order
    Args:
        perturbed_array: numpy array (H, W, C) uint8
        indices: permutation indices from forward perturbation
    Returns:
        original_array: numpy array (H, W, C) uint8
    """
    print(f"[Inverse Perturbation] Input shape: {perturbed_array.shape}")
    # TODO: Implement inverse of your perturbation
    # Example: Inverse shuffle
    shape = perturbed_array.shape
    flat = perturbed_array.flatten()
    
    # Create inverse permutation
    inv_indices = np.argsort(indices)
    restored = flat[inv_indices].reshape(shape)
    
    print(f"[Inverse Perturbation] Output shape: {restored.shape}")
    return restored


# ==================== FULL ENCRYPTION PIPELINE ====================

def encrypt_image_full_pipeline(image_path, biometric_path, perturbation_seed=42):
    """
    Complete encryption pipeline:
    1. Load image
    2. Substitution
    3. Perturbation
    4. AES encryption with biometric-derived key
    
    Args:
        image_path: path to image to encrypt
        biometric_path: path to biometric image for enrollment
        perturbation_seed: seed for deterministic perturbation
    
    Returns:
        encryption_bundle: dict containing all necessary data for decryption
    """
    print("\n" + "="*70)
    print("FULL ENCRYPTION PIPELINE")
    print("="*70)
    
    # Step 1: Load image
    print("\n[Step 1/4] Loading image...")
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image, dtype=np.uint8)
    print(f"Original image shape: {image_array.shape}")
    
    # Step 2: Substitution
    print("\n[Step 2/4] Applying substitution...")
    substituted_array = substitution(image_array)
    
    # Step 3: Perturbation
    print("\n[Step 3/4] Applying perturbation...")
    perturbed_array, perm_indices = perturbation(substituted_array, key_or_seed=perturbation_seed)
    
    # Step 4: Enroll biometric and encrypt with AES
    print("\n[Step 4/4] Enrolling biometric and encrypting with AES...")
    key, helper_data = enroll_biometric(biometric_path)
    ciphertext, iv, metadata = aes_encrypt_numpy_array(perturbed_array, key)
    
    # Bundle everything needed for decryption
    encryption_bundle = {
        "ciphertext": ciphertext,
        "iv": iv,
        "metadata": metadata,
        "helper_data": helper_data,
        "perm_indices": perm_indices.tolist(),  # For inverse perturbation
        "perturbation_seed": perturbation_seed,
        "original_shape": list(image_array.shape)
    }
    
    print("\n" + "="*70)
    print("✓ ENCRYPTION COMPLETE")
    print(f"  Ciphertext size: {len(ciphertext)} bytes")
    print(f"  Helper data size: {len(helper_data['P'])} bits")
    print("="*70)
    
    return encryption_bundle


def decrypt_image_full_pipeline(encryption_bundle, biometric_probe_path):
    """
    Complete decryption pipeline:
    1. Reproduce key from probe biometric
    2. AES decryption → get perturbed_array back
    3. Inverse perturbation
    4. Inverse substitution
    5. Reconstruct original image
    
    Args:
        encryption_bundle: dict from encryption pipeline
        biometric_probe_path: path to probe biometric image
    
    Returns:
        decrypted_image_array: numpy array (H, W, C) uint8
    """
    print("\n" + "="*70)
    print("FULL DECRYPTION PIPELINE")
    print("="*70)
    
    # Step 1: Reproduce key from biometric probe
    print("\n[Step 1/4] Reproducing key from biometric probe...")
    try:
        key = reproduce_biometric_key(biometric_probe_path, encryption_bundle["helper_data"])
    except ValueError as e:
        print(f"\n✗ BIOMETRIC AUTHENTICATION FAILED")
        print(f"   {e}")
        raise
    
    # Step 2: AES decryption
    print("\n[Step 2/4] Decrypting with AES...")
    try:
        perturbed_array = aes_decrypt_to_numpy_array(
            encryption_bundle["ciphertext"],
            encryption_bundle["iv"],
            key,
            encryption_bundle["metadata"]
        )
    except Exception as e:
        print(f"\n✗ AES DECRYPTION FAILED")
        print(f"   {e}")
        raise
    
    # Step 3: Inverse perturbation
    print("\n[Step 3/4] Applying inverse perturbation...")
    perm_indices = np.array(encryption_bundle["perm_indices"])
    substituted_array = inverse_perturbation(perturbed_array, perm_indices)
    
    # Step 4: Inverse substitution
    print("\n[Step 4/4] Applying inverse substitution...")
    decrypted_image_array = inverse_substitution(substituted_array)
    
    print("\n" + "="*70)
    print("✓ DECRYPTION COMPLETE")
    print(f"  Decrypted image shape: {decrypted_image_array.shape}")
    print("="*70)
    
    return decrypted_image_array


# ==================== STORAGE FUNCTIONS ====================

def save_encryption_bundle(bundle, filepath="encryption_bundle.pkl"):
    """Save encryption bundle to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"\n✓ Encryption bundle saved to: {filepath}")


def load_encryption_bundle(filepath="encryption_bundle.pkl"):
    """Load encryption bundle from file"""
    with open(filepath, 'rb') as f:
        bundle = pickle.load(f)
    print(f"\n✓ Encryption bundle loaded from: {filepath}")
    return bundle


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Configuration
    image_to_encrypt = "Final_Enctption_decryption_algorithm/image.jpg"
    biometric_enroll = "Final_Enctption_decryption_algorithm/biometric images/kelvinl3.jpg"
    biometric_probe = "Final_Enctption_decryption_algorithm/biometric images/kelvinl5.jpg"
    
    print("="*70)
    print("BIOMETRIC IMAGE ENCRYPTION SYSTEM")
    print("="*70)
    print(f"Image to encrypt: {image_to_encrypt}")
    print(f"Biometric (enroll): {biometric_enroll}")
    print(f"Biometric (probe): {biometric_probe}")
    print("="*70)
    
    # ========== ENCRYPTION ==========
    encryption_bundle = encrypt_image_full_pipeline(
        image_path=image_to_encrypt,
        biometric_path=biometric_enroll,
        perturbation_seed=42
    )
    
    # Save to disk
    save_encryption_bundle(encryption_bundle, "encryption_bundle.pkl")
    
    # ========== DECRYPTION ==========
    try:
        decrypted_image_array = decrypt_image_full_pipeline(
            encryption_bundle=encryption_bundle,
            biometric_probe_path=biometric_probe
        )
        
        # Save decrypted image
        decrypted_image = Image.fromarray(decrypted_image_array)
        decrypted_image.save("decrypted_image.png")
        print(f"\n✓ Decrypted image saved to: decrypted_image.png")
        
        # Verify perfect recovery
        original_image = Image.open(image_to_encrypt).convert("RGB")
        original_array = np.array(original_image, dtype=np.uint8)
        
        if np.array_equal(original_array, decrypted_image_array):
            print("\n" + "="*70)
            print("✓✓✓ SUCCESS: PERFECT IMAGE RECOVERY ✓✓✓")
            print("="*70)
        else:
            diff = np.sum(np.abs(original_array.astype(int) - decrypted_image_array.astype(int)))
            print(f"\n⚠ Warning: Image differs from original (total pixel diff: {diff})")
    
    except Exception as e:
        print("\n" + "="*70)
        print("✗✗✗ DECRYPTION FAILED ✗✗✗")
        print(f"Error: {e}")
        print("="*70)




def ciphertext_to_image(cipher_bytes, shape):
    """Map ciphertext bytes to a 2D uint8 array with given shape (H,W)."""
    H, W = shape
    total = H * W
    arr = np.frombuffer(cipher_bytes, dtype=np.uint8)

    # If AES added padding/overhead, trim to first H*W bytes
    if arr.size >= total:
        arr = arr[:total]
    else:
        # Rare: if shorter (shouldn’t happen with your pipeline), pad with random-like tail
        pad = np.random.randint(0, 256, size=(total - arr.size,), dtype=np.uint8)
        arr = np.concatenate([arr, pad])

    return arr.reshape(H, W)