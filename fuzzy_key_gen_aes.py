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
import json
import pickle


# ----- BCH (galois) params -----
# Using BCH(255, 131) - can correct up to t=18 errors
bch = galois.BCH(255, 131)
print(f"BCH parameters: n={bch.n}, k={bch.k}, t={bch.t} (correctable errors)")


# ----- VGG16 feature extractor -----
VGG_INPUT_SHAPE = (224, 224, 3)
vgg_model = VGG16(include_top=False, pooling="avg", input_shape=VGG_INPUT_SHAPE)
print("VGG16 model loaded")


AES_BLOCK_SIZE = 16  # bytes


# ---------- Helper utilities ----------
def load_and_preprocess_image(path, target_size=(224, 224)):
    """Load image and preprocess for VGG16"""
    img = Image.open(path).convert("RGB").resize(target_size)
    arr = np.asarray(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)  # VGG preprocessing
    return arr


def features_from_image_path(path):
    """Extract VGG16 features from image"""
    preproc_batch = load_and_preprocess_image(path, target_size=(224, 224))
    feats = vgg_model.predict(preproc_batch, verbose=0)[0]  # 1D float vector (512 dims)
    return feats


def float_features_to_bits(feats, n_bits):
    """
    Convert float feature vector to binary vector of length n_bits
    Strategy: min-max normalize -> threshold at 0.5 -> pad/truncate to n_bits
    """
    feats = np.asarray(feats).astype(np.float32)
    
    # Normalize to [0, 1]
    if feats.max() == feats.min():
        norm = np.zeros_like(feats)
    else:
        norm = (feats - feats.min()) / (feats.max() - feats.min())
    
    # Threshold at 0.5 to get binary
    bits = (norm > 0.5).astype(np.uint8).flatten()
    
    # Pad or truncate to exactly n_bits
    if bits.size >= n_bits:
        bits = bits[:n_bits]
    else:
        bits = np.pad(bits, (0, n_bits - bits.size), constant_values=0)
    
    return bits


def bits_to_bytes(bits):
    """Pack bit array into bytes"""
    return np.packbits(bits).tobytes()


def bytes_to_bits(b, n_bits=None):
    """Unpack bytes to bit array"""
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    if n_bits is not None:
        return bits[:n_bits]
    return bits


def hamming_distance(bits1, bits2):
    """Calculate Hamming distance between two bit arrays"""
    return np.sum(np.bitwise_xor(bits1, bits2))


# ---------- Fuzzy extractor (Gen / Rep) ----------
def enroll(biometric_image_path):
    """
    Enrollment (Gen):
    1. Extract VGG16 features from biometric image
    2. Binarize to n bits
    3. Generate random message bits, encode with BCH to get codeword
    4. Helper data P = bio_bits XOR codeword
    5. Key = SHA-256(codeword)
    
    IMPORTANT: We use a RANDOM message, not derived from biometrics.
    The biometrics are only used to create helper data P for reconstruction.
    
    Returns: (key_bytes, helper_data_dict, debug_info)
    """
    print(f"Enrolling with: {biometric_image_path}")
    
    # Extract features and binarize
    feats = features_from_image_path(biometric_image_path)
    bio_bits = float_features_to_bits(feats, n_bits=bch.n)  # length n=255
    
    # CRITICAL FIX: Generate RANDOM message bits (not from biometric)
    # This ensures the key is uniformly random and reproducible via error correction
    message_bits = np.random.randint(0, 2, size=bch.k, dtype=np.uint8)
    message_gf = galois.GF2(message_bits)
    
    # BCH encode: message (k bits) -> codeword (n bits)
    codeword_gf = bch.encode(message_gf)
    codeword = np.array(codeword_gf, dtype=np.uint8)  # length n=255
    
    print(f"Message length: {len(message_gf)}, Codeword length: {len(codeword)}")
    
    # Helper data: P = bio_bits XOR codeword
    # This "binds" the random codeword to the biometric
    P = np.bitwise_xor(bio_bits, codeword).astype(np.uint8)
    
    # Derive AES key from codeword
    codeword_bytes = bits_to_bytes(codeword)
    key = hashlib.sha256(codeword_bytes).digest()  # 32 bytes
    
    helper_data = {
        "P": P.tolist(),  # store as list for JSON compatibility
        "n": int(bch.n),
        "k": int(bch.k),
        "t": int(bch.t)
    }
    
    debug_info = {
        "enrollment_bio_bits": bio_bits.copy(),
        "enrollment_codeword": codeword.copy(),
        "enrollment_message": message_bits.copy()
    }
    
    print(f"Enrollment complete. Key derived (32 bytes)")
    return key, helper_data, debug_info


def reproduce_key(biometric_image_path, helper_data, debug_info=None):
    """
    Reproduction (Rep):
    1. Extract features from probe biometric
    2. Binarize to n bits
    3. Compute noisy_codeword = bio_bits_probe XOR P
    4. BCH decode to correct errors -> corrected_codeword
    5. Key = SHA-256(corrected_codeword)
    
    The BCH decoder will correct errors in noisy_codeword to recover the original codeword
    used during enrollment (as long as errors <= t).
    
    Returns: key_bytes (32 bytes), reproduction_debug_info
    """
    print(f"Reproducing key with: {biometric_image_path}")
    
    # Extract features and binarize
    feats = features_from_image_path(biometric_image_path)
    bio_bits_probe = float_features_to_bits(feats, n_bits=helper_data["n"])
    
    # Recover helper data
    P = np.array(helper_data["P"], dtype=np.uint8)
    
    # Compute noisy codeword: this reverses the XOR from enrollment
    # noisy_codeword ≈ original_codeword (with some bit errors from biometric noise)
    noisy_codeword = np.bitwise_xor(bio_bits_probe, P).astype(np.uint8)
    noisy_codeword_gf = galois.GF2(noisy_codeword)
    
    # Calculate error statistics if debug info available
    repro_debug = {}
    if debug_info is not None:
        enrollment_bits = debug_info["enrollment_bio_bits"]
        bio_hamming = hamming_distance(enrollment_bits, bio_bits_probe)
        
        enrollment_codeword = debug_info["enrollment_codeword"]
        codeword_hamming = hamming_distance(enrollment_codeword, noisy_codeword)
        
        repro_debug["bio_hamming_distance"] = int(bio_hamming)
        repro_debug["noisy_codeword_hamming"] = int(codeword_hamming)
        repro_debug["enrollment_codeword_sample"] = enrollment_codeword[:10].tolist()
        repro_debug["noisy_codeword_sample"] = noisy_codeword[:10].tolist()
        
        print(f"Biometric Hamming distance: {bio_hamming}/{helper_data['n']} bits")
        print(f"Noisy codeword errors: {codeword_hamming}/{helper_data['n']} bits (max correctable: {helper_data['t']})")
    
    # BCH decode to correct errors
    # BCH.decode returns the MESSAGE (k bits), not the codeword (n bits)
    # We need to re-encode to get the corrected codeword
    try:
        decoded_message_gf = bch.decode(noisy_codeword_gf)  # Returns message (k=131 bits)
        decoded_message = np.array(decoded_message_gf, dtype=np.uint8)
        
        # Re-encode the message to get the corrected codeword (n=255 bits)
        corrected_codeword_gf = bch.encode(decoded_message_gf)
        corrected_codeword = np.array(corrected_codeword_gf, dtype=np.uint8)
        
        # Verify correction if debug info available
        if debug_info is not None:
            enrollment_codeword = debug_info["enrollment_codeword"]
            enrollment_message = debug_info["enrollment_message"]
            
            message_match = np.array_equal(decoded_message, enrollment_message)
            codeword_match = np.array_equal(corrected_codeword, enrollment_codeword)
            
            if message_match and codeword_match:
                print(f"✓ BCH decoding successful - message and codeword perfectly recovered!")
                repro_debug["codeword_recovered"] = True
                repro_debug["message_recovered"] = True
            else:
                msg_errors = hamming_distance(decoded_message, enrollment_message)
                cw_errors = hamming_distance(corrected_codeword, enrollment_codeword)
                print(f"⚠ BCH decoded but differences remain:")
                print(f"   Message errors: {msg_errors}/{bch.k} bits")
                print(f"   Codeword errors: {cw_errors}/{bch.n} bits")
                repro_debug["codeword_recovered"] = codeword_match
                repro_debug["message_recovered"] = message_match
                repro_debug["message_errors"] = int(msg_errors)
                repro_debug["codeword_errors"] = int(cw_errors)
        else:
            print(f"BCH decoding successful")
        
        repro_debug["decode_success"] = True
        
    except Exception as e:
        repro_debug["decode_success"] = False
        repro_debug["error"] = str(e)
        raise ValueError(f"BCH decode failed - too many bit errors (>{helper_data['t']}): {e}")
    
    # Derive AES key from corrected codeword (must use codeword, not message!)
    corrected_bytes = bits_to_bytes(corrected_codeword)
    key = hashlib.sha256(corrected_bytes).digest()
    
    print(f"Key reproduced successfully (32 bytes)")
    return key, repro_debug


# ---------- AES encrypt/decrypt helpers (AES-CBC) ----------
def aes_cbc_encrypt_numpy_array(np_array_uint8, key_bytes):
    """
    Encrypt numpy array (uint8) with AES-CBC
    Returns: (ciphertext, iv, metadata)
    """
    plaintext = np_array_uint8.tobytes()
    cipher = AES.new(key_bytes, AES.MODE_CBC)
    iv = cipher.iv
    ct = cipher.encrypt(pad(plaintext, AES_BLOCK_SIZE))
    
    meta = {
        "shape": list(np_array_uint8.shape),
        "dtype": str(np_array_uint8.dtype)
    }
    return ct, iv, meta


def aes_cbc_decrypt_to_numpy(ciphertext, iv, key_bytes, meta):
    """
    Decrypt AES-CBC ciphertext to numpy array
    """
    cipher = AES.new(key_bytes, AES.MODE_CBC, iv=iv)
    pt_padded = cipher.decrypt(ciphertext)
    pt = unpad(pt_padded, AES_BLOCK_SIZE)
    
    arr = np.frombuffer(pt, dtype=np.dtype(meta["dtype"]))
    arr = arr.reshape(tuple(meta["shape"]))
    return arr


# ---------- High-level pipeline functions ----------
def encrypt_image_with_biometric(image_to_encrypt_path, biometric_image_path):
    """
    Full encryption pipeline:
    1. Load image to encrypt
    2. Enroll biometric -> derive AES key + helper data
    3. Encrypt image with AES-CBC
    
    Returns: cipher_bundle dict
    """
    print(f"\n=== ENCRYPTION ===")
    print(f"Image to encrypt: {image_to_encrypt_path}")
    
    # Load image to encrypt
    img = Image.open(image_to_encrypt_path).convert("RGB")
    arr = np.asarray(img).astype(np.uint8)
    print(f"Image shape: {arr.shape}")
    
    # Enroll biometric and derive key
    key, helper_data, debug_info = enroll(biometric_image_path)
    
    # Encrypt with AES-CBC
    ct, iv, meta = aes_cbc_encrypt_numpy_array(arr, key)
    print(f"Encryption complete. Ciphertext size: {len(ct)} bytes")
    
    return {
        "ciphertext": ct,
        "iv": iv,
        "helper_data": helper_data,
        "meta": meta,
        "debug_info": debug_info
    }


def decrypt_image_with_biometric(cipher_bundle, biometric_probe_path):
    """
    Full decryption pipeline:
    1. Reproduce key from probe biometric + helper data
    2. Decrypt with AES-CBC
    
    Returns: recovered numpy array (uint8), reproduction_debug_info
    """
    print(f"\n=== DECRYPTION ===")
    
    helper_data = cipher_bundle["helper_data"]
    ct = cipher_bundle["ciphertext"]
    iv = cipher_bundle["iv"]
    meta = cipher_bundle["meta"]
    debug_info = cipher_bundle.get("debug_info", None)
    
    # Reproduce key from biometric probe
    key_reproduced, repro_debug = reproduce_key(biometric_probe_path, helper_data, debug_info)
    
    # Decrypt
    arr = aes_cbc_decrypt_to_numpy(ct, iv, key_reproduced, meta)
    print(f"Decryption complete. Recovered shape: {arr.shape}")
    
    return arr, repro_debug


# ---------- Storage functions ----------
def save_cipher_bundle(bundle, filepath="cipher_bundle.pkl"):
    """Save cipher bundle to file (pickle for binary data)"""
    with open(filepath, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"Cipher bundle saved to: {filepath}")


def load_cipher_bundle(filepath="cipher_bundle.pkl"):
    """Load cipher bundle from file"""
    with open(filepath, 'rb') as f:
        bundle = pickle.load(f)
    print(f"Cipher bundle loaded from: {filepath}")
    return bundle


def save_helper_data_json(helper_data, filepath="helper_data.json"):
    """Save just the helper data as JSON (public data)"""
    with open(filepath, 'w') as f:
        json.dump(helper_data, f, indent=2)
    print(f"Helper data saved to: {filepath}")


def load_helper_data_json(filepath="helper_data.json"):
    """Load helper data from JSON"""
    with open(filepath, 'r') as f:
        helper_data = json.load(f)
    print(f"Helper data loaded from: {filepath}")
    return helper_data


# ---------- Testing and analysis functions ----------
def test_biometric_pair(bio_path1, bio_path2, test_name="Test"):
    """
    Test a pair of biometric images to see if they can reproduce the same key
    """
    print(f"\n{'='*60}")
    print(f"{test_name}")
    print(f"{'='*60}")
    
    # Enroll with first image
    key1, helper_data, debug_info = enroll(bio_path1)
    
    # Try to reproduce with second image
    try:
        key2, repro_debug = reproduce_key(bio_path2, helper_data, debug_info)
        
        # Check if keys match
        keys_match = (key1 == key2)
        
        print(f"\n{'='*60}")
        print(f"RESULT: {'✓ SUCCESS' if keys_match else '✗ FAILURE'}")
        print(f"Keys match: {keys_match}")
        print(f"Bio Hamming distance: {repro_debug.get('bio_hamming_distance', 'N/A')}/{bch.n}")
        print(f"Noisy codeword errors: {repro_debug.get('noisy_codeword_hamming', 'N/A')}/{bch.n}")
        print(f"Max correctable errors: {bch.t}")
        print(f"{'='*60}\n")
        
        return keys_match, repro_debug
        
    except ValueError as e:
        print(f"\n{'='*60}")
        print(f"RESULT: ✗ FAILURE")
        print(f"Error: {e}")
        print(f"{'='*60}\n")
        return False, {"error": str(e)}


def analyze_feature_similarity(bio_path1, bio_path2):
    """
    Analyze similarity between two biometric images at feature level
    """
    print(f"\n=== Feature Similarity Analysis ===")
    print(f"Image 1: {bio_path1}")
    print(f"Image 2: {bio_path2}")
    
    # Extract features
    feats1 = features_from_image_path(bio_path1)
    feats2 = features_from_image_path(bio_path2)
    
    # Feature-level statistics
    cosine_sim = np.dot(feats1, feats2) / (np.linalg.norm(feats1) * np.linalg.norm(feats2))
    l2_distance = np.linalg.norm(feats1 - feats2)
    
    # Bit-level statistics
    bits1 = float_features_to_bits(feats1, n_bits=bch.n)
    bits2 = float_features_to_bits(feats2, n_bits=bch.n)
    hamming_dist = hamming_distance(bits1, bits2)
    hamming_ratio = hamming_dist / bch.n
    
    print(f"Feature dimension: {len(feats1)}")
    print(f"Cosine similarity: {cosine_sim:.4f}")
    print(f"L2 distance: {l2_distance:.4f}")
    print(f"Hamming distance: {hamming_dist}/{bch.n} ({hamming_ratio*100:.2f}%)")
    print(f"BCH can correct: up to {bch.t} errors ({bch.t/bch.n*100:.2f}%)")
    print(f"Status: {'Within correction capability' if hamming_dist <= bch.t else 'Exceeds correction capability'}")
    
    return {
        "cosine_similarity": float(cosine_sim),
        "l2_distance": float(l2_distance),
        "hamming_distance": int(hamming_dist),
        "hamming_ratio": float(hamming_ratio),
        "correctable": hamming_dist <= bch.t
    }


# ---------- Main execution ----------
if __name__ == "__main__":
    # File paths
    biometric_enroll = "Final_Enctption_decryption_algorithm/biometric images/kelvinl3.jpg"   # enrollment biometric
    biometric_probe = "Final_Enctption_decryption_algorithm/biometric images/kelvinl5.jpg"     # probe biometric (same person)
    image_to_encrypt = "Final_Enctption_decryption_algorithm/image.jpg"                        # image to encrypt
    
    print("=" * 60)
    print("Biometric Fuzzy Extractor with BCH + AES")
    print("=" * 60)
    
    # ========== PART 1: Feature Analysis ==========
    print("\n" + "="*60)
    print("PART 1: Analyzing Biometric Similarity")
    print("="*60)
    
    if os.path.exists(biometric_enroll) and os.path.exists(biometric_probe):
        similarity_stats = analyze_feature_similarity(biometric_enroll, biometric_probe)
    else:
        print(f"Warning: Biometric files not found. Skipping similarity analysis.")
    
    # ========== PART 2: Key Reproduction Test ==========
    print("\n" + "="*60)
    print("PART 2: Testing Key Reproduction")
    print("="*60)
    
    if os.path.exists(biometric_enroll) and os.path.exists(biometric_probe):
        success, debug = test_biometric_pair(biometric_enroll, biometric_probe, 
                                             "Same Person Test (kelvinl3 vs kelvinl5)")
    else:
        print(f"Warning: Biometric files not found. Skipping key reproduction test.")
    
    # ========== PART 3: Full Encrypt/Decrypt Pipeline ==========
    print("\n" + "="*60)
    print("PART 3: Full Encryption/Decryption Pipeline")
    print("="*60)
    
    if not os.path.exists(image_to_encrypt):
        print(f"Warning: Image to encrypt '{image_to_encrypt}' not found.")
        print("Skipping encryption/decryption demo.")
    elif not os.path.exists(biometric_enroll) or not os.path.exists(biometric_probe):
        print(f"Warning: Biometric images not found.")
        print("Skipping encryption/decryption demo.")
    else:
        # STEP 1: ENROLL + ENCRYPT
        bundle = encrypt_image_with_biometric(image_to_encrypt, biometric_enroll)
        
        # Save bundle to disk
        save_cipher_bundle(bundle, "cipher_bundle.pkl")
        save_helper_data_json(bundle["helper_data"], "helper_data.json")
        
        # STEP 2: DECRYPT using probe biometric
        try:
            recovered, repro_debug = decrypt_image_with_biometric(bundle, biometric_probe)
            
            # Save recovered image
            recovered_img = Image.fromarray(recovered)
            recovered_img.save("recovered.png")
            
            # Verify recovery
            original_img = Image.open(image_to_encrypt).convert("RGB")
            original_arr = np.asarray(original_img)
            
            if np.array_equal(original_arr, recovered):
                print(f"\n✓ PERFECT RECOVERY: Image matches original exactly!")
            else:
                diff = np.sum(np.abs(original_arr.astype(int) - recovered.astype(int)))
                print(f"\n⚠ IMPERFECT RECOVERY: Total pixel difference = {diff}")
            
            print(f"✓ Recovered image saved to 'recovered.png'")
            
        except ValueError as e:
            print(f"\n✗ DECRYPTION FAILED: {e}")
            print("The probe biometric differs too much from enrollment.")
            print(f"BCH can correct up to {bch.t} bit errors out of {bch.n} bits.")
    
    # ========== PART 4: Summary ==========
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"BCH Code: BCH({bch.n}, {bch.k}) with t={bch.t} error correction")
    print(f"Feature Extractor: VGG16 (512-dim features)")
    print(f"Encryption: AES-256-CBC")
    print(f"Key Derivation: SHA-256(BCH_codeword)")
    print("="*60)
    
    