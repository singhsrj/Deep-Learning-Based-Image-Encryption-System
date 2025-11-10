# %%
from file import test_complete_system
import numpy as np
from substitute_perbutate import FresnelSubstitution , PixelPerturbation
from helper import enroll_biometric, aes_encrypt_numpy_array,aes_decrypt_to_numpy_array,reproduce_biometric_key,ciphertext_to_image
from PIL import Image
import matplotlib.pyplot as plt

image_to_encrypt = "image.jpg"
biometric_enroll = "biometric images/kelvinl3.jpg"
biometric_probe = "biometric images/kelvinl5.jpg"

# %%
def deep_learning_based_encryption_decryption(image_path, biometric_enroll, biometric_probe):
    """
    Complete encryption-decryption
    
    Args:
        image_path: Path to the input image
        biometric_enroll: Path to the biometric enrollment image
        biometric_probe: Path to the biometric probe image
    """
    # parameters previously used
    seed_d=12345
    seed_f=67890
    r=3.99
    x=0.5

    [image_array , substitued_array , perturbed_array] = substitution_perturbation(image_path=image_path)
    #AES Encryption
    key, helper_data = enroll_biometric(biometric_enroll)
    ciphertext, iv, metadata = aes_encrypt_numpy_array(perturbed_array, key)
    # Display ciphertext as image

    H, W = image_array.shape
    cipher_img = ciphertext_to_image(ciphertext, (H, W))
    #Reproduce key from biometric probe
    try:
        key = reproduce_biometric_key(biometric_probe, helper_data)
    except ValueError as e:
        print(f"\n✗ BIOMETRIC AUTHENTICATION FAILED")
        print(f"   {e}")
        raise

    #AES Decryption
    try:
        recovered_perturbed_array = aes_decrypt_to_numpy_array(
            ciphertext,
            iv,
            key,
            metadata
    )
    except Exception as e:
        print(f"\n✗ AES DECRYPTION FAILED")
        print(f"   {e}")
        raise
    
    #Inverse Perturbation
    pp2 = PixelPerturbation(r_init=r, x_init=x)
    pp2.x_original = x
    inv_subs_array = pp2.perturbate_image_inverse(recovered_perturbed_array.copy())
    print("✓ Inverse Perturbation complete!")

    # INVERSE Substitution
    fs = FresnelSubstitution(seed_d=seed_d, seed_f=seed_f)
    recovered_array = np.zeros_like(inv_subs_array)
    height, width = inv_subs_array.shape
    for i in range(height):
        if i % 50 == 0:
            print(f"  Processing row {i}/{height}")
        row = inv_subs_array[i, :]
        recovered_row = fs.substitute_inv(row)
        recovered_array[i, :] = recovered_row

    return [image_array , substitued_array , perturbed_array ,cipher_img, recovered_perturbed_array , inv_subs_array, recovered_array]



def substitution_perturbation(image_path, seed_d=12345, seed_f=67890, r=3.99, x=0.5):
    """
    Combined encryption: Substitution -> Perturbation -> Inverse Perturbation -> Inverse Substitution
    
    
    Args:
        image_path: Path to the input image
        seed_d: Seed for distance parameter (substitution)
        seed_f: Seed for frequency parameter (substitution)
        r: Logistic map parameter (perturbation)
        x: Initial value for logistic map (perturbation)
        
    Returns:
        original_img, after_substitution, after_perturbation, after_inv_perturbation, final_recovered
    """
    # Load image
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    print(f"Original image shape: {img_array.shape}")
    height, width = img_array.shape
    
    # ==================== STEP 1: SUBSTITUTION ====================
    print("\n" + "="*60)
    print("STEP 1: SUBSTITUTION")
    print("="*60)
    
    fs = FresnelSubstitution(seed_d=seed_d, seed_f=seed_f)
    substituted_array = np.zeros_like(img_array)
    
    print("Performing substitution...")
    for i in range(height):
        if i % 50 == 0:
            print(f"  Processing row {i}/{height}")
        row = img_array[i, :]
        substituted_row = fs.substitute(row)
        substituted_array[i, :] = substituted_row
    
    print("✓ Substitution complete!")
    
    # ==================== STEP 2: PERTURBATION ====================
    print("\n" + "="*60)
    print("STEP 2: PERTURBATION")
    print("="*60)
    
    pp = PixelPerturbation(r_init=r, x_init=x)
    pp.x_original = x
    
    print("Performing perturbation on substituted image...")
    perturbed_array = pp.perturbate_image(substituted_array.copy())
    print("✓ Perturbation complete!")
    
    return img_array, substituted_array, perturbed_array