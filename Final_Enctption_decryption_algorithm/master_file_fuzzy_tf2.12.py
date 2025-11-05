# %%
from file import test_complete_system
import numpy as np
from substitute_perbutate import FresnelSubstitution , PixelPerturbation
from PIL import Image
import matplotlib.pyplot as plt

# %%
image_to_encrypt = "image.jpg"
biometric_enroll = "biometric images/kelvinl3.jpg"
biometric_probe = "biometric images/kelvinl5.jpg"

# %%

def combined_encryption(image_path, seed_d=12345, seed_f=67890, r=3.99, x=0.5):
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


# %%
[image_array , substitued_array , perturbed_array] = combined_encryption(image_path='image.jpg')

# %%
import matplotlib.pyplot as plt

if image_array is not None:
    plt.imshow(image_array, cmap='gray')
    plt.title("Image")
    plt.axis('off')
    plt.show()

# %%
import matplotlib.pyplot as plt

if perturbed_array is not None:
    plt.imshow(perturbed_array, cmap='gray')
    plt.title("Perturbed Image")
    plt.axis('off')
    plt.show()

# %%
type(perturbed_array)

# %% [markdown]
# #### Enroll biometric and encrypt with AES
# 

# %%
from helper import enroll_biometric, aes_encrypt_numpy_array

# %%
key, helper_data = enroll_biometric(biometric_enroll)
ciphertext, iv, metadata = aes_encrypt_numpy_array(perturbed_array, key)

# %% [markdown]
# #### Reproduce key from biometric probe

# %%
from helper import reproduce_biometric_key

# %%
try:
    key = reproduce_biometric_key(biometric_probe, helper_data)
except ValueError as e:
    print(f"\n✗ BIOMETRIC AUTHENTICATION FAILED")
    print(f"   {e}")
    raise

# %% [markdown]
# #### AES decryption

# %%
from helper import aes_decrypt_to_numpy_array

# %%
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

# %% [markdown]
# ### successfully recovered perturbed_array

# %%
np.abs(recovered_perturbed_array-perturbed_array)

# %% [markdown]
# #### INVERSE PERTURBATION

# %%
# parameters previously used
seed_d=12345
seed_f=67890
r=3.99
x=0.5

# %%
    
pp2 = PixelPerturbation(r_init=r, x_init=x)
pp2.x_original = x
    

# %%

inv_subs_array = pp2.perturbate_image_inverse(recovered_perturbed_array.copy())


# %%
import matplotlib.pyplot as plt

if inv_subs_array is not None:
    plt.imshow(inv_subs_array, cmap='gray')
    plt.title("Loaded & Prepared Image")
    plt.axis('off')
    plt.show()


# %% [markdown]
# #### INVERSE Substitution

# %%
fs = FresnelSubstitution(seed_d=seed_d, seed_f=seed_f)
recovered_array = np.zeros_like(inv_subs_array)


# %%
height, width = inv_subs_array.shape


# %%
for i in range(height):
        if i % 50 == 0:
            print(f"  Processing row {i}/{height}")
        row = inv_subs_array[i, :]
        recovered_row = fs.substitute_inv(row)
        recovered_array[i, :] = recovered_row
    

# %%
import matplotlib.pyplot as plt

if recovered_array is not None:
    plt.imshow(recovered_array, cmap='gray')
    plt.title("Recovered Image")
    plt.axis('off')
    plt.show()



