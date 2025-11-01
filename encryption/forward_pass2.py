"""
Medical Image Encryption - Forward Pass Module
Implements: Fresnel Substitution + Markov Chain + Pixel Perturbation

FINAL WORKING VERSION - All bugs fixed
"""

import numpy as np
import math

# ============================================================================
# CORE STATE UPDATE FUNCTION (FULLY FIXED)
# ============================================================================

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)

def update_df(R: float, c: int, d: int):
    """
    Update-df based on Algorithm 2 from Fresnel Crypto Paper.
    R: float (Fresnel radius)
    c: uint8 pixel (0–255)
    d: uint64 evolving state
    """

    # Step-1: f = Long(R)
    f = np.uint64(int(R)) & MASK

    # Ensure bitwise operands are unsigned 64-bit
    c_u = np.uint64(c) & MASK
    d_u = np.uint64(d) & MASK

    # Extract z from right-most 5 bits of pixel c
    z = int(c_u & np.uint64(31))  # modulo 32

    if z != 0:
        # Shift amount from old d
        shift = int(d_u % z)
        d_u = (c_u << shift) ^ d_u
    else:
        d_u ^= c_u  # fallback rule in paper when mod=0

    d_u &= MASK

    # Steps 3–5: Xorshift to spread influence of bits
    d_u ^= (d_u << 21) & MASK
    d_u ^= (d_u >> 35)
    d_u ^= (d_u << 4) & MASK

    return int(f), int(d_u)


# ============================================================================
# SINGLE-PASS FRESNEL SUBSTITUTION
# ============================================================================

def Substitute(B):
    """
    Single-pass Fresnel substitution with robust state management.
    
    Args:
        B: list/array of pixel values [b1, b2, ..., bn]
    
    Returns:
        list of substituted pixels
    """
    n = len(B)
    out = []
    
    f = np.int64(1)
    d = np.int64(1)
    
    def safe_R(d, f):
        """Calculate R with overflow protection."""
        if not math.isfinite(float(d)): 
            d = np.int64(1)
        if not math.isfinite(float(f)): 
            f = np.int64(1)
        d_abs = min(abs(int(d)), 2**31 - 1)
        f_abs = min(abs(int(f)), 2**31 - 1)
        val = d_abs / (4.0 * f_abs + 1e-9)
        return 17.32 * math.sqrt(val if val >= 0 else 0.0)
    
    for bi in B:
        R = safe_R(d, f)
        k = int(R) % 256
        si = k ^ int(bi)
        out.append(si)
        f, d = update_df(R, bi, d)
    
    return [int(x) & 0xFF for x in out]


def Substitute_Inv(out):
    """
    Inverse of single-pass Fresnel substitution.
    
    Args:
        out: encrypted block
    
    Returns:
        original block B
    """
    out = [int(x) & 0xFF for x in out]
    B = []
    
    f = np.int64(1)
    d = np.int64(1)
    
    def safe_R(d, f):
        if not math.isfinite(float(d)): 
            d = np.int64(1)
        if not math.isfinite(float(f)): 
            f = np.int64(1)
        d_abs = min(abs(int(d)), 2**31 - 1)
        f_abs = min(abs(int(f)), 2**31 - 1)
        val = d_abs / (4.0 * f_abs + 1e-9)
        return 17.32 * math.sqrt(val if val >= 0 else 0.0)
    
    for si in out:
        R = safe_R(d, f)
        k = int(R) % 256
        bi = k ^ int(si)
        B.append(bi)
        f, d = update_df(R, bi, d)
    
    return [int(x) & 0xFF for x in B]


# ============================================================================
# MARKOV CHAIN CIPHER
# ============================================================================

class MarkovChainCipher:
    """
    Markov chain-based cipher for state-dependent transformations.
    """
    
    def __init__(self, key: str, num_states: int = 16):
        self.num_states = num_states
        self.seed = self._key_to_seed(key)
        
        np.random.seed(self.seed)
        self.transition_matrix = np.random.dirichlet(
            np.ones(num_states), size=num_states
        )
        
        self.emission_matrices = []
        self.inverse_emissions = []
        
        for state in range(num_states):
            perm = np.random.permutation(256)
            self.emission_matrices.append(perm)
            
            inverse = np.zeros(256, dtype=int)
            for i, v in enumerate(perm):
                inverse[v] = i
            self.inverse_emissions.append(inverse)
    
    def _key_to_seed(self, key: str) -> int:
        return sum(ord(c) * (i + 1) for i, c in enumerate(key)) % (2**31)
    
    def _select_next_state(self, current_state: int, pixel_value: int) -> int:
        probs = self.transition_matrix[current_state]
        cumsum = np.cumsum(probs)
        threshold = (pixel_value / 255.0)
        next_state = np.searchsorted(cumsum, threshold)
        return min(next_state, self.num_states - 1)
    
    def encrypt(self, block: list) -> list:
        encrypted = []
        current_state = 0
        
        for pixel in block:
            transformed = int(self.emission_matrices[current_state][pixel])
            encrypted.append(transformed)
            current_state = self._select_next_state(current_state, pixel)
        
        return encrypted
    
    def decrypt(self, encrypted_block: list) -> list:
        decrypted = []
        current_state = 0
        
        for encrypted_pixel in encrypted_block:
            original = int(self.inverse_emissions[current_state][encrypted_pixel])
            decrypted.append(original)
            current_state = self._select_next_state(current_state, original)
        
        return decrypted


# ============================================================================
# COMBINED FRESNEL + MARKOV (RECOMMENDED)
# ============================================================================

def Substitute_Markov(B, key: str):
    """
    Two-stage substitution: Fresnel + Markov Chain.
    RECOMMENDED for best security.
    """
    stage1 = Substitute(B)
    markov = MarkovChainCipher(key, num_states=16)
    stage2 = markov.encrypt(stage1)
    return stage2


def Substitute_Markov_Inv(encrypted, key: str):
    """Inverse of two-stage substitution."""
    markov = MarkovChainCipher(key, num_states=16)
    stage1 = markov.decrypt(encrypted)
    stage2 = Substitute_Inv(stage1)
    return stage2


# ============================================================================
# PERTURBATION FUNCTIONS (FIXED)
# ============================================================================

Seed_r = np.int64(0)
Seed_c = np.int64(0)

def Randomize(seed):
    """64-bit safe randomizer."""
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    seed = np.uint64(seed) & mask
    
    seed ^= (seed << np.uint64(21)) & mask
    seed ^= (seed >> np.uint64(35)) & mask
    seed ^= (seed << np.uint64(4)) & mask
    
    return np.int64(seed & mask)


def Update(r: int, c: int, s: int, N: int, M: int):
    """Update row and column positions."""
    global Seed_r, Seed_c
    
    Seed_r = np.int64(Seed_r ^ np.int64(s))
    Seed_c = np.int64(Seed_c ^ np.int64((s << 3) | (s >> 5)))
    
    Seed_r = Randomize(Seed_r)
    Seed_c = Randomize(Seed_c)
    
    r_new = int((int(Seed_r) % N) ^ int(r))
    c_new = int((int(Seed_c) % M) ^ int(c))
    
    r_new = r_new % N
    c_new = c_new % M
    
    return r_new, c_new


def Perturbation(Image: np.ndarray, r_init, c_init):
    """Scramble pixels using chaotic position updates."""
    global Seed_r, Seed_c
    N, M = Image.shape
    
    img = np.array(Image, copy=True)
    Image_p = np.full((N, M), -1, dtype=np.int32)
    
    Seed_r = np.int64(0)
    Seed_c = np.int64(0)
    
    def _map_to_index(v, length):
        if isinstance(v, (float, np.floating)):
            frac = v - math.floor(v)
            return int((frac * length)) % length
        else:
            return int(v) % length
    
    r = _map_to_index(r_init, N)
    c = _map_to_index(c_init, M)
    
    for i in range(N):
        for j in range(M):
            pixel = int(img[i, j])
            
            placed = False
            if Image_p[r, c] == -1:
                Image_p[r, c] = pixel
                placed = True
            else:
                for rr in range(N):
                    if Image_p[rr, c] == -1:
                        Image_p[rr, c] = pixel
                        placed = True
                        r = rr
                        break
                if not placed:
                    outer_break = False
                    for cc in range(M):
                        for rr in range(N):
                            if Image_p[rr, cc] == -1:
                                Image_p[rr, cc] = pixel
                                r, c = rr, cc
                                placed = True
                                outer_break = True
                                break
                        if outer_break:
                            break
            
            r, c = Update(r, c, pixel, N, M)
    
    Image_p[Image_p == -1] = 0
    return Image_p.astype(np.uint8)


def Perturbation_Inv(Image_p, r_init, c_init):
    """Restore original image from scrambled version."""
    global Seed_r, Seed_c
    N, M = Image_p.shape
    
    Image_p_copy = np.array(Image_p, dtype=np.int32, copy=True)
    Image = np.full((N, M), -1, dtype=np.int32)
    
    Seed_r = np.int64(0)
    Seed_c = np.int64(0)
    
    def _map_to_index(v, length):
        if isinstance(v, (float, np.floating)):
            frac = v - math.floor(v)
            return int((frac * length)) % length
        else:
            return int(v) % length
    
    r = _map_to_index(r_init, N)
    c = _map_to_index(c_init, M)
    
    for i in range(N):
        for j in range(M):
            pixel = None
            if Image_p_copy[r, c] != -1:
                pixel = Image_p_copy[r, c]
                Image_p_copy[r, c] = -1
            else:
                for rr in range(N):
                    if Image_p_copy[rr, c] != -1:
                        pixel = Image_p_copy[rr, c]
                        Image_p_copy[rr, c] = -1
                        r = rr
                        break
                if pixel is None:
                    found = False
                    for cc in range(M):
                        for rr in range(N):
                            if Image_p_copy[rr, cc] != -1:
                                pixel = Image_p_copy[rr, cc]
                                Image_p_copy[rr, cc] = -1
                                r, c = rr, cc
                                found = True
                                break
                        if found:
                            break
            
            if pixel is None:
                pixel = 0
            
            Image[i, j] = pixel
            r, c = Update(r, c, int(pixel), N, M)
    
    return Image.astype(np.uint8)


# ============================================================================
# TESTING
# ============================================================================

def test_all():
    """Comprehensive testing suite."""
    print("="*70)
    print("MEDICAL IMAGE ENCRYPTION - TESTING SUITE")
    print("="*70)
    
    test_key = "MySecretKey123"
    
    # Test 1: Substitution functions
    print("\n[TEST 1] Single-Pass Fresnel Substitution")
    print("-" * 70)
    test_blocks = [
        [120, 45, 200, 88],
        [0, 0, 0, 0],
        [255, 255, 255, 255],
        list(range(100)),
        [42] * 50
    ]
    
    all_passed = True
    for i, block in enumerate(test_blocks):
        try:
            encrypted = Substitute(block)
            decrypted = Substitute_Inv(encrypted)
            passed = (block == decrypted)
            print(f"  Test {i+1}: {'✓ PASSED' if passed else '✗ FAILED'} (size={len(block)})")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  Test {i+1}: ✗ FAILED - {str(e)[:50]}")
            all_passed = False
    
    print(f"\n  {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    
    # Test 2: Markov chain
    print("\n[TEST 2] Fresnel + Markov Chain Substitution")
    print("-" * 70)
    all_passed = True
    for i, block in enumerate(test_blocks):
        try:
            encrypted = Substitute_Markov(block, test_key)
            decrypted = Substitute_Markov_Inv(encrypted, test_key)
            passed = (block == decrypted)
            print(f"  Test {i+1}: {'✓ PASSED' if passed else '✗ FAILED'} (size={len(block)})")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  Test {i+1}: ✗ FAILED - {str(e)[:50]}")
            all_passed = False
    
    print(f"\n  {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    
    # Test 3: Perturbation
    print("\n[TEST 3] Pixel Perturbation")
    print("-" * 70)
    try:
        test_image = np.random.randint(0, 256, (50, 60), dtype=np.uint8)
        r_init, c_init = 0.5, 0.7
        
        perturbed = Perturbation(test_image.copy(), r_init, c_init)
        restored = Perturbation_Inv(perturbed.copy(), r_init, c_init)
        
        passed = np.array_equal(test_image, restored)
        print(f"  Perturbation Test: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        if not passed:
            diff = np.sum(test_image != restored)
            print(f"  Mismatches: {diff}/{test_image.size} pixels")
            print(f"  Accuracy: {100*(1-diff/test_image.size):.2f}%")
    except Exception as e:
        print(f"  Perturbation Test: ✗ FAILED - {str(e)}")
    
    # Security analysis
    print("\n[TEST 4] Security Analysis")
    print("-" * 70)
    try:
        original = list(range(100))
        encrypted = Substitute_Markov(original, test_key)
        
        # Avalanche effect
        original_flipped = original.copy()
        original_flipped[0] = (original_flipped[0] + 1) % 256
        encrypted_flipped = Substitute_Markov(original_flipped, test_key)
        
        diff_bits = sum(bin(a ^ b).count('1') for a, b in zip(encrypted, encrypted_flipped))
        avalanche = diff_bits / (len(encrypted) * 8)
        
        # Entropy
        from collections import Counter
        counts = Counter(encrypted)
        probs = np.array([count / len(encrypted) for count in counts.values()])
        entropy = -np.sum(probs * np.log2(probs))
        
        print(f"  Avalanche Effect: {avalanche:.2%} {'✓' if avalanche > 0.45 else '✗'}")
        print(f"  Ciphertext Entropy: {entropy:.2f} bits {'✓' if entropy > 7.0 else '✗'}")
    except Exception as e:
        print(f"  Security Test: ✗ FAILED - {str(e)}")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    test_all()
    
    print("\n" + "="*70)
    print("USAGE:")
    print("="*70)
    print("""
# In your encryption pipeline:

from forward_pass import Substitute_Markov, Substitute_Markov_Inv
from forward_pass import Perturbation, Perturbation_Inv

PASSWORD = "YourSecretPassword"

# Encrypt row
encrypted_row = Substitute_Markov(row.tolist(), PASSWORD)

# Decrypt row  
decrypted_row = Substitute_Markov_Inv(encrypted_row, PASSWORD)

# Perturbation
r_init, c_init = 0.5, 0.7  # From your logistic map
scrambled = Perturbation(image, r_init, c_init)
unscrambled = Perturbation_Inv(scrambled, r_init, c_init)
    """)
    print("="*70)