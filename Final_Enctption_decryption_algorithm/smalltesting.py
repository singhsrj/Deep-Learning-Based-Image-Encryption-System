import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import matplotlib.pyplot as plt

class LogisticMap:
    """
    Logistic map for generating chaotic sequences.
    According to paper Eq. (3): r_{i,d+1} = μ_i * r_{i,d} * (1 - r_{i,d})
    where r_{i,d} ∈ (0, 1) and μ_i ∈ [3.57, 4]
    """
    def __init__(self, r0, mu):
        if not (0 < r0 < 1):
            raise ValueError("Initial value r0 must be in (0, 1)")
        if not (3.57 <= mu <= 4.0):
            raise ValueError("System parameter mu must be in [3.57, 4]")
        
        self.r0 = r0
        self.mu = mu
    
    def generate_sequence(self, length, warm_up=100):
        """Generate chaotic sequence after warm-up iterations."""
        r = self.r0
        
        # Warm-up to reach chaotic regime
        for _ in range(warm_up):
            r = self.mu * r * (1 - r)
        
        # Generate sequence
        sequence = np.zeros(length)
        for i in range(length):
            r = self.mu * r * (1 - r)
            sequence[i] = r
        
        return sequence


def generate_dct_matrix(m, n):
    """
    Generate DCT coefficient matrix of size m × n.
    Creates a proper DCT transformation matrix.
    """
    W = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            if i == 0:
                W[i, j] = np.sqrt(1.0 / n)
            else:
                W[i, j] = np.sqrt(2.0 / n) * np.cos(np.pi * i * (2*j + 1) / (2*n))
    
    return W


class EDNN:
    """
    Encryption unit with Deep Neural Network (EDNN).
    
    Network structure (Fig. 1 in paper):
    - Input layer: x_1 ∈ R^{n_1 × 1}
    - Hidden layers (h-1 layers): x_{i+1} = ReLU(W_i * x_i), i = 1,...,h-1
    - Output layer: y = tanh(W_h * x_h)
    """
    
    def __init__(self, layer_sizes, logistic_params):
        """
        Initialize EDNN.
        
        Args:
            layer_sizes: List [n_1, n_2, ..., n_{h+1}]
            logistic_params: List of (r_{i,0}, μ_i) tuples
        """
        self.layer_sizes = layer_sizes
        self.h = len(layer_sizes) - 1  # Number of weight matrices
        self.logistic_params = logistic_params
        
        if len(logistic_params) != self.h:
            raise ValueError(f"Need {self.h} logistic map parameters")
        
        # Generate all weight matrices
        print(f"Generating {self.h} weight matrices...")
        self.weight_matrices = []
        for i in range(self.h):
            W = self._generate_scrambled_weight_matrix(
                layer_sizes[i+1], 
                layer_sizes[i],
                logistic_params[i],
                layer_index=i
            )
            self.weight_matrices.append(W)
            print(f"  W_{i+1} shape: {W.shape}")
    
    def _generate_scrambled_weight_matrix(self, m, n, logistic_param, layer_index):
        """Generate scrambled DCT coefficient matrix."""
        r0, mu = logistic_param
        
        # Step 1: Generate DCT coefficient matrix W̄_i
        W_bar = generate_dct_matrix(m, n)
        
        # Step 2: Generate chaotic sequence P_i for row scrambling
        logistic_map = LogisticMap(r0, mu)
        P_i = logistic_map.generate_sequence(m)
        
        # Step 3: Scramble rows according to ascending order of P_i
        sorted_indices = np.argsort(P_i)
        W_i = W_bar[sorted_indices, :]
        
        return W_i
    
    @staticmethod
    def relu(x):
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)
    
    @staticmethod
    def tanh(x):
        """Hyperbolic tangent activation"""
        return np.tanh(x)
    
    def encrypt(self, x_1):
        """
        Encrypt input vector through EDNN.
        
        Args:
            x_1: Input vector, shape (n_1, 1) or (n_1,)
        
        Returns:
            y: Encrypted vector, shape (n_{h+1}, 1)
        """
        # Ensure column vector
        if len(x_1.shape) == 1:
            x_1 = x_1.reshape(-1, 1)
        
        if x_1.shape[0] != self.layer_sizes[0]:
            raise ValueError(f"Input size must be {self.layer_sizes[0]}, got {x_1.shape[0]}")
        
        # Forward propagation through hidden layers with ReLU
        x = x_1
        for i in range(self.h - 1):  # h-1 hidden layers
            z = np.dot(self.weight_matrices[i], x)
            x = self.relu(z)
        
        # Output layer with tanh
        z = np.dot(self.weight_matrices[-1], x)
        y = self.tanh(z)
        
        return y
    
    def get_secret_keys(self):
        """Return secret keys."""
        keys = {}
        for i, (r0, mu) in enumerate(self.logistic_params):
            keys[f'r_{i+1}_0'] = r0
            keys[f'mu_{i+1}'] = mu
        return keys


class DDNN:
    """
    Decryption unit with Deep Neural Network (DDNN).
    
    According to paper (Section 2):
    - Network structure is symmetric with EDNN
    - Uses FISTA (Algorithm 1) to recover x_h from y
    - Uses AD-LPMM (Algorithm 2) to recover x_j from x_{j+1}
    """
    
    def __init__(self, ednn):
        """
        Initialize DDNN with same structure as EDNN.
        
        Args:
            ednn: EDNN instance (for weight matrices)
        """
        self.ednn = ednn
        self.layer_sizes = ednn.layer_sizes
        self.h = ednn.h
        self.weight_matrices = ednn.weight_matrices
    
    def fista(self, y, W_h, gamma=1e-5, K=100):
        """
        FISTA algorithm (Algorithm 1 in paper) to recover x_h from y.
        
        Solves: x̂_h = arg min (1/2)||y - tanh(W_h * x̂_h)||²_2 + γ * 1^T * x̂_h
        subject to: x̂_h ≥ 0
        
        Args:
            y: Encrypted output vector (n_{h+1} × 1)
            W_h: Weight matrix of output layer
            gamma: Regularization parameter (default: 1e-5 from paper)
            K: Number of iterations (reduced for 5x5)
        
        Returns:
            x̂_h: Recovered vector
        """
        n_h = W_h.shape[1]
        
        # Initialization
        x_hat = np.zeros((n_h, 1))
        d = np.zeros((n_h, 1))
        t = 1.0
        
        # Step 3: Compute step size α
        eigenvalues = np.linalg.eigvalsh(W_h.T @ W_h)
        alpha = 1.0 / np.max(eigenvalues)
        
        ones = np.ones((n_h, 1))
        
        # Iterations
        for k in range(K):
            # Step 5: Compute gradient
            Wx = W_h @ x_hat
            tanh_Wx = np.tanh(Wx)
            tanh_prime = 1 - tanh_Wx**2  # derivative of tanh
            
            # g = W_h^T * tanh'(W_h * x̂_h) ∘ [tanh(W_h * x̂_h) - y]
            residual = tanh_Wx - y
            g = W_h.T @ (tanh_prime * residual) + gamma * ones
            
            # Step 6: Update d with ReLU projection
            d_new = np.maximum(0, x_hat - alpha * g)
            
            # Step 7: Update t
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            
            # Step 8: Update x̂_h with momentum
            x_hat = d_new + ((t - 1) / t_new) * (d_new - d)
            
            d = d_new
            t = t_new
        
        return x_hat
    
    def ad_lpmm(self, x_j_plus_1, W_j, lambda_j=0.01, rho_j=1e-4, K_j=500):
        """
        AD-LPMM algorithm (Algorithm 2 in paper) to recover x_j from x_{j+1}.
        
        Solves: x̂_j = arg min (1/2)||x̂^S_{j+1} - W^S_j * x̂_j||²_2 + λ_j * 1^T * x̂_j
        subject to: x̂_j ≥ 0, W^{S^c}_j * x̂_j = 0
        
        Args:
            x_j_plus_1: Output of layer j (input to this recovery), shape (n_{j+1}, 1)
            W_j: Weight matrix of layer j
            lambda_j: Regularization parameter (default: 0.01 from paper)
            rho_j: Penalty parameter (default: 1e-4 from paper)
            K_j: Number of iterations (reduced for 5x5)
        
        Returns:
            x̂_j: Recovered vector
        """
        n_j = W_j.shape[1]
        
        # Step 3: Find indices of non-zero elements in x_{j+1}
        S = np.where(x_j_plus_1.flatten() > 1e-10)[0]  # Support set
        S_c = np.setdiff1d(np.arange(x_j_plus_1.shape[0]), S)  # Complement
        
        # Extract subsets
        x_S_j_plus_1 = x_j_plus_1[S]
        
        # Step 4: Extract sub-matrices
        W_S_j = W_j[S, :]
        W_Sc_j = W_j[S_c, :]
        
        # Step 5: Compute step size η
        if len(S) > 0:
            eigenval_S = np.linalg.eigvalsh(W_S_j.T @ W_S_j + rho_j * W_Sc_j.T @ W_Sc_j)
            eta = np.max(eigenval_S) + 0.1
        else:
            eta = 1.0
        
        # Initialization (Step 2)
        x_hat = np.zeros((n_j, 1))
        b = np.zeros((len(S_c), 1))
        u = np.zeros((len(S_c), 1))
        
        ones = np.ones((n_j, 1))
        
        # Iterations
        for k in range(K_j):
            # Step 7: Update x̂_j with ReLU projection
            if len(S) > 0:
                grad_term = W_S_j.T @ (W_S_j @ x_hat - x_S_j_plus_1)
            else:
                grad_term = 0
            
            if len(S_c) > 0:
                penalty_term = rho_j * W_Sc_j.T @ (W_Sc_j @ x_hat - b - u)
            else:
                penalty_term = 0
            
            x_hat = np.maximum(0, x_hat - (grad_term + penalty_term + lambda_j * ones) / eta)
            
            # Step 8: Update b
            if len(S_c) > 0:
                b = -np.maximum(0, u - W_Sc_j @ x_hat)
            
            # Step 9: Update u
            if len(S_c) > 0:
                u = u + b - W_Sc_j @ x_hat
        
        return x_hat
    
    def decrypt(self, y, gamma=1e-5, lambda_j=0.01, rho_j=1e-4, K=100, K_j=500):
        """
        Decrypt encrypted vector y to recover original x_1.
        
        Process:
        1. Use FISTA to recover x_h from y (output layer)
        2. Use AD-LPMM to recover x_j from x_{j+1} for each hidden layer
        
        Args:
            y: Encrypted vector (n_{h+1} × 1)
            gamma, lambda_j, rho_j: Regularization parameters from paper
            K, K_j: Number of iterations (reduced for 5x5)
        
        Returns:
            x̂_1: Recovered original vector
        """
        # Ensure column vector
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        print(f"  Starting decryption...")
        
        # Step 1: Recover x_h from y using FISTA (Algorithm 1)
        print(f"    Recovering layer {self.h} using FISTA...")
        W_h = self.weight_matrices[-1]
        x_hat = self.fista(y, W_h, gamma=gamma, K=K)
        
        # Step 2: Recover x_j from x_{j+1} for j = h-1, h-2, ..., 1 using AD-LPMM
        for j in range(self.h - 2, -1, -1):  # From h-1 down to 0
            print(f"    Recovering layer {j+1} using AD-LPMM...")
            W_j = self.weight_matrices[j]
            x_hat = self.ad_lpmm(x_hat, W_j, lambda_j=lambda_j, rho_j=rho_j, K_j=K_j)
        
        return x_hat


class ImageEncryptionSystem:
    """
    Complete image encryption/decryption system.
    Modified for 5×5 images for testing.
    """
    
    def __init__(self, ednn):
        self.ednn = ednn
        self.ddnn = DDNN(ednn)
    
    def preprocess_image(self, image):
        """Preprocess image into normalized vector."""
        if image.shape != (5, 5):
            raise ValueError(f"Image must be 5×5, got {image.shape}")
        
        # Flatten and normalize to [0, 1]
        vector = image.flatten().astype(np.float64) / 255.0
        
        return vector
    
    def encrypt_image(self, image):
        """Encrypt entire image."""
        # Preprocess into single vector
        image_vector = self.preprocess_image(image)
        
        # Encrypt
        print(f"Encrypting 5×5 image...")
        encrypted = self.ednn.encrypt(image_vector)
        
        # Reshape encrypted data for visualization
        encrypted_image = self._reshape_to_image(encrypted)
        
        return encrypted, encrypted_image
    
    def decrypt_image(self, encrypted_vector, gamma=1e-5, lambda_j=0.01, 
                     rho_j=1e-4, K=100, K_j=500):
        """
        Decrypt image from encrypted vector.
        
        Args:
            encrypted_vector: Encrypted vector
            gamma, lambda_j, rho_j, K, K_j: Parameters (reduced iterations for 5x5)
        
        Returns:
            decrypted_image: Reconstructed 5×5 image
        """
        # Decrypt
        print(f"Decrypting 5×5 image...")
        decrypted = self.ddnn.decrypt(
            encrypted_vector, 
            gamma=gamma, 
            lambda_j=lambda_j, 
            rho_j=rho_j, 
            K=K, 
            K_j=K_j
        )
        
        # Reconstruct image
        decrypted_image = self._reconstruct_image(decrypted)
        
        return decrypted_image
    
    def _reconstruct_image(self, vector):
        """
        Reconstruct 5×5 image from decrypted vector.
        
        Args:
            vector: Decrypted vector of size 25×1
        
        Returns:
            5×5 reconstructed image
        """
        # Reshape to 5×5
        vector_data = vector.flatten()
        # Denormalize from [0, 1] to [0, 255]
        vector_data = np.clip(vector_data * 255, 0, 255).astype(np.uint8)
        image = vector_data.reshape(5, 5)
        
        return image
    
    def _reshape_to_image(self, encrypted_data):
        """Reshape encrypted data to square image for visualization."""
        data = encrypted_data.flatten()
        
        # Calculate closest square size
        size = int(np.ceil(np.sqrt(len(data))))
        target_size = size * size
        
        # Pad if necessary
        if data.shape[0] < target_size:
            data = np.pad(data, (0, target_size - data.shape[0]))
        else:
            data = data[:target_size]
        
        # Normalize to [0, 255] for visualization
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            data_normalized = (data - data_min) / (data_max - data_min)
        else:
            data_normalized = np.zeros_like(data)
        
        image_data = (data_normalized * 255).astype(np.uint8)
        encrypted_image = image_data.reshape(size, size)
        
        return encrypted_image


def calculate_psnr(original, reconstructed):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    Used in paper to evaluate decryption quality.
    """
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def test_complete_system():
    """Test complete encryption and decryption system with 5×5 images."""
    
    print("="*70)
    print("EDNN IMAGE ENCRYPTION - 5×5 TEST (FAST PERFORMANCE)")
    print("="*70)
    
    # Scaled down configuration for 5×5 images (25 pixels)
    print("\n1. Configuring system for 5×5 images:")
    print("   Layer sizes: [25, 30, 35]")
    print("   Logistic params: [(0.3, 3.99), (0.4, 3.99)]")
    print("   Decryption params: γ=1e-5, λ=0.01, ρ=1e-4, K=100, K_j=500")
    
    layer_sizes = [25, 30, 35]  # 5×5 = 25 pixels
    logistic_params = [(0.3, 3.99), (0.4, 3.99)]
    
    # Initialize system
    print("\n2. Initializing EDNN...")
    ednn = EDNN(layer_sizes, logistic_params)
    
    print("\n3. Creating encryption/decryption system...")
    system = ImageEncryptionSystem(ednn)
    
    # Create test image 5×5
    print("\n4. Creating test image (5×5)...")
    test_image = np.array([
        [255, 200, 150, 100,  50],
        [200, 255, 180, 120,  60],
        [150, 180, 255, 140,  70],
        [100, 120, 140, 255,  80],
        [ 50,  60,  70,  80, 255]
    ], dtype=np.uint8)
    
    # Encrypt
    print("\n5. ENCRYPTION PHASE:")
    encrypted_vector, encrypted_image = system.encrypt_image(test_image)
    print(f"   ✓ Encryption complete")
    print(f"   Original: {test_image.shape}, Encrypted: {encrypted_image.shape}")
    
    # Decrypt
    print("\n6. DECRYPTION PHASE:")
    decrypted_image = system.decrypt_image(
        encrypted_vector,
        gamma=1e-5,
        lambda_j=0.01,
        rho_j=1e-4,
        K=100,
        K_j=500
    )
    print(f"   ✓ Decryption complete")
    
    # Calculate PSNR
    psnr = calculate_psnr(test_image, decrypted_image)
    print(f"\n7. QUALITY ASSESSMENT:")
    print(f"   PSNR = {psnr:.4f} dB")
    if psnr > 30:
        print(f"   Quality: Excellent (PSNR > 30 dB)")
    elif psnr > 20:
        print(f"   Quality: Good (PSNR > 20 dB)")
    else:
        print(f"   Quality: Fair")
    
    # Visualization
    print("\n8. Generating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(test_image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    axes[0].set_title('(a) Original Image\n5×5', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(encrypted_image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    axes[1].set_title(f'(b) Encrypted Image\n{encrypted_image.shape[0]}×{encrypted_image.shape[1]}', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(decrypted_image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    axes[2].set_title(f'(c) Decrypted Image\n5×5\nPSNR = {psnr:.4f} dB', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('encryption_5x5_test.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved as 'encryption_5x5_test.png'")
    
    # Statistics
    print("\n9. STATISTICAL COMPARISON:")
    print(f"   Original    - Mean: {test_image.mean():.2f}, Std: {test_image.std():.2f}")
    print(f"   Encrypted   - Mean: {encrypted_image.mean():.2f}, Std: {encrypted_image.std():.2f}")
    print(f"   Decrypted   - Mean: {decrypted_image.mean():.2f}, Std: {decrypted_image.std():.2f}")
    
    # Pixel comparison
    print("\n10. PIXEL-BY-PIXEL COMPARISON:")
    print("    Original:")
    print(test_image)
    print("    Decrypted:")
    print(decrypted_image)
    print("    Absolute Difference:")
    print(np.abs(test_image.astype(int) - decrypted_image.astype(int)))
    
    print("\n" + "="*70)
    print("✓ FAST 5×5 TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    return system, test_image, encrypted_image, decrypted_image, psnr


if __name__ == "__main__":
    system, original, encrypted, decrypted, psnr = test_complete_system()