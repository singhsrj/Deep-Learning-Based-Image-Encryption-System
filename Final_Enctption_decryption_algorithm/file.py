# import numpy as np
# from scipy.fftpack import dct, idct
# from PIL import Image
# import matplotlib.pyplot as plt

# class LogisticMap:
#     """
#     Logistic map for generating chaotic sequences.
#     According to paper Eq. (3): r_{i,d+1} = μ_i * r_{i,d} * (1 - r_{i,d})
#     where r_{i,d} ∈ (0, 1) and μ_i ∈ [3.57, 4]
#     """
#     def __init__(self, r0, mu):
#         if not (0 < r0 < 1):
#             raise ValueError("Initial value r0 must be in (0, 1)")
#         if not (3.57 <= mu <= 4.0):
#             raise ValueError("System parameter mu must be in [3.57, 4]")
        
#         self.r0 = r0
#         self.mu = mu
    
#     def generate_sequence(self, length, warm_up=1000):
#         """Generate chaotic sequence after warm-up iterations."""
#         r = self.r0
        
#         # Warm-up to reach chaotic regime
#         for _ in range(warm_up):
#             r = self.mu * r * (1 - r)
        
#         # Generate sequence
#         sequence = np.zeros(length)
#         for i in range(length):
#             r = self.mu * r * (1 - r)
#             sequence[i] = r
        
#         return sequence


# def generate_dct_matrix(m, n):
#     """
#     Generate DCT coefficient matrix of size m × n.
#     Creates a proper DCT transformation matrix.
    
#     Args:
#         m: Number of rows
#         n: Number of columns
    
#     Returns:
#         DCT coefficient matrix of size m × n
#     """
#     # Generate DCT basis matrix
#     # For a proper DCT matrix, we create orthonormal basis
#     W = np.zeros((m, n))
    
#     for i in range(m):
#         for j in range(n):
#             if i == 0:
#                 W[i, j] = np.sqrt(1.0 / n)
#             else:
#                 W[i, j] = np.sqrt(2.0 / n) * np.cos(np.pi * i * (2*j + 1) / (2*n))
    
#     return W


# class EDNN:
#     """
#     Encryption unit with Deep Neural Network (EDNN).
    
#     Network structure (Fig. 1 in paper):
#     - Input layer: x_1 ∈ R^{n_1 × 1}
#     - Hidden layers (h-1 layers): x_{i+1} = ReLU(W_i * x_i), i = 1,...,h-1
#     - Output layer: y = tanh(W_h * x_h)
    
#     Weight matrices are scrambled DCT coefficient matrices.
#     """
    
#     def __init__(self, layer_sizes, logistic_params):
#         """
#         Initialize EDNN.
        
#         Args:
#             layer_sizes: List [n_1, n_2, ..., n_{h+1}]
#                 Paper config: [4096, 9025, 10000]
#             logistic_params: List of (r_{i,0}, μ_i) tuples
#                 Paper config: [(0.3, 3.99), (0.4, 3.99)]
#         """
#         self.layer_sizes = layer_sizes
#         self.h = len(layer_sizes) - 1  # Number of weight matrices (h in paper)
#         self.logistic_params = logistic_params
        
#         if len(logistic_params) != self.h:
#             raise ValueError(f"Need {self.h} logistic map parameters")
        
#         # Validate dimensions according to paper
#         # Hidden layers: n_{i+1} < n_i (for compression)
#         # Output layer: n_{h+1} > n_h (for expansion)
#         for i in range(self.h - 1):
#             if layer_sizes[i+1] >= layer_sizes[i]:
#                 print(f"Warning: Hidden layer {i+1} size should be < layer {i} size")
        
#         if layer_sizes[-1] <= layer_sizes[-2]:
#             print(f"Warning: Output layer size should be > last hidden layer size")
        
#         # Generate all weight matrices
#         print(f"Generating {self.h} weight matrices...")
#         self.weight_matrices = []
#         for i in range(self.h):
#             W = self._generate_scrambled_weight_matrix(
#                 layer_sizes[i+1], 
#                 layer_sizes[i],
#                 logistic_params[i],
#                 layer_index=i
#             )
#             self.weight_matrices.append(W)
#             print(f"  W_{i+1} shape: {W.shape}")
    
#     def _generate_scrambled_weight_matrix(self, m, n, logistic_param, layer_index):
#         """
#         Generate scrambled DCT coefficient matrix.
        
#         Process (from paper Section 2):
#         1. Generate DCT coefficient matrix W̄_i of size m × n
#         2. Generate random sequence P_i using logistic map
#         3. Sort rows of W̄_i in ascending order of P_i to get W_i
        
#         Args:
#             m: Number of rows (n_{i+1})
#             n: Number of columns (n_i)
#             logistic_param: Tuple (r_{i,0}, μ_i)
#             layer_index: Index for identification
        
#         Returns:
#             Scrambled weight matrix W_i of size m × n
#         """
#         r0, mu = logistic_param
        
#         # Step 1: Generate DCT coefficient matrix W̄_i
#         print(f"    Generating DCT matrix for layer {layer_index+1}...")
#         W_bar = generate_dct_matrix(m, n)
        
#         # Step 2: Generate chaotic sequence P_i for row scrambling
#         logistic_map = LogisticMap(r0, mu)
#         P_i = logistic_map.generate_sequence(m)
        
#         # Step 3: Scramble rows according to ascending order of P_i
#         sorted_indices = np.argsort(P_i)
#         W_i = W_bar[sorted_indices, :]
        
#         return W_i
    
#     @staticmethod
#     def relu(x):
#         """ReLU activation: max(0, x)"""
#         return np.maximum(0, x)
    
#     @staticmethod
#     def tanh(x):
#         """Hyperbolic tangent activation"""
#         return np.tanh(x)
    
#     def encrypt(self, x_1):
#         """
#         Encrypt input vector through EDNN.
        
#         Process:
#         - Hidden layers: x_{i+1} = ReLU(W_i * x_i), Eq. (1)
#         - Output layer: y = tanh(W_h * x_h), Eq. (2)
        
#         Args:
#             x_1: Input vector, shape (n_1, 1) or (n_1,)
        
#         Returns:
#             y: Encrypted vector, shape (n_{h+1}, 1)
#         """
#         # Ensure column vector
#         if len(x_1.shape) == 1:
#             x_1 = x_1.reshape(-1, 1)
        
#         if x_1.shape[0] != self.layer_sizes[0]:
#             raise ValueError(f"Input size must be {self.layer_sizes[0]}, got {x_1.shape[0]}")
        
#         # Forward propagation through hidden layers with ReLU
#         x = x_1
#         for i in range(self.h - 1):  # h-1 hidden layers
#             z = np.dot(self.weight_matrices[i], x)
#             x = self.relu(z)
        
#         # Output layer with tanh
#         z = np.dot(self.weight_matrices[-1], x)
#         y = self.tanh(z)
        
#         return y
    
#     def get_secret_keys(self):
#         """Return secret keys (initial values and parameters of logistic maps)."""
#         keys = {}
#         for i, (r0, mu) in enumerate(self.logistic_params):
#             keys[f'r_{i+1}_0'] = r0
#             keys[f'mu_{i+1}'] = mu
#         return keys


# class ImageEncryptionSystem:
#     """
#     Complete image encryption system.
    
#     Paper's process (Section 3):
#     - Input: 128×128 grayscale image
#     - Divide into 4 non-overlapping 64×64 sub-blocks
#     - Each sub-block → 4096×1 vector, normalized to [0,1]
#     - Encrypt each vector through EDNN
#     - Output: 200×200 encrypted image
#     """
    
#     def __init__(self, ednn):
#         self.ednn = ednn
    
#     def preprocess_image(self, image):
#         """
#         Preprocess image into normalized block vectors.
        
#         Args:
#             image: 128×128 grayscale image (numpy array)
        
#         Returns:
#             List of 4 normalized vectors, each of size 4096×1
#         """
#         if image.shape != (128, 128):
#             raise ValueError(f"Image must be 128×128, got {image.shape}")
        
#         blocks = []
#         # Divide into 4 blocks of 64×64
#         for i in range(2):
#             for j in range(2):
#                 block = image[i*64:(i+1)*64, j*64:(j+1)*64]
#                 # Flatten and normalize to [0, 1]
#                 vector = block.flatten().astype(np.float64) / 255.0
#                 blocks.append(vector)
        
#         return blocks
    
#     def encrypt_image(self, image):
#         """
#         Encrypt entire image.
        
#         Args:
#             image: 128×128 grayscale image
        
#         Returns:
#             encrypted_vectors: List of 4 encrypted vectors
#             encrypted_image: 200×200 encrypted image for visualization
#         """
#         # Preprocess into 4 blocks
#         block_vectors = self.preprocess_image(image)
        
#         # Encrypt each block
#         encrypted_vectors = []
#         for i, block in enumerate(block_vectors):
#             print(f"Encrypting block {i+1}/4...")
#             encrypted = self.ednn.encrypt(block)
#             encrypted_vectors.append(encrypted)
        
#         # Concatenate all encrypted vectors
#         encrypted_data = np.concatenate(encrypted_vectors, axis=0)
        
#         # Reshape to 200×200 for visualization
#         # Total: 4 blocks × 10000 = 40000 pixels = 200×200
#         encrypted_image = self._reshape_to_image(encrypted_data)
        
#         return encrypted_vectors, encrypted_image
    
#     def _reshape_to_image(self, encrypted_data):
#         """
#         Reshape encrypted data to 200×200 image.
        
#         Args:
#             encrypted_data: Concatenated encrypted vectors (40000×1)
        
#         Returns:
#             200×200 image array
#         """
#         # Flatten
#         data = encrypted_data.flatten()
        
#         # Should be 40000 elements (4 blocks × 10000)
#         if data.shape[0] != 40000:
#             print(f"Warning: Expected 40000 elements, got {data.shape[0]}")
#             # Pad or truncate
#             if data.shape[0] < 40000:
#                 data = np.pad(data, (0, 40000 - data.shape[0]))
#             else:
#                 data = data[:40000]
        
#         # Normalize to [0, 255] for visualization
#         data_min, data_max = data.min(), data.max()
#         if data_max > data_min:
#             data_normalized = (data - data_min) / (data_max - data_min)
#         else:
#             data_normalized = np.zeros_like(data)
        
#         image_data = (data_normalized * 255).astype(np.uint8)
        
#         # Reshape to 200×200
#         encrypted_image = image_data.reshape(200, 200)
        
#         return encrypted_image


# def test_encryption_system():
#     """Test the encryption system with paper's configuration."""
    
#     print("="*60)
#     print("EDNN Image Encryption System")
#     print("Based on: 'A novel image encryption algorithm with deep neural network'")
#     print("="*60)
    
#     # Paper's configuration (Section 3)
#     print("\n1. Configuring EDNN with paper's parameters:")
#     print("   - h = 2 (3 layers: input + 1 hidden + output)")
#     print("   - Layer sizes: n_1=4096, n_2=9025, n_3=10000")
#     print("   - r_1,0=0.3, r_2,0=0.4, μ_1=μ_2=3.99")
    
#     layer_sizes = [4096, 9025, 10000]
#     logistic_params = [(0.3, 3.99), (0.4, 3.99)]
    
#     # Initialize EDNN
#     print("\n2. Initializing EDNN...")
#     ednn = EDNN(layer_sizes, logistic_params)
    
#     # Display secret keys
#     print("\n3. Secret Keys (for decryption):")
#     keys = ednn.get_secret_keys()
#     for key, value in keys.items():
#         print(f"   {key} = {value}")
    
#     # Key space calculation (from paper)
#     print(f"\n4. Key Space Analysis:")
#     print(f"   With precision 10^15, key space = (10^15)^4 = 10^60 ≈ 2^199")
#     print(f"   This exceeds theoretical requirement of 2^100")
    
#     # Create encryption system
#     encryption_system = ImageEncryptionSystem(ednn)
    
#     # Generate or load test image
#     print("\n5. Generating test image (128×128)...")
#     # Create a test pattern
#     test_image = np.zeros((128, 128), dtype=np.uint8)
#     # Add some patterns
#     test_image[20:50, 20:50] = 200
#     test_image[60:100, 60:100] = 150
#     test_image[10:120, 10:120] += np.random.randint(0, 50, (110, 110), dtype=np.uint8)
    
#     # Encrypt
#     print("\n6. Encrypting image...")
#     encrypted_vectors, encrypted_image = encryption_system.encrypt_image(test_image)
    
#     print(f"   Original image: {test_image.shape}")
#     print(f"   Encrypted image: {encrypted_image.shape}")
#     print(f"   Number of encrypted vectors: {len(encrypted_vectors)}")
#     print(f"   Each encrypted vector size: {encrypted_vectors[0].shape}")
    
#     # Visualization
#     print("\n7. Visualizing results...")
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
#     axes[0].imshow(test_image, cmap='gray', vmin=0, vmax=255)
#     axes[0].set_title('Original Image (128×128)', fontsize=14, fontweight='bold')
#     axes[0].axis('off')
    
#     axes[1].imshow(encrypted_image, cmap='gray', vmin=0, vmax=255)
#     axes[1].set_title('Encrypted Image (200×200)', fontsize=14, fontweight='bold')
#     axes[1].axis('off')
    
#     plt.tight_layout()
#     plt.savefig('ednn_encryption_result.png', dpi=150, bbox_inches='tight')
#     print("   Saved as 'ednn_encryption_result.png'")
    
#     # Statistics
#     print("\n8. Encryption Statistics:")
#     print(f"   Original image - Mean: {test_image.mean():.2f}, Std: {test_image.std():.2f}")
#     print(f"   Encrypted image - Mean: {encrypted_image.mean():.2f}, Std: {encrypted_image.std():.2f}")
    
#     print("\n" + "="*60)
#     print("Encryption completed successfully!")
#     print("="*60)
    
#     return ednn, encryption_system, test_image, encrypted_image


# if __name__ == "__main__":
#     ednn, encryption_system, original, encrypted = test_encryption_system()




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
    
    def generate_sequence(self, length, warm_up=1000):
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
    
    def fista(self, y, W_h, gamma=1e-5, K=500):
        """
        FISTA algorithm (Algorithm 1 in paper) to recover x_h from y.
        
        Solves: x̂_h = arg min (1/2)||y - tanh(W_h * x̂_h)||²_2 + γ * 1^T * x̂_h
        subject to: x̂_h ≥ 0
        
        Args:
            y: Encrypted output vector (n_{h+1} × 1)
            W_h: Weight matrix of output layer
            gamma: Regularization parameter (default: 1e-5 from paper)
            K: Number of iterations (default: 500 from paper)
        
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
    
    def ad_lpmm(self, x_j_plus_1, W_j, lambda_j=0.01, rho_j=1e-4, K_j=5000):
        """
        AD-LPMM algorithm (Algorithm 2 in paper) to recover x_j from x_{j+1}.
        
        Solves: x̂_j = arg min (1/2)||x̂^S_{j+1} - W^S_j * x̂_j||²_2 + λ_j * 1^T * x̂_j
        subject to: x̂_j ≥ 0, W^{S^c}_j * x̂_j = 0
        
        Args:
            x_j_plus_1: Output of layer j (input to this recovery), shape (n_{j+1}, 1)
            W_j: Weight matrix of layer j
            lambda_j: Regularization parameter (default: 0.01 from paper)
            rho_j: Penalty parameter (default: 1e-4 from paper)
            K_j: Number of iterations (default: 5000 from paper)
        
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
    
    def decrypt(self, y, gamma=1e-5, lambda_j=0.01, rho_j=1e-4, K=500, K_j=5000):
        """
        Decrypt encrypted vector y to recover original x_1.
        
        Process:
        1. Use FISTA to recover x_h from y (output layer)
        2. Use AD-LPMM to recover x_j from x_{j+1} for each hidden layer
        
        Args:
            y: Encrypted vector (n_{h+1} × 1)
            gamma, lambda_j, rho_j: Regularization parameters from paper
            K, K_j: Number of iterations from paper
        
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
    """
    
    def __init__(self, ednn):
        self.ednn = ednn
        self.ddnn = DDNN(ednn)
    
    def preprocess_image(self, image):
        """Preprocess image into normalized block vectors."""
        if image.shape != (128, 128):
            raise ValueError(f"Image must be 128×128, got {image.shape}")
        
        blocks = []
        # Divide into 4 blocks of 64×64
        for i in range(2):
            for j in range(2):
                block = image[i*64:(i+1)*64, j*64:(j+1)*64]
                # Flatten and normalize to [0, 1]
                vector = block.flatten().astype(np.float64) / 255.0
                blocks.append(vector)
        
        return blocks
    
    def encrypt_image(self, image):
        """Encrypt entire image."""
        # Preprocess into 4 blocks
        block_vectors = self.preprocess_image(image)
        
        # Encrypt each block
        encrypted_vectors = []
        for i, block in enumerate(block_vectors):
            print(f"Encrypting block {i+1}/4...")
            encrypted = self.ednn.encrypt(block)
            encrypted_vectors.append(encrypted)
        
        # Concatenate all encrypted vectors
        encrypted_data = np.concatenate(encrypted_vectors, axis=0)
        
        # Reshape to 200×200 for visualization
        encrypted_image = self._reshape_to_image(encrypted_data)
        
        return encrypted_vectors, encrypted_image
    
    def decrypt_image(self, encrypted_vectors, gamma=1e-5, lambda_j=0.01, 
                     rho_j=1e-4, K=500, K_j=5000):
        """
        Decrypt image from encrypted vectors.
        
        Args:
            encrypted_vectors: List of 4 encrypted vectors
            gamma, lambda_j, rho_j, K, K_j: Parameters from paper (Section 3)
        
        Returns:
            decrypted_image: Reconstructed 128×128 image
        """
        # Decrypt each block
        decrypted_blocks = []
        for i, encrypted_vector in enumerate(encrypted_vectors):
            print(f"Decrypting block {i+1}/4...")
            decrypted = self.ddnn.decrypt(
                encrypted_vector, 
                gamma=gamma, 
                lambda_j=lambda_j, 
                rho_j=rho_j, 
                K=K, 
                K_j=K_j
            )
            decrypted_blocks.append(decrypted)
        
        # Reconstruct image from blocks
        decrypted_image = self._reconstruct_image(decrypted_blocks)
        
        return decrypted_image
    
    def _reconstruct_image(self, blocks):
        """
        Reconstruct 128×128 image from 4 decrypted blocks.
        
        Args:
            blocks: List of 4 vectors, each of size 4096×1
        
        Returns:
            128×128 reconstructed image
        """
        image = np.zeros((128, 128), dtype=np.uint8)
        
        block_idx = 0
        for i in range(2):
            for j in range(2):
                # Reshape block to 64×64
                block_data = blocks[block_idx].flatten()
                # Denormalize from [0, 1] to [0, 255]
                block_data = np.clip(block_data * 255, 0, 255).astype(np.uint8)
                block_2d = block_data.reshape(64, 64)
                
                # Place in image
                image[i*64:(i+1)*64, j*64:(j+1)*64] = block_2d
                block_idx += 1
        
        return image
    
    def _reshape_to_image(self, encrypted_data):
        """Reshape encrypted data to 200×200 image."""
        data = encrypted_data.flatten()
        
        if data.shape[0] != 40000:
            if data.shape[0] < 40000:
                data = np.pad(data, (0, 40000 - data.shape[0]))
            else:
                data = data[:40000]
        
        # Normalize to [0, 255] for visualization
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            data_normalized = (data - data_min) / (data_max - data_min)
        else:
            data_normalized = np.zeros_like(data)
        
        image_data = (data_normalized * 255).astype(np.uint8)
        encrypted_image = image_data.reshape(200, 200)
        
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


def load_and_prepare_image(image_path):
    """
    Load an image from file and prepare it for encryption.
    
    Args:
        image_path: Path to image file (PNG, JPG, etc.)
    
    Returns:
        Grayscale image resized to 128×128
    """
    try:
        # Load image
        img = Image.open(image_path)
        print(f"   Loaded image: {img.size} (W×H), Mode: {img.mode}")
        
        # Convert to grayscale
        img_gray = img.convert('L')
        
        # Resize to 128×128
        img_resized = img_gray.resize((128, 128), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img_resized)
        print(f"   Prepared: {img_array.shape} grayscale image")
        return img_array
    
    except FileNotFoundError:
        print(f"   ERROR: File '{image_path}' not found!")
        return None
    except Exception as e:
        print(f"   ERROR: {str(e)}")
        return None


def test_complete_system(image_path=None):
    """
    Test complete encryption and decryption system.
    
    Args:
        image_path: Optional path to your image file. If None, uses generated test image.
    """
    
    print("="*70)
    print("COMPLETE EDNN IMAGE ENCRYPTION AND DECRYPTION SYSTEM")
    print("="*70)
    
    # Paper's configuration
    print("\n1. Configuring system with paper's parameters:")
    print("   Layer sizes: [4096, 9025, 10000]")
    print("   Logistic params: [(0.3, 3.99), (0.4, 3.99)]")
    print("   Decryption params: γ=1e-5, λ=0.01, ρ=1e-4, K=500, K_j=5000")
    
    layer_sizes = [4096, 9025, 10000]
    logistic_params = [(0.3, 3.99), (0.4, 3.99)]
    
    # Initialize EDNN
    print("\n2. Initializing EDNN...")
    ednn = EDNN(layer_sizes, logistic_params)
    
    print("\n3. Creating encryption/decryption system...")
    system = ImageEncryptionSystem(ednn)
    
    # Load or create test image
    if image_path:
        print(f"\n4. Loading your image from: {image_path}")
        test_image = load_and_prepare_image(image_path)
        if test_image is None:
            print("   Falling back to generated test image...")
            image_path = None
    
    if not image_path:
        print("\n4. Creating test image (128×128)...")
        test_image = np.zeros((128, 128), dtype=np.uint8)
        # Add patterns
        test_image[20:50, 20:50] = 200
        test_image[70:110, 70:110] = 150
        test_image[40:90, 10:60] = 100
        # Add noise for texture
        noise = np.random.randint(0, 30, (128, 128), dtype=np.uint8)
        test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Encrypt
    print("\n5. ENCRYPTION PHASE:")
    encrypted_vectors, encrypted_image = system.encrypt_image(test_image)
    print(f"   ✓ Encryption complete")
    print(f"   Original: {test_image.shape}, Encrypted: {encrypted_image.shape}")
    
    # Decrypt
    print("\n6. DECRYPTION PHASE:")
    decrypted_image = system.decrypt_image(
        encrypted_vectors,
        gamma=1e-5,
        lambda_j=0.01,
        rho_j=1e-4,
        K=500,
        K_j=5000
    )
    print(f"   ✓ Decryption complete")
    
    # Calculate PSNR
    psnr = calculate_psnr(test_image, decrypted_image)
    print(f"\n7. QUALITY ASSESSMENT:")
    print(f"   PSNR = {psnr:.4f} dB")
    print(f"   (Paper reports: House=38.21dB, Pagodas=30.94dB, Bridge=33.39dB)")
    
    # Visualization
    print("\n8. Generating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(test_image, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('(a) Original Image\n128×128', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(encrypted_image, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('(b) Encrypted Image\n200×200', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(decrypted_image, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'(c) Decrypted Image\n128×128\nPSNR = {psnr:.4f} dB', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('complete_encryption_decryption.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved as 'complete_encryption_decryption.png'")
    
    # Statistics
    print("\n9. STATISTICAL COMPARISON:")
    print(f"   Original    - Mean: {test_image.mean():.2f}, Std: {test_image.std():.2f}")
    print(f"   Encrypted   - Mean: {encrypted_image.mean():.2f}, Std: {encrypted_image.std():.2f}")
    print(f"   Decrypted   - Mean: {decrypted_image.mean():.2f}, Std: {decrypted_image.std():.2f}")
    
    print("\n" + "="*70)
    print("✓ SYSTEM TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    return system, test_image, encrypted_image, decrypted_image, psnr


# if __name__ == "__main__":
#     # Example 1: Use a test image (generated)
#     print("EXAMPLE 1: Using generated test image")
#     print("-" * 70)
#     system, original, encrypted, decrypted, psnr = test_complete_system(image_path="")
    
#     print("\n\n")
#     print("="*70)
#     print("EXAMPLE 2: How to use YOUR OWN IMAGE")
#     print("="*70)
#     print("""
# To encrypt your own image, use one of these methods:

# METHOD 1: Direct function call with image path
# --------------------------------------------------
# system, original, encrypted, decrypted, psnr = test_complete_system('path/to/your/image.jpg')

# METHOD 2: Load and encrypt separately
# --------------------------------------------------
# # Load your image
# my_image = load_and_prepare_image('path/to/your/image.png')

# # Initialize system
# layer_sizes = [4096, 9025, 10000]
# logistic_params = [(0.3, 3.99), (0.4, 3.99)]
# ednn = EDNN(layer_sizes, logistic_params)
# system = ImageEncryptionSystem(ednn)

# # Encrypt
# encrypted_vectors, encrypted_image = system.encrypt_image(my_image)

# # Decrypt
# decrypted_image = system.decrypt_image(encrypted_vectors)

# # Calculate quality
# psnr = calculate_psnr(my_image, decrypted_image)
# print(f"PSNR: {psnr:.4f} dB")

# METHOD 3: Save and load encryption results
# --------------------------------------------------
# # Save encrypted vectors for later
# np.save('encrypted_data.npy', encrypted_vectors)

# # Load and decrypt later
# loaded_vectors = np.load('encrypted_data.npy', allow_pickle=True)
# decrypted_image = system.decrypt_image(loaded_vectors)

# SUPPORTED IMAGE FORMATS:
# ------------------------
# - PNG, JPG, JPEG, BMP, TIFF, GIF
# - Any size (will be resized to 128×128)
# - Color images will be converted to grayscale

# NOTES:
# ------
# - Image will be automatically resized to 128×128
# - Color images are converted to grayscale
# - Secret keys: r_1,0=0.3, r_2,0=0.4, μ_1=μ_2=3.99
# - Keep these keys secret for decryption!
#     """)