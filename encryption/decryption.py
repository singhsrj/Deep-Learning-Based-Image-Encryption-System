import numpy as np
from PIL import Image

# Import all necessary functions from your project files
from logistic_map import calculate_r_and_x
from generate_weights import create_weights
from forward_pass import Substitute_Inv, Perturbation_Inv
from Deferentail_Neural_network import DifferentialNeuralNetwork

def decrypt_image(C_matrix, password):
    """
    Decrypts an image (C_matrix) using the password.
    
    Args:
        C_matrix (np.array): The encrypted image as a NumPy array.
        password (str): The secret password used for encryption.
        
    Returns:
        np.array: The decrypted image as a NumPy array.
    """
    
    print("--- Decryption Started ---")
    
    # --- 1. Key Generation ---
    # Generate all keys and parameters exactly as in encryption
    print("Step 1: Generating keys from password...")
    try:
        [x, r] = calculate_r_and_x(password)
        [c_perbutation , r_perbutation] = calculate_r_and_x(password)
        
        num_neurons = len(password)
        num_layers = 5 # 1 input + 3 hidden + 1 output
        total_weights_needed = (num_layers - 1) * (num_neurons * num_neurons)
        
        W_i = create_weights(x , total_weights_needed)
    except Exception as e:
        print(f"Error during key generation: {e}")
        print("Please ensure all helper files (logistic_map.py, generate_weights.py) are present.")
        return None

    # --- 2. Reverse DNN Blurring (C -> V) ---
    print("Step 2: Reversing Differential Neural Network...")
    
    # Initialize the DNN exactly as it was for encryption
    dnn = DifferentialNeuralNetwork(password, W_i, num_neurons=num_neurons)
    
    V_rows = []
    block_len = C_matrix.shape[1]

    for c_i in C_matrix:
        all_codes = []
        v_i_segments = []
        
        # This inner loop manually replicates the DNN's generate_and_update
        # logic to perform decryption.
        while len(all_codes) < block_len:
            # --- Lines 5-13: Feedforward Pass to get codes ---
            # (This logic is copied from DifferentialNeuralNetwork.generate_codes_and_update)
            current_layer_values = dnn.input_layer_state.astype(np.float64)
            first_hidden_layer_output = None

            for i, layer_weights in enumerate(dnn.weights):
                z = np.dot(current_layer_values, layer_weights)
                if i == 0:
                    z += dnn.bias_vector
                    first_hidden_layer_output = z
                current_layer_values = z

            current_codes = (current_layer_values.astype(np.uint64) % 256).astype(np.uint8)

            # --- DECRYPTION STEP ---
            # Get the current encrypted segment
            start_idx = len(all_codes)
            end_idx = start_idx + dnn.num_neurons
            c_segment = c_i[start_idx:end_idx]
            
            # Decrypt the segment by XORing with the generated codes
            v_segment = np.bitwise_xor(c_segment, current_codes[:len(c_segment)])
            
            # Store codes and the *decrypted* segment
            all_codes.extend(current_codes)
            v_i_segments.append(v_segment)

            # --- Line 14: Update bias vector using the *DECRYPTED* segment (v_segment) ---
            block_segment = v_segment
            if len(block_segment) < dnn.num_neurons:
                 padding = np.zeros(dnn.num_neurons - len(block_segment), dtype=np.uint8)
                 block_segment = np.concatenate((block_segment, padding))

            diff = block_segment.astype(np.int16) - current_codes.astype(np.int16)
            dnn.bias_vector = np.bitwise_xor(block_segment, diff.astype(np.uint8))
            
            # --- Line 15: Update input state (same as encryption) ---
            dnn.input_layer_state = (first_hidden_layer_output.astype(np.uint64) % 256).astype(np.uint8)
        
        # Combine all decrypted segments to form the full row
        v_i = np.concatenate(v_i_segments)
        v_i = v_i[:block_len] # Trim to exact block length
        V_rows.append(v_i)

    V_matrix = np.array(V_rows, dtype=np.uint8)
    print(f"Step 2 complete. Reconstructed V matrix with shape: {V_matrix.shape}")

    # --- 3. Reverse Second Substitution (V -> P) ---
    print("Step 3: Reversing second substitution...")
    P_rows = []
    try:
        for v_i in V_matrix:
            # Apply the inverse Substitute function
            p_i = Substitute_Inv(v_i)
            P_rows.append(p_i)
    except Exception as e:
        print(f"Error during inverse substitution (V->P): {e}")
        print("Please ensure forward_pass.py contains Substitute_Inv.")
        return None
        
    P_matrix = np.array(P_rows, dtype=np.uint8)
    print(f"Step 3 complete. Reconstructed P matrix with shape: {P_matrix.shape}")


    # --- 4. Reverse Perturbation (P -> T) ---
    print("Step 4: Reversing perturbation...")
    try:
        # Apply the inverse Perturbation function
        T_matrix = Perturbation_Inv(P_matrix, r_perbutation, c_perbutation)
    except Exception as e:
        print(f"Error during inverse perturbation (P->T): {e}")
        print("Please ensure forward_pass.py contains Perturbation_Inv.")
        return None
        
    print(f"Step 4 complete. Reconstructed T matrix with shape: {T_matrix.shape}")

    # --- 5. Reverse First Substitution (T -> Img) ---
    print("Step 5: Reversing first substitution...")
    Img_rows = []
    try:
        for t_i in T_matrix:
            # Apply the inverse Substitute function again
            img_row_i = Substitute_Inv(t_i)
            Img_rows.append(img_row_i)
    except Exception as e:
        print(f"Error during inverse substitution (T->Img): {e}")
        return None

    decrypted_image_array = np.array(Img_rows, dtype=np.uint8)
    print(f"Step 5 complete. Final decrypted image shape: {decrypted_image_array.shape}")
    
    print("--- Decryption Finished ---")
    return decrypted_image_array