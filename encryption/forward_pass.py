# %%
import numpy as np
import math

# %%
def update_df(R: float, c, d):
    """
    Update (f, d) pair safely. Prevent overflow or invalid shifts.
    """
    # Force all to int64
    f = np.int64(int(R) & 0xFFFFFFFFFFFFFFFF)
    c = np.int64(c)
    d = np.int64(d)

    z = int(c % 32)
    if z != 0:
        shift_val = int(abs(d) % z)  # ensure shift count valid
        d = np.int64((int(c) << shift_val) ^ int(d))
    else:
        d = np.int64(int(d) ^ int(c))

    # Xorshift-like pattern with masking to 64 bits
    mask = np.int64(0xFFFFFFFFFFFFFFFF)
    d = np.int64((d ^ (d << 21)) & mask)
    d = np.int64((d ^ (d >> 35)) & mask)
    d = np.int64((d ^ (d << 4)) & mask)

    return f, d


# %%
update_df(2.345647768 , 157 , 524592858)

# %%
# ---- Main: Substitute function (forward + backward) ----
def Substitute(B):
    """
    Forward + backward substitution with overflow protection.
    B: list of pixel values [b1, b2, ..., bn]
    Returns: encrypted block
    """
    n = len(B)
    out1 = []

    f = np.int64(1)
    d = np.int64(1)

    def safe_R(d, f):
        # ensure no NaN or Inf
        if not math.isfinite(float(d)): d = np.int64(1)
        if not math.isfinite(float(f)): f = np.int64(1)
        d_abs = min(abs(int(d)), 2**31 - 1)
        f_abs = min(abs(int(f)), 2**31 - 1)
        val = d_abs / (4.0 * f_abs + 1e-9)
        return 17.32 * math.sqrt(val if val >= 0 else 0.0)

    # --- Forward pass ---
    for bi in B:
        R = safe_R(d, f)
        k = int(R) % 256
        si = k ^ int(bi)
        out1.append(si)
        f, d = update_df(R, bi, d)

    # --- Backward pass ---
    out = []
    f = np.int64(1)
    d = np.int64(1)

    for si in reversed(out1):
        R = safe_R(d, f)
        k = int(R) % 256
        ci = k ^ int(si)
        out.insert(0, ci)
        f, d = update_df(R, si, d)

    return [int(x) & 0xFF for x in out]



# %%
def Substitute_Inv(out):
    """
    Robust inverse of Substitute().
    Recovers original block from encrypted block.
    """
    out = [int(x) & 0xFF for x in out]
    out1 = []

    f = np.int64(1)
    d = np.int64(1)

    def safe_R(d, f):
        if not math.isfinite(float(d)): d = np.int64(1)
        if not math.isfinite(float(f)): f = np.int64(1)
        d_abs = min(abs(int(d)), 2**31 - 1)
        f_abs = min(abs(int(f)), 2**31 - 1)
        val = d_abs / (4.0 * f_abs + 1e-9)
        return 17.32 * math.sqrt(val if val >= 0 else 0.0)

    # --- Undo backward pass ---
    for ci in reversed(out):
        R = safe_R(d, f)
        k = int(R) % 256
        si = k ^ int(ci)
        out1.insert(0, si)
        f, d = update_df(R, si, d)

    # --- Undo forward pass ---
    B = []
    f = np.int64(1)
    d = np.int64(1)

    for si in out1:
        R = safe_R(d, f)
        k = int(R) % 256
        bi = k ^ int(si)
        B.append(bi)
        f, d = update_df(R, bi, d)

    return [int(x) & 0xFF for x in B]





# %%
# --- Global seeds (will be set inside perturbation) ---
Seed_r = np.int64(0)
Seed_c = np.int64(0)

def Randomize(seed):
    """64-bit safe randomizer using xorshift pattern — avoids OverflowError."""
    # use unsigned 64-bit (np.uint64) for masking and bitwise operations
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    seed = np.uint64(seed) & mask

    seed ^= (seed << np.uint64(21)) & mask
    seed ^= (seed >> np.uint64(35)) & mask
    seed ^= (seed << np.uint64(4)) & mask

    # convert back to signed 64-bit for compatibility with np.int64 operations
    return np.int64(seed & mask)

def Update(r: int, c: int, s: int, N: int, M: int):
    """
    Update row and column positions based on pixel value s (0..255).
    Keeps everything in 64-bit range to avoid overflow.
    """
    global Seed_r, Seed_c

    # Mix pixel into seeds
    Seed_r = np.int64(Seed_r ^ np.int64(s))
    Seed_c = np.int64(Seed_c ^ np.int64((s << 3) | (s >> 5)))

    # Randomize both seeds safely (64-bit masked)
    Seed_r = Randomize(Seed_r)
    Seed_c = Randomize(Seed_c)

    # Map to valid positions
    r_new = int((int(Seed_r) % N) ^ int(r))
    c_new = int((int(Seed_c) % M) ^ int(c))

    # Ensure indices in bounds
    r_new = r_new % N
    c_new = c_new % M

    return r_new, c_new


# %%
def Perturbation(Image: np.ndarray, r_init, c_init):
    global Seed_r, Seed_c
    N, M = Image.shape  # works even if rectangular

    img = np.array(Image, copy=False)

    # ✅ Fix: use (N, M)
    Image_p = np.full((N, M), -1, dtype=int)

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
    return Image_p.astype(img.dtype)

# %%
def Perturbation_Inv(Image_p , r, c):
    """
    Restore the original image from a scrambled Image_p.
    Image_p: N x N scrambled image
    Returns: restored Image
    """
    N = Image_p.shape[0]
    M = Image_p.shape[1]
    Image = np.full((N, M), -1, dtype=int)  # initialize output

    # --- Initialize positions using same logistic map as forward ---
  

    # --- Reset global seeds ---
    global Seed_r, Seed_c
    Seed_r = np.int64(0)
    Seed_c = np.int64(0)

    # --- Main loop over pixels ---
    for i in range(N):
        for j in range(M):
            # Try to get pixel from (r, c)
            pixel = None
            if Image_p[r, c] != -1:
                pixel = Image_p[r, c]
                Image_p[r, c] = -1  # mark as taken
            else:
                # Search down the column first
                for rr in range(N):
                    if Image_p[rr, c] != -1:
                        pixel = Image_p[rr, c]
                        Image_p[rr, c] = -1
                        r = rr
                        break
                # If not found, search next columns
                if pixel is None:
                    found = False
                    for cc in range(M):
                        for rr in range(N):
                            if Image_p[rr, cc] != -1:
                                pixel = Image_p[rr, cc]
                                Image_p[rr, cc] = -1
                                r, c = rr, cc
                                found = True
                                break
                        if found:
                            break

            Image[i, j] = pixel

            # --- Update position using Update function ---
            r, c = Update(r, c, Image[i, j], N , M)

    return Image


