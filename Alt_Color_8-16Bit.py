import numpy as np
import cv2
import os
import tifffile as tiff
import tkinter as tk
from tkinter import filedialog

# ---------------- Helper: Normalize based on bit depth ----------------
def normalize_image(img):
    """
    Returns float64 in range [0,1] and the original bit depth.
    """
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0, 8
    elif img.dtype == np.uint16:
        return img.astype(np.float64) / 65535.0, 16
    else:
        raise ValueError("Unsupported image bit depth: {}".format(img.dtype))

# ---------------- Opponent Color Transform ----------------
def to_opponent(img):
    R, G, B = img[...,0], img[...,1], img[...,2]
    O1 = (R - G) / np.sqrt(2)
    O2 = (R + G - 2*B) / np.sqrt(6)
    O3 = (R + G + B) / np.sqrt(3)   # intensity-like
    return O1, O2, O3

# ---------------- Approximate CMYK ----------------
def to_cmyk(img):
    R, G, B = img[...,0], img[...,1], img[...,2]
    K = 1 - np.max(img, axis=2)
    C = (1 - R - K) / (1 - K + 1e-8)
    M = (1 - G - K) / (1 - K + 1e-8)
    Y = (1 - B - K) / (1 - K + 1e-8)
    return C, M, Y, K

# ---------------- Save grayscale in original bit depth ----------------
def save_gray(path, arr_float, bitdepth):
    if bitdepth == 8:
        arr = np.clip(arr_float * 255, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr_float * 65535, 0, 65535).astype(np.uint16)
    tiff.imwrite(path, arr, photometric='minisblack')

# ---------------- Main ----------------
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title="Select folder of RGB TIFF images (8 or 16 bit)")
if not folder_path:
    print("No folder selected.")
    exit()

files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))])

for fname in files:
    print(f"\nProcessing: {fname}")
    im = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3:
        print("Skipping (not RGB).")
        continue

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_norm, bitdepth = normalize_image(im)

    h, w, _ = im_norm.shape
    base = os.path.splitext(fname)[0]

    # -------- Create output dir --------
    out_dir = os.path.join(folder_path, f"{base}_ColorSpaces")
    os.makedirs(out_dir, exist_ok=True)

    # -------- HSV --------
    hsv = cv2.cvtColor((im_norm * 255).astype(np.uint8), cv2.COLOR_RGB2HSV) / 255.0
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    save_gray(os.path.join(out_dir, f"{base}_HSV_H.tif"), H, bitdepth)
    save_gray(os.path.join(out_dir, f"{base}_HSV_S.tif"), S, bitdepth)
    save_gray(os.path.join(out_dir, f"{base}_HSV_V.tif"), V, bitdepth)

    # -------- HLS (HSL equivalent in OpenCV) --------
    hls = cv2.cvtColor((im_norm * 255).astype(np.uint8), cv2.COLOR_RGB2HLS) / 255.0
    H2, L2, S2 = hls[...,0], hls[...,1], hls[...,2]
    save_gray(os.path.join(out_dir, f"{base}_HSL_H.tif"), H2, bitdepth)
    save_gray(os.path.join(out_dir, f"{base}_HSL_L.tif"), L2, bitdepth)
    save_gray(os.path.join(out_dir, f"{base}_HSL_S.tif"), S2, bitdepth)

    # -------- LAB --------
    lab = cv2.cvtColor((im_norm * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L3 = lab[...,0] / 255.0
    a3 = (lab[...,1] - 128) / 255.0
    b3 = (lab[...,2] - 128) / 255.0
    save_gray(os.path.join(out_dir, f"{base}_LAB_L.tif"), L3, bitdepth)
    save_gray(os.path.join(out_dir, f"{base}_LAB_a.tif"), a3, bitdepth)
    save_gray(os.path.join(out_dir, f"{base}_LAB_b.tif"), b3, bitdepth)

    # -------- LUV --------
    luv = cv2.cvtColor((im_norm * 255).astype(np.uint8), cv2.COLOR_RGB2LUV).astype(np.float32)
    L4 = luv[...,0] / 255.0
    u4 = (luv[...,1] - 128) / 255.0
    v4 = (luv[...,2] - 128) / 255.0
    save_gray(os.path.join(out_dir, f"{base}_LUV_L.tif"), L4, bitdepth)
    save_gray(os.path.join(out_dir, f"{base}_LUV_u.tif"), u4, bitdepth)
    save_gray(os.path.join(out_dir, f"{base}_LUV_v.tif"), v4, bitdepth)

    # -------- YCrCb --------
    ycc = cv2.cvtColor((im_norm * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    Yc = ycc[...,0] / 255.0
    Cr = ycc[...,1] / 255.0
    Cb = ycc[...,2] / 255.0
    save_gray(os.path.join(out_dir, f"{base}_YCrCb_Y.tif"), Yc, bitdepth)
    save_gray(os.path.join(out_dir, f"{base}_YCrCb_Cr.tif"), Cr, bitdepth)
    save_gray(os.path.join(out_dir, f"{base}_YCrCb_Cb.tif"), Cb, bitdepth)

    # -------- Opponent Color Space (OPP) --------
    O1, O2, O3 = to_opponent(im_norm)
    save_gray(os.path.join(out_dir, f"{base}_OPP_O1.tif"), (O1+1)/2, bitdepth)
    save_gray(os.path.join(out_dir, f"{base}_OPP_O2.tif"), (O2+1)/2, bitdepth)
    save_gray(os.path.join(out_dir, f"{base}_OPP_O3.tif"), (O3+1)/2, bitdepth)

    # -------- CMYK --------
    C, M, Y, K = to_cmyk(im_norm)
    save_gray(os.path.join(out_dir, f"{base}_CMYK_C.tif"), C, bitdepth)
    save_gray(os.path.join(out_dir, f"{base}_CMYK_M.tif"), M, bitdepth)
    save_gray(os.path.join(out_dir, f"{base}_CMYK_Y.tif"), Y, bitdepth)
    save_gray(os.path.join(out_dir, f"{base}_CMYK_K.tif"), K, bitdepth)

print("\n✅ Color-space extraction complete.")
