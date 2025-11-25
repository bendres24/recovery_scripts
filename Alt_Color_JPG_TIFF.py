import numpy as np
import cv2
import os
import tifffile as tiff
import tkinter as tk
from tkinter import filedialog, simpledialog

# ---------------- Helper: Normalize based on bit depth ----------------
def normalize_image(img, ext):
    """
    Returns float64 in range [0,1] and the original bit depth.
    JPG/JPEG always forced to 8-bit.
    """
    if ext in ('.jpg', '.jpeg'):
        return img.astype(np.float64) / 255.0, 8

    # TIFF handling
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0, 8
    elif img.dtype == np.uint16:
        return img.astype(np.float64) / 65535.0, 16
    else:
        raise ValueError(f"Unsupported image bit depth: {img.dtype}")

# ---------------- Opponent Color Transform ----------------
def to_opponent(img):
    R, G, B = img[...,0], img[...,1], img[...,2]
    O1 = (R - G) / np.sqrt(2)
    O2 = (R + G - 2*B) / np.sqrt(6)
    O3 = (R + G + B) / np.sqrt(3)
    return O1, O2, O3

# ---------------- Approximate CMYK ----------------
def to_cmyk(img):
    R, G, B = img[...,0], img[...,1], img[...,2]
    K = 1 - np.max(img, axis=2)
    C = (1 - R - K) / (1 - K + 1e-8)
    M = (1 - G - K) / (1 - K + 1e-8)
    Y = (1 - B - K) / (1 - K + 1e-8)
    return C, M, Y, K

# ---------------- Save grayscale helpers ----------------
def save_gray_tif(path, arr_float, bitdepth):
    if bitdepth == 8:
        arr = np.clip(arr_float * 255, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr_float * 65535, 0, 65535).astype(np.uint16)
    tiff.imwrite(path, arr, photometric='minisblack')

def save_gray_jpg(path, arr_float):
    """
    JPG ALWAYS 8-bit.
    """
    arr = np.clip(arr_float * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, arr)

# ---------------- Main ----------------
root = tk.Tk()
root.withdraw()

folder_path = filedialog.askdirectory(
    title="Select folder of RGB TIFF or JPG images"
)
if not folder_path:
    print("No folder selected.")
    exit()

# Ask user for output format
out_format = simpledialog.askstring(
    "Output Format",
    "Save channel outputs as 'tif' or 'jpg'?",
    initialvalue="tif"
)
if out_format is None:
    print("Cancelled.")
    exit()

out_format = out_format.strip().lower()
if out_format not in ("tif", "jpg"):
    print("Invalid format. Use 'tif' or 'jpg'.")
    exit()

files = sorted([
    f for f in os.listdir(folder_path)
    if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg'))
])

# Dispatcher
def save_gray(path, arr_float, bitdepth):
    if out_format == "tif":
        save_gray_tif(path, arr_float, bitdepth)
    else:
        save_gray_jpg(path.replace(".tif", ".jpg"), arr_float)

for fname in files:
    print(f"\nProcessing: {fname}")
    ext = os.path.splitext(fname)[1].lower()
    im_path = os.path.join(folder_path, fname)

    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3:
        print("Skipping (not RGB).")
        continue

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_norm, bitdepth = normalize_image(im, ext)

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

    # -------- HLS --------
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

    # -------- Opponent Color --------
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
