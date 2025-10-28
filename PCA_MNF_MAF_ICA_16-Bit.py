import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from scipy import linalg
import matplotlib.pyplot as plt

# -------------------- PCA --------------------
def pca_transform(image_stack):
    num_bands, h, w = image_stack.shape
    X = image_stack.reshape(num_bands, -1).T
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    comps = pca.fit_transform(X_scaled)
    transformed = comps.T.reshape(num_bands, h, w)
    return transformed, pca.explained_variance_ratio_

# -------------------- MNF --------------------
def mnf_transform(image_stack):
    num_bands, h, w = image_stack.shape
    X = image_stack.reshape(num_bands, -1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.T).T

    # Estimate noise (Laplacian)
    diff_stack = np.array([cv2.Laplacian(Xs[b], cv2.CV_64F) for b in range(num_bands)])
    Cx = np.cov(Xs.reshape(num_bands, -1))
    Cn = np.cov(diff_stack.reshape(num_bands, -1))

    # Solve generalized eigenproblem
    eigvals, eigvecs = linalg.eig(Cx, Cn)
    idx = np.argsort(np.real(eigvals))[::-1]
    eigvals = np.real(eigvals[idx])
    eigvecs = np.real(eigvecs[:, idx])

    transformed = eigvecs.T @ Xs.reshape(num_bands, -1)
    transformed_stack = transformed.reshape(num_bands, h, w)
    return transformed_stack, eigvals

# -------------------- MAF --------------------
def maf_transform(image_stack, window_size=3):
    num_bands, h, w = image_stack.shape
    pixels = image_stack.reshape(num_bands, -1).T
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels)
    scaled_stack = pixels_scaled.T.reshape(num_bands, h, w)

    diff_stack = np.array([cv2.Laplacian(scaled_stack[b], cv2.CV_64F, ksize=window_size) for b in range(num_bands)])
    S = np.cov(scaled_stack.reshape(num_bands, -1))
    D = np.cov(diff_stack.reshape(num_bands, -1))

    eigvals, eigvecs = linalg.eig(D, S)
    idx = np.argsort(np.real(eigvals))
    eigvals = np.real(eigvals[idx])
    eigvecs = np.real(eigvecs[:, idx])

    transformed = eigvecs.T @ scaled_stack.reshape(num_bands, -1)
    transformed_stack = transformed.reshape(num_bands, h, w)
    return transformed_stack, eigvals

# -------------------- ICA --------------------
def ica_transform(image_stack):
    num_bands, h, w = image_stack.shape
    X = image_stack.reshape(num_bands, -1).T
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ica = FastICA(n_components=num_bands, max_iter=1000)
    comps = ica.fit_transform(X_scaled)
    transformed = comps.T.reshape(num_bands, h, w)
    return transformed

# -------------------- File Handling --------------------
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title="Select folder of 16-bit multispectral images")
if not folder_path:
    print("No folder selected. Exiting.")
    exit()

files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.tif')])
if not files:
    print("No TIFF images found.")
    exit()

images = []
for f in files:
    img = cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️ Skipping {f} (unreadable)")
        continue
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float64)
    if img.dtype == np.uint16:
        img /= 65535.0
    elif img.dtype == np.uint8:
        img /= 255.0
    images.append(img)

if not images:
    print("No valid images loaded. Exiting.")
    exit()

h, w = images[0].shape
num_bands = len(images)
print(f"Loaded {num_bands} images ({h}x{w})")

# -------------------- ROI Selection --------------------
# Properly scale float64 image for ROI display
img_disp = images[0]
disp_norm = cv2.normalize(img_disp, None, 0, 255, cv2.NORM_MINMAX)
first_img_disp = disp_norm.astype(np.uint8)
roi = cv2.selectROI("Select ROI for training", first_img_disp, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()
x, y, w_roi, h_roi = roi
if w_roi == 0 or h_roi == 0:
    print("Invalid ROI. Exiting.")
    exit()

roi_stack = np.stack([img[y:y+h_roi, x:x+w_roi] for img in images], axis=0)
full_stack = np.stack(images, axis=0)

# -------------------- RUN ALL METHODS --------------------
results = {}
eig_plots = {}

print("\nRunning PCA...")
pca_out, pca_var = pca_transform(roi_stack)
full_pca, _ = pca_transform(full_stack)
results['PCA'] = (full_pca, pca_var)
eig_plots['PCA'] = pca_var

print("Running MNF...")
mnf_out, mnf_eig = mnf_transform(roi_stack)
full_mnf, _ = mnf_transform(full_stack)
results['MNF'] = (full_mnf, mnf_eig)
eig_plots['MNF'] = mnf_eig

print("Running MAF...")
maf_out, maf_eig = maf_transform(roi_stack)
full_maf, _ = maf_transform(full_stack)
results['MAF'] = (full_maf, maf_eig)
eig_plots['MAF'] = maf_eig

print("Running ICA...")
full_ica = ica_transform(full_stack)
results['ICA'] = (full_ica, None)

# -------------------- SAVE RESULTS --------------------
for method, (stack, eig) in results.items():
    out_dir = os.path.join(folder_path, f"{method}_Results_16bit")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving {method} components to {out_dir}...")
    for i in range(num_bands):
        comp = stack[i]
        p_low, p_high = np.percentile(comp, (2, 98))
        comp = np.clip(comp, p_low, p_high)
        scaled = ((comp - p_low) / (p_high - p_low) * 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(out_dir, f"{method}_Component_{i+1:02d}.tif"), scaled)

# -------------------- EIGENVALUE COMPARISON PLOT --------------------
plt.figure(figsize=(12, 6))
for method, vals in eig_plots.items():
    plt.plot(range(1, len(vals)+1), vals / np.max(vals), 'o-', label=method)
plt.title("Comparison of Eigenvalue / Variance Curves")
plt.xlabel("Component Number")
plt.ylabel("Normalized Eigenvalue / Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(folder_path, "Eigenvalue_Comparison.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print("\n✅ Analysis complete.")
print(f"Eigenvalue comparison plot saved as: {plot_path}")
print("All results saved in their respective folders.")
