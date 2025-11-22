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

    diff_stack = np.array([cv2.Laplacian(Xs[b], cv2.CV_64F) for b in range(num_bands)])
    Cx = np.cov(Xs.reshape(num_bands, -1))
    Cn = np.cov(diff_stack.reshape(num_bands, -1))

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
folder_path = filedialog.askdirectory(title="Select folder of 8-bit RGB TIFF images")
if not folder_path:
    print("No folder selected. Exiting.")
    exit()

files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.tif')])
if not files:
    print("No TIFF images found.")
    exit()

for f in files:
    img = cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3:
        print(f"Skipping {f} (not RGB)")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    h, w, _ = img.shape
    image_stack = img.transpose(2, 0, 1)  # (3, h, w)

    # ROI selection
    gray_disp = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    roi = cv2.selectROI("Select ROI", gray_disp, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w_roi, h_roi = roi
    if w_roi == 0 or h_roi == 0:
        print("Invalid ROI. Exiting.")
        exit()

    roi_stack = image_stack[:, y:y+h_roi, x:x+w_roi]

    results = {}
    eig_plots = {}

    print(f"\nRunning PCA on {f}...")
    pca_out, pca_var = pca_transform(roi_stack)
    full_pca, _ = pca_transform(image_stack)
    results['PCA'] = (full_pca, pca_var)
    eig_plots['PCA'] = pca_var

    print("Running MNF...")
    mnf_out, mnf_eig = mnf_transform(roi_stack)
    full_mnf, _ = mnf_transform(image_stack)
    results['MNF'] = (full_mnf, mnf_eig)
    eig_plots['MNF'] = mnf_eig

    print("Running MAF...")
    maf_out, maf_eig = maf_transform(roi_stack)
    full_maf, _ = maf_transform(image_stack)
    results['MAF'] = (full_maf, maf_eig)
    eig_plots['MAF'] = maf_eig

    print("Running ICA...")
    full_ica = ica_transform(image_stack)
    results['ICA'] = (full_ica, None)

    for method, (stack, eig) in results.items():
        out_dir = os.path.join(folder_path, f"{method}_Results_8bit_RGB")
        os.makedirs(out_dir, exist_ok=True)
        for i in range(stack.shape[0]):
            comp = stack[i]
            p_low, p_high = np.percentile(comp, (2, 98))
            comp = np.clip(comp, p_low, p_high)
            scaled = ((comp - p_low) / (p_high - p_low) * 255).astype(np.uint8)
             cv2.imwrite(
                os.path.join(out_dir, f"{os.path.splitext(f)[0]}_{method}_Comp{i+1}.tif"),
                scaled
            )

    plt.figure(figsize=(8, 5))
    for method, vals in eig_plots.items():
        plt.plot(range(1, len(vals)+1), vals / np.max(vals), 'o-', label=method)
    plt.title(f"Eigenvalue / Variance Curves — {f}")
    plt.xlabel("Component")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f"{os.path.splitext(f)[0]}_Eigen_8bit_RGB.png"), dpi=300)
    plt.close()

print("\n✅ 8-bit RGB analysis complete.")
