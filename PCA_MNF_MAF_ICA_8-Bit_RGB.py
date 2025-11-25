#!/usr/bin/env python3
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.decomposition import PCA, FastICA
from scipy import linalg
import matplotlib.pyplot as plt

# ---------- Utilities ----------
def clean_and_normalize_img(img):
    # Convert BGR -> RGB, remove NaNs/Infs, clamp to valid range, normalize to [0,1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    img[~np.isfinite(img)] = 0.0
    img = np.clip(img, 0.0, 65535.0)
    maxval = img.max()
    if maxval > 1.0:
        denom = 65535.0 if maxval > 255.0 else 255.0
        img = img / denom
    img = np.clip(img, 0.0, 1.0)
    return img

def safer_standardize(X, eps=1e-4):
    # X: (n_samples, n_features)
    mean = X.mean(axis=0)
    std = np.clip(X.std(axis=0), eps, None)
    Xs = (X - mean) / std
    return Xs, mean, std

def percentile_scale_to_uint8(comp, low=2, high=98):
    p_low, p_high = np.percentile(comp, (low, high))
    if p_high - p_low <= 0:
        return np.zeros_like(comp, dtype=np.uint8)
    clipped = np.clip(comp, p_low, p_high)
    scaled = ((clipped - p_low) / (p_high - p_low) * 255.0)
    return np.round(scaled).astype(np.uint8)

# ---------- ROI-fit / full-apply wrappers ----------
def pca_fit_on_roi(roi_stack):
    b, h, w = roi_stack.shape
    X = roi_stack.reshape(b, -1).T
    Xs, mean, std_safe = safer_standardize(X)
    pca = PCA()
    pca.fit(Xs)
    return pca, mean, std_safe

def pca_apply(full_stack, pca_model, mean, std_safe):
    b, h, w = full_stack.shape
    X = full_stack.reshape(b, -1).T
    Xs = (X - mean) / std_safe
    comps = pca_model.transform(Xs)
    return comps.T.reshape(comps.shape[1], h, w)

def mnf_fit_on_roi(roi_stack, lap_kernel=cv2.CV_64F):
    b, h, w = roi_stack.shape
    X = roi_stack.reshape(b, -1).T
    Xs, mean, std_safe = safer_standardize(X)
    Xs = Xs.T
    diff_stack = np.array([cv2.Laplacian(roi_stack[i], lap_kernel) for i in range(b)])
    Cx = np.cov(Xs)
    Cn = np.cov(diff_stack.reshape(b, -1))
    eigvals, eigvecs = linalg.eig(Cx, Cn)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvecs, eigvals, mean, std_safe

def mnf_apply(full_stack, eigvecs, mean, std_safe):
    b, h, w = full_stack.shape
    X = full_stack.reshape(b, -1).T
    Xs = (X - mean) / std_safe
    Xs = Xs.T
    comps = eigvecs.T @ Xs
    return comps.reshape(comps.shape[0], h, w)

def maf_fit_on_roi(roi_stack, window_size=3):
    b, h, w = roi_stack.shape
    pixels = roi_stack.reshape(b, -1).T
    Xs, mean, std_safe = safer_standardize(pixels)
    scaled_stack = Xs.T.reshape(b, h, w)
    diff_stack = np.array([cv2.Laplacian(scaled_stack[i], cv2.CV_64F, ksize=window_size) for i in range(b)])
    S = np.cov(scaled_stack.reshape(b, -1))
    D = np.cov(diff_stack.reshape(b, -1))
    eigvals, eigvecs = linalg.eig(D, S)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvecs, eigvals, mean, std_safe

def maf_apply(full_stack, eigvecs, mean, std_safe):
    b, h, w = full_stack.shape
    X = full_stack.reshape(b, -1).T
    Xs = (X - mean) / std_safe
    Xs = Xs.T
    comps = eigvecs.T @ Xs
    return comps.reshape(comps.shape[0], h, w)

def ica_fit_on_roi(roi_stack, max_iter=1000):
    b, h, w = roi_stack.shape
    X = roi_stack.reshape(b, -1).T
    Xs, mean, std_safe = safer_standardize(X)
    ica = FastICA(n_components=b, max_iter=max_iter, random_state=0)
    ica.fit(Xs)
    return ica, mean, std_safe

def ica_apply(full_stack, ica_model, mean, std_safe):
    b, h, w = full_stack.shape
    X = full_stack.reshape(b, -1).T
    Xs = (X - mean) / std_safe
    comps = ica_model.transform(Xs)
    transformed = comps.T.reshape(comps.shape[1], h, w)
    if transformed.shape[0] < b:
        out = np.zeros((b, h, w), dtype=np.float64)
        out[:transformed.shape[0]] = transformed
        return out
    return transformed

# ---------- Main Loop ----------
def main():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select folder of 8-bit RGB TIFF images")
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.tif')])
    if not files:
        print("No TIFF images found.")
        return

    for f in files:
        print(f"\nProcessing {f} ...")
        img = cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3:
            print(f"Skipping {f} (not RGB or unreadable).")
            continue
        img = clean_and_normalize_img(img)
        h, w, _ = img.shape
        image_stack = img.transpose(2, 0, 1)  # (bands, h, w)

        # ROI selection
        gray_disp = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        roi = cv2.selectROI("Select ROI", gray_disp, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        x, y, w_roi, h_roi = roi
        if w_roi == 0 or h_roi == 0:
            print("Invalid ROI. Skipping file.")
            continue
        roi_stack = image_stack[:, y:y+h_roi, x:x+w_roi]

        results = {}
        eig_plots = {}

        # PCA
        print("Fitting PCA on ROI...")
        pca_model, pca_mean, pca_std = pca_fit_on_roi(roi_stack)
        print("Applying PCA to full image...")
        pca_full = pca_apply(image_stack, pca_model, pca_mean, pca_std)
        results['PCA'] = (pca_full, pca_model.explained_variance_ratio_)
        eig_plots['PCA'] = pca_model.explained_variance_ratio_

        # MNF
        print("Fitting MNF on ROI...")
        mnf_eigvecs, mnf_eigvals, mnf_mean, mnf_std = mnf_fit_on_roi(roi_stack)
        print("Applying MNF to full image...")
        mnf_full = mnf_apply(image_stack, mnf_eigvecs, mnf_mean, mnf_std)
        results['MNF'] = (mnf_full, mnf_eigvals)
        eig_plots['MNF'] = mnf_eigvals

        # MAF
        print("Fitting MAF on ROI...")
        maf_eigvecs, maf_eigvals, maf_mean, maf_std = maf_fit_on_roi(roi_stack)
        print("Applying MAF to full image...")
        maf_full = maf_apply(image_stack, maf_eigvecs, maf_mean, maf_std)
        results['MAF'] = (maf_full, maf_eigvals)
        eig_plots['MAF'] = maf_eigvals

        # ICA
        print("Fitting ICA on ROI...")
        ica_model, ica_mean, ica_std = ica_fit_on_roi(roi_stack)
        print("Applying ICA to full image...")
        ica_full = ica_apply(image_stack, ica_model, ica_mean, ica_std)
        results['ICA'] = (ica_full, None)

        # Save components
        for method, (stack, eig) in results.items():
            out_dir = os.path.join(folder_path, f"{method}_Results_8bit_Gray")
            os.makedirs(out_dir, exist_ok=True)
            for i in range(stack.shape[0]):
                out_im = percentile_scale_to_uint8(stack[i])
                out_name = os.path.join(out_dir, f"{os.path.splitext(f)[0]}_{method}_Comp{i+1}.tif")
                cv2.imwrite(out_name, out_im)

        # Eigen/variance plot
        plt.figure(figsize=(8,5))
        for method, vals in eig_plots.items():
            if vals is None or len(vals)==0:
                continue
            vals = np.asarray(vals, dtype=np.float64)
            m = vals.max()
            plot_vals = vals/m if m>0 else vals
            plt.plot(range(1,len(plot_vals)+1), plot_vals,'o-',label=method)
        plt.title(f"Eigenvalue / Variance Curves — {f}")
        plt.xlabel("Component")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, f"{os.path.splitext(f)[0]}_Eigen_8bit_RGB.png"), dpi=300)
        plt.close()

    print("\n✅ 8-bit RGB analysis complete.")

if __name__ == "__main__":
    main()
