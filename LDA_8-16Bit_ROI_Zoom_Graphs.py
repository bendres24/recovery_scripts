import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog, simpledialog
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------- LDA TRANSFORM --------------------
def lda_transform(image_stack, mask_stack, num_classes):
    num_bands, h, w = image_stack.shape
    X = image_stack.reshape(num_bands, -1).T
    y = mask_stack.flatten()

    valid = y >= 0
    X = X[valid]
    y = y[valid]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lda = LinearDiscriminantAnalysis(n_components=min(num_classes - 1, num_bands))
    comps = lda.fit_transform(X_scaled, y)

    X_full = image_stack.reshape(num_bands, -1).T
    X_full_scaled = scaler.transform(X_full)
    transformed = lda.transform(X_full_scaled)

    num_comps = transformed.shape[1]
    transformed_stack = np.zeros((num_comps, h, w))
    for i in range(num_comps):
        transformed_stack[i] = transformed[:, i].reshape(h, w)

    return transformed_stack, getattr(lda, 'explained_variance_ratio_', None), lda, scaler

# -------------------- CUSTOM ZOOM/PAN ROI SELECTOR --------------------
def select_roi_zoomable(window_name, image):
    clone = image.copy()
    zoom = 1.0
    pan_x, pan_y = 0, 0
    roi = None
    selecting = False
    start_pt, end_pt = None, None
    h, w = image.shape[:2]

    def update_display():
        nonlocal zoom, pan_x, pan_y
        zoomed = cv2.resize(clone, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
        zh, zw = zoomed.shape[:2]
        max_pan_x = max(0, zw - w)
        max_pan_y = max(0, zh - h)
        pan_x_clamped = int(np.clip(pan_x, 0, max_pan_x))
        pan_y_clamped = int(np.clip(pan_y, 0, max_pan_y))
        view = zoomed[pan_y_clamped:pan_y_clamped + h, pan_x_clamped:pan_x_clamped + w]
        return view

    def draw_roi(img, pt1, pt2):
        overlay = img.copy()
        cv2.rectangle(overlay, pt1, pt2, (0, 255, 0), 2)
        return overlay

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback := lambda *a: None)

    def mouse_callback(event, x, y, flags, param):
        nonlocal start_pt, end_pt, selecting, roi
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            start_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            end_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            end_pt = (x, y)
            x1, x2 = sorted([start_pt[0], end_pt[0]])
            y1, y2 = sorted([start_pt[1], end_pt[1]])
            roi = (x1, y1, x2 - x1, y2 - y1)

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        view = update_display()
        if selecting and start_pt and end_pt:
            view = draw_roi(view, start_pt, end_pt)

        cv2.imshow(window_name, view)
        key = cv2.waitKeyEx(10)

        if key == ord('='):          # zoom in
            zoom = min(zoom * 1.2, 20.0)
        elif key == ord('-'):        # zoom out
            zoom = max(zoom / 1.2, 1.0)
        elif key in [ord('a'), ord('A')]:  # pan left
            pan_x -= 50
        elif key in [ord('d'), ord('D')]:  # pan right
            pan_x += 50
        elif key in [ord('w'), ord('W')]:  # pan up
            pan_y -= 50
        elif key in [ord('s'), ord('S')]:  # pan down
            pan_y += 50
        elif key == 13:              # Enter
            break
        elif key == 27:              # Esc
            roi = None
            break

    cv2.destroyWindow(window_name)
    return roi


# -------------------- FILE HANDLING --------------------
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title="Select folder of 8- or 16-bit multispectral images")
if not folder_path:
    print("No folder selected. Exiting.")
    exit()

files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.tif')])
if not files:
    print("No TIFF images found. Exiting.")
    exit()

images = []
display_images = []

for f in files:
    img_path = os.path.join(folder_path, f)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️ Skipping {f}")
        continue

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize to [0,1]
    if img.dtype == np.uint16:
        img_float = img.astype(np.float64) / 65535.0
    elif img.dtype == np.uint8:
        img_float = img.astype(np.float64) / 255.0
    else:
        img_float = np.clip(img.astype(np.float64), 0, 1)

    images.append(img_float)

    # Display version (8-bit)
    disp = cv2.normalize(img_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    display_images.append(disp)

if not images:
    print("No valid images loaded. Exiting.")
    exit()

h, w = images[0].shape
num_bands = len(images)
print(f"Loaded {num_bands} images ({h}x{w})")

image_stack = np.stack(images, axis=0)
display_stack = np.stack(display_images, axis=0)

# -------------------- MULTICLASS ROI SELECTION --------------------
mask = np.full((h, w), -1, dtype=np.int32)
class_labels = []
class_index = 0

while True:
    class_name = simpledialog.askstring(
        "LDA Class Input",
        f"Enter name for class #{class_index + 1} (e.g., 'damage', 'parchment', 'text'):\n"
        "(Press Enter with no text to finish)"
    )
    if not class_name:
        break

    print(f"\nSelect ROI for class '{class_name}':")
    print("Controls: = zoom in, - zoom out, arrow keys to pan, Enter to confirm, Esc to cancel")
    roi = select_roi_zoomable(f"ROI for {class_name}", display_stack[0])

    if roi is None or roi[2] == 0 or roi[3] == 0:
        print("Invalid ROI, skipping this class.")
        continue

    x, y, w_roi, h_roi = roi
    mask[y:y+h_roi, x:x+w_roi] = class_index
    class_labels.append(class_name)
    class_index += 1

if class_index < 2:
    print("Need at least two valid classes. Exiting.")
    exit()

print(f"\nClasses defined: {class_labels}")

# -------------------- RUN LDA --------------------
print("\nRunning LDA with multiple classes...")
lda_stack, lda_var, lda, scaler = lda_transform(image_stack, mask, num_classes=class_index)

# -------------------- SAVE RESULTS --------------------
out_dir = os.path.join(folder_path, "LDA_Results")
os.makedirs(out_dir, exist_ok=True)

num_comps = lda_stack.shape[0]
for i in range(num_comps):
    comp = lda_stack[i]
    p_low, p_high = np.percentile(comp, (2, 98))
    comp = np.clip(comp, p_low, p_high)
    scaled = ((comp - p_low) / (p_high - p_low) * 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(out_dir, f"LDA_Component_{i+1:02d}.tif"), scaled)

# -------------------- LDA VISUALIZATIONS --------------------
if lda_var is not None:
    # --- 1. Variance Plot with Pairwise Class Labels ---
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(lda_var) + 1), lda_var / np.max(lda_var), 'o-', label='LDA Variance')

    pair_labels = []
    for i in range(len(class_labels)):
        for j in range(i + 1, len(class_labels)):
            pair_labels.append(f"{class_labels[i]} vs {class_labels[j]}")

    if len(pair_labels) < len(lda_var):
        pair_labels += [f"Comp {i+1}" for i in range(len(pair_labels), len(lda_var))]

    pair_labels = pair_labels[:len(lda_var)]
    plt.xticks(range(1, len(lda_var) + 1), pair_labels, rotation=30, ha='right')

    plt.title("LDA Explained Variance Ratio by Class Comparison")
    plt.xlabel("Discriminant Component")
    plt.ylabel("Normalized Variance")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "LDA_Variance.png"), dpi=300)
    plt.close()

    # --- 2. Component Loading Plot ---
    loadings = lda.scalings_  # weights per band
    plt.figure(figsize=(8, 4))
    for i in range(loadings.shape[1]):
        plt.plot(loadings[:, i], marker='o', label=f'LDA{i+1}')
    plt.xticks(range(num_bands), files, rotation=45, ha='right')
    plt.title("Spectral Band Contributions to LDA Components")
    plt.xlabel("Spectral Band (Image)")
    plt.ylabel("Weight (Loading)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "LDA_Loadings.png"), dpi=300)
    plt.close()

    # --- 3. Class Separation Scatter Plot ---
    valid = mask.flatten() >= 0
    X = image_stack.reshape(num_bands, -1).T[valid]
    y_valid = mask.flatten()[valid]
    X_scaled = scaler.transform(X)
    X_lda = lda.transform(X_scaled)
    plt.figure(figsize=(6, 5))
    if X_lda.shape[1] >= 2:
        for i, name in enumerate(class_labels):
            plt.scatter(X_lda[y_valid == i, 0], X_lda[y_valid == i, 1], label=name, alpha=0.6, s=10)
        plt.xlabel("LDA1")
        plt.ylabel("LDA2")
    else:
        for i, name in enumerate(class_labels):
            plt.scatter(X_lda[y_valid == i, 0], np.zeros_like(X_lda[y_valid == i, 0]), label=name, alpha=0.6, s=10)
        plt.xlabel("LDA1")
        plt.ylabel("Value")
    plt.title("Class Separation in LDA Space")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "LDA_Class_Separation.png"), dpi=300)
    plt.close()

    # --- 4. RGB Composite of First Three LDA Components ---
    if lda_stack.shape[0] >= 3:
        rgb = np.stack([
            cv2.normalize(lda_stack[0], None, 0, 1, cv2.NORM_MINMAX),
            cv2.normalize(lda_stack[1], None, 0, 1, cv2.NORM_MINMAX),
            cv2.normalize(lda_stack[2], None, 0, 1, cv2.NORM_MINMAX)
        ], axis=-1)
        cv2.imwrite(os.path.join(out_dir, "LDA_RGB_Composite.tif"), (rgb * 255).astype(np.uint8))

print("\n✅ LDA analysis complete.")
print(f"Classes used: {class_labels}")
print(f"Results and plots saved in: {out_dir}")

