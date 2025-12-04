import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.filters
import os
import csv

# --------------------------- GUI & Display Setup ---------------------------
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Calibri']
plt.rcParams['axes.unicode_minus'] = False


# --------------------------- Manual Scale Bar Selection ---------------------------
def select_scale_bar_area(img, max_display_size=1000):
    """
    Manually select the scale bar area (auto resize for display).
    Returns the detected scale bar pixel length.
    """
    h, w = img.shape[:2]
    scale_factor = 1.0

    if max(h, w) > max_display_size:
        scale_factor = max_display_size / max(h, w)
        display_img = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)))
    else:
        display_img = img.copy()

    clone = display_img.copy()
    rect = []
    selecting = False
    done = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal rect, selecting, done, display_img
        temp_img = clone.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            rect = [(x, y)]
            selecting = True
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            cv2.rectangle(temp_img, rect[0], (x, y), (0, 0, 255), 2)
            cv2.imshow("Select Scale Bar", temp_img)
        elif event == cv2.EVENT_LBUTTONUP:
            rect.append((x, y))
            selecting = False
            done = True
            cv2.rectangle(temp_img, rect[0], rect[1], (0, 0, 255), 2)
            cv2.imshow("Select Scale Bar", temp_img)

    cv2.namedWindow("Select Scale Bar", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Scale Bar", mouse_callback)
    cv2.imshow("Select Scale Bar", display_img)
    print("Draw a rectangle around the scale bar, then press Enter to confirm or Esc to cancel.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and done:
            break
        elif key == 27:
            rect = []
            break

    cv2.destroyAllWindows()

    if len(rect) != 2:
        print("No scale bar region selected.")
        return None

    # Convert back to original coordinates
    x1, y1 = [int(pt / scale_factor) for pt in rect[0]]
    x2, y2 = [int(pt / scale_factor) for pt in rect[1]]
    roi = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    proj = np.sum(th, axis=0)
    indices = np.where(proj > 0.5 * np.max(proj))[0]
    if len(indices) > 0:
        scale_length_px = indices[-1] - indices[0]
        print(f"Detected scale bar pixel length: {scale_length_px} px")
        return scale_length_px
    else:
        print("No distinct scale bar line detected. Please retry.")
        return None


# --------------------------- Interactive Cropper ---------------------------
class NMSquareCropper:
    def __init__(self, img, nm_per_pixel, target_nm=10, ax=None, output_dir="."):
        self.img = img
        self.ax = ax
        self.nm_per_pixel = nm_per_pixel
        self.target_nm = target_nm
        self.target_pix = int(round(self.target_nm / self.nm_per_pixel))
        self.output_dir = output_dir
        self.roi_coords = None
        self.preview_rect = None
        self.confirmed = False
        self.h, self.w = img.shape[:2]
        self.cropped_image = None

    def on_click(self, event):
        if event.button == 1 and event.inaxes:
            x_start = int(event.xdata)
            y_start = int(event.ydata)
            S = self.target_pix
            x_end = x_start + S
            y_end = y_start + S

            if x_end > self.w or y_end > self.h or x_start < 0 or y_start < 0:
                print(f"Region ({S}×{S} px) is out of bounds.")
                return

            self.roi_coords = {'x1': x_start, 'y1': y_start, 'x2': x_end, 'y2': y_end}
            self.draw_preview_box(x_start, y_start, S)
            print(f"Selected region: ({x_start}, {y_start}) -> ({x_end}, {y_end})")

    def draw_preview_box(self, x, y, S):
        if self.preview_rect:
            self.preview_rect.remove()
        self.preview_rect = plt.Rectangle((x, y), S, S, fill=False,
                                          edgecolor='red', linewidth=2.5, linestyle='--')
        self.ax.add_patch(self.preview_rect)
        self.ax.figure.canvas.draw()

    def on_key_press(self, event):
        if event.key in ['enter', ' ']:
            if self.roi_coords:
                print("Region confirmed.")
                self.confirmed = True
                plt.close(event.canvas.figure)
        elif event.key in ['escape', 'q']:
            print("Operation cancelled.")
            self.confirmed = False
            plt.close(event.canvas.figure)

    def execute_crop(self):
        if self.confirmed and self.roi_coords:
            x1, y1, x2, y2 = [self.roi_coords[k] for k in ['x1', 'y1', 'x2', 'y2']]
            cropped_image = self.img[y1:y2, x1:x2]
            outname = os.path.join(self.output_dir, f"{self.target_nm}nm_crop_{x1}_{y1}.png")
            cv2.imwrite(outname, cropped_image)
            print(f"Cropped image saved: {outname}")
            return cropped_image
        else:
            print("No crop executed.")
            return None


# --------------------------- New Orderliness Calculation (Structure Tensor) ---------------------------
def calculate_coherence_map(image, window_size=16, stride=8, sigma=1.0):
    """
   coherence ∈ [0,1]
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    Ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    
    J11 = cv2.GaussianBlur(Ix * Ix, (0, 0), sigma)
    J22 = cv2.GaussianBlur(Iy * Iy, (0, 0), sigma)
    J12 = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma)

    img_h, img_w = image.shape
    map_h = (img_h - window_size) // stride + 1
    map_w = (img_w - window_size) // stride + 1
    order_map = np.zeros((map_h, map_w), dtype=np.float32)

    for y in range(map_h):
        for x in range(map_w):
            ys, xs = y * stride, x * stride
            ye, xe = ys + window_size, xs + window_size

            a = J11[ys:ye, xs:xe].mean()
            b = J22[ys:ye, xs:xe].mean()
            c = J12[ys:ye, xs:xe].mean()

            trace = a + b
            diff = a - b
            lam = np.sqrt(diff * diff + 4.0 * c * c)
            lam1 = (trace + lam) / 2.0
            lam2 = (trace - lam) / 2.0

            
            coh = (lam1 - lam2) / (lam1 + lam2 + 1e-6)
            order_map[y, x] = coh

    return order_map, image.shape


def extract_order_features(order_map):
    """
    Extract several global statistical features from the coherence map
    """
    vals = order_map.flatten()
    mean_R = float(np.mean(vals))
    std_R  = float(np.std(vals))
    R_90    = float(np.quantile(vals, 0.90))
    max_R  = float(np.max(vals))

    return {
        "mean_R": mean_R,
        "std_R": std_R,
        "90_R": R_90,
        "max_R": max_R,
    }



def classify_order(features):
   
    R_90 = features["90_R"]

    if R_90 > 0.70:
        label = "highly_ordered"      
    elif R_90 > 0.40:
        label = "partially_ordered"   
    else:
        label = "disordered"          

    return label



# --------------------------- Visualization & Saving ---------------------------
def visualize_and_analyze(cropped_img, target_nm, output_dir="TEM_Analysis_Results"):
    os.makedirs(output_dir, exist_ok=True)

   
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) if len(cropped_img.shape) == 3 else cropped_img
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    
    order_map, img_shape = calculate_coherence_map(
        gray_blur,
        window_size=32,   
        stride=16,
        sigma=1
    )

    
    flat_vals = order_map.flatten()
    counts, bins = np.histogram(flat_vals, bins=30)

    
    features = extract_order_features(order_map)
    label = classify_order(features)

    
    csv_path = os.path.join(output_dir, f"{target_nm}nm_Orderliness_Analysis.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

       
        writer.writerow(["Coherence", "Frequency"])   
        for r, freq in zip(bins[:-1], counts):
            writer.writerow([f"{r:.6f}", freq])

       
        writer.writerow([])

        
        writer.writerow(["Feature", "Value"])
        writer.writerow(["Class", label])
        for k, v in features.items():
            writer.writerow([k, f"{v:.6f}"])

    print(f"CSV (histogram + features) saved: {csv_path}")
    print("Estimated class:", label, "features:", features)

    
    plt.figure(figsize=(6, 4))
    plt.hist(flat_vals, bins=30, color='steelblue', edgecolor='black')
    plt.xlabel("Coherence")
    plt.ylabel("Frequency")
    plt.title(f"Coherence Distribution ({target_nm} nm region)")
    hist_path = os.path.join(output_dir, f"{target_nm}nm_Orderliness_Histogram.png")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Histogram saved: {hist_path}")

    
    heatmap_resized = cv2.resize(order_map, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
    plt.imshow(gray, cmap='gray', alpha=0.8)
    plt.imshow(heatmap_resized, cmap='hot', alpha=0.5)
    plt.axis('off')
    heatmap_path = os.path.join(output_dir, f"{target_nm}nm_Orderliness_Heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: {heatmap_path}")



# --------------------------- Main ---------------------------
if __name__ == "__main__":
    
    IMG_PATH = r"path\to\original_TEM_image"  # Path to the original TEM image

    SCALE_NM = 10
    TARGET_SIZE_NM = 10
    
    OUTPUT_DIR = r"path\to\output_folder" # Directory to save all output results

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")

    detected_px = select_scale_bar_area(img)
    if detected_px is None:
        raise RuntimeError("Scale bar selection failed.")

    nm_per_px = SCALE_NM / detected_px
    print(f"\n Calibration: {SCALE_NM} nm = {detected_px} px → 1 px = {nm_per_px:.6f} nm")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Click to select top-left corner ({TARGET_SIZE_NM} nm region)")
    cropper = NMSquareCropper(img, nm_per_pixel=nm_per_px,
                              target_nm=TARGET_SIZE_NM, ax=ax, output_dir=OUTPUT_DIR)
    fig.canvas.mpl_connect('button_press_event', cropper.on_click)
    fig.canvas.mpl_connect('key_press_event', cropper.on_key_press)
    plt.show()

    cropped = cropper.execute_crop()
    if cropped is not None:
        visualize_and_analyze(cropped, TARGET_SIZE_NM, OUTPUT_DIR)
