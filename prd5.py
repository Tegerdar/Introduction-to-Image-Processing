import cv2
import numpy as np
from collections import deque
import os
from google.colab.patches import cv2_imshow

INTENSITY_THRESHOLD = 15
GRID_SPACING        = 30
OUTPUT_DIR          = "output"
TEST_IMAGES = [
    "img1.png",
    "img2.png",
    "img3.png",
]

def generate_seeds(height: int, width: int, spacing: int) -> list[tuple[int, int]]:
    seeds = []
    for r in range(spacing // 2, height, spacing):
        for c in range(spacing // 2, width, spacing):
            seeds.append((r, c))
    return seeds

def region_growing(gray: np.ndarray, seeds: list[tuple[int, int]], threshold: int) -> np.ndarray:
    h, w = gray.shape
    label_map = np.zeros((h, w), dtype=np.int32)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    region_id = 0
    for seed_r, seed_c in seeds:
        if label_map[seed_r, seed_c] != 0:
            continue

        region_id += 1
        seed_intensity = int(gray[seed_r, seed_c])
        queue = deque()
        queue.append((seed_r, seed_c))
        label_map[seed_r, seed_c] = region_id

        while queue:
            r, c = queue.popleft()
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and label_map[nr, nc] == 0:
                    if abs(int(gray[nr, nc]) - seed_intensity) < threshold:
                        label_map[nr, nc] = region_id
                        queue.append((nr, nc))

    return label_map

def process_image(image_path: str) -> None:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[SKIP] Cannot read '{image_path}'")
        return

    print(f"[INFO] Processing '{image_path}'  "
          f"size={img_bgr.shape[1]}x{img_bgr.shape[0]}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    seeds     = generate_seeds(h, w, GRID_SPACING)
    label_map = region_growing(gray, seeds, INTENSITY_THRESHOLD)
    segmented = np.where(label_map > 0, 255, 0).astype(np.uint8)

    thresh_mean = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        199, 5
    )

    segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
    thresh_mean_bgr = cv2.cvtColor(thresh_mean, cv2.COLOR_GRAY2BGR)

    num_regions = label_map.max()
    print(f"[INFO] Regions found: {num_regions}")

    side_by_side = np.hstack([img_bgr, segmented_bgr, thresh_mean_bgr])
    side_by_side[:, w] = (255, 255, 255)
    side_by_side[:, 2 * w] = (255, 255, 255)

    window_title = f"Original | Segmented | AdaptiveThresh — {os.path.basename(image_path)}"
    cv2_imshow(side_by_side)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name   = os.path.splitext(os.path.basename(image_path))[0]
    out_seg     = os.path.join(OUTPUT_DIR, f"{base_name}_segmented.png")
    out_sbs     = os.path.join(OUTPUT_DIR, f"{base_name}_comparison.png")
    cv2.imwrite(out_seg, segmented)
    cv2.imwrite(out_sbs, side_by_side)
    print(f"[INFO] Saved → {out_seg}")
    print(f"[INFO] Saved → {out_sbs}")

def main() -> None:
    found = [p for p in TEST_IMAGES if os.path.isfile(p)]
    if not found:
        print("[ERROR] None of the listed test images were found.")
        print("        Place image files next to this script and update TEST_IMAGES.")
        return

    for path in found:
        process_image(path)

if __name__ == "__main__":
    main()
