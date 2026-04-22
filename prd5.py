import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from skimage.segmentation import slic

T = 199
C = 5

def load_image(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return rgb, gray

def get_seeds_slic(rgb_image, n_segments=200, compactness=10):
    segments = slic(rgb_image, n_segments=n_segments, compactness=compactness, start_label=0)
    seeds = []
    for seg_id in np.unique(segments):
        coords = np.argwhere(segments == seg_id)
        centroid = coords.mean(axis=0).astype(int)
        if segments[centroid[0], centroid[1]] != seg_id:
            centroid = coords[len(coords) // 2]
        seeds.append(tuple(centroid))
    return seeds

def mean_adaptive_thresholding(gray_img, block_size, constant):
    return cv2.adaptiveThreshold(
        gray_img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size, constant
    )

def region_grow(gray_image, seeds, threshold=10):
    h, w = gray_image.shape
    visited = np.zeros((h, w), dtype=bool)
    output  = np.zeros((h, w), dtype=np.int32)

    for label, seed in enumerate(seeds, start=1):
        if visited[seed]:
            continue

        visited[seed] = True
        output[seed]  = label
        queue         = deque([seed])
        region_sum    = int(gray_image[seed])
        region_count  = 1

        while queue:
            x, y = queue.popleft()
            region_mean = region_sum / region_count

            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                    visited[nx, ny] = True
                    if abs(int(gray_image[nx, ny]) - region_mean) < threshold:
                        output[nx, ny] = label
                        queue.append((nx, ny))
                        region_sum   += int(gray_image[nx, ny])
                        region_count += 1

    return output

def show_image(img, title, cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

RGB_img1, gray_img1 = load_image('img1.png')
RGB_img2, gray_img2 = load_image('img2.png')
RGB_img3, gray_img3 = load_image('img3.png')

SEEDS_IMG1 = get_seeds_slic(RGB_img1)
SEEDS_IMG2 = get_seeds_slic(RGB_img2)
SEEDS_IMG3 = get_seeds_slic(RGB_img3)

thresh_img1 = mean_adaptive_thresholding(gray_img1, T, C)
thresh_img2 = mean_adaptive_thresholding(gray_img2, T, C)
thresh_img3 = mean_adaptive_thresholding(gray_img3, T, C)

mask_img1 = (region_grow(gray_img1, SEEDS_IMG1, threshold=10) > 0).astype(np.uint8) * 255
mask_img2 = (region_grow(gray_img2, SEEDS_IMG2, threshold=10) > 0).astype(np.uint8) * 255
mask_img3 = (region_grow(gray_img3, SEEDS_IMG3, threshold=10) > 0).astype(np.uint8) * 255

for rgb, thresh, mask, n in zip(
    [RGB_img1, RGB_img2, RGB_img3],
    [thresh_img1, thresh_img2, thresh_img3],
    [mask_img1, mask_img2, mask_img3],
    [1, 2, 3]
):
    show_image(rgb,   f'image {n}')
    show_image(thresh, 'mean adaptive thresholding')
    show_image(mask,   'region growing')
