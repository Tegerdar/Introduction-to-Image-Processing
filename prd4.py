import cv2
import numpy as np
import matplotlib.pyplot as plt

def hysteresis(img, weak, strong=255):
    H, W = img.shape
    output = img.copy()
    weak_i, weak_j = np.where(img == weak)
    strong_i, strong_j = np.where(img == strong)
    for i, j in zip(weak_i, weak_j):
        neighborhood = img[max(0,i-1):min(H,i+2), max(0,j-1):min(W,j+2)]
        if np.any(neighborhood == strong):
            output[i, j] = strong
        else:
            output[i, j] = 0
    return output

def non_max_suppression(G, theta):
    H, W = G.shape
    Z = np.zeros((H, W), dtype=np.float32)
    angle = np.rad2deg(theta)
    angle[angle < 0] += 180
    for i in range(1, H-1):
        for j in range(1, W-1):
            q = 255
            r = 255
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = G[i, j+1]
                r = G[i, j-1]
            elif (22.5 <= angle[i,j] < 67.5):
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            elif (67.5 <= angle[i,j] < 112.5):
                q = G[i+1, j]
                r = G[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5):
                q = G[i-1, j-1]
                r = G[i+1, j+1]
            if (G[i,j] >= q) and (G[i,j] >= r):
                Z[i,j] = G[i,j]
            else:
                Z[i,j] = 0
    return Z

def threshold(img, ltr, htr):
    highThreshold = img.max() * htr
    lowThreshold = img.max() * ltr
    res = np.zeros(img.shape, dtype=np.uint8)
    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img < highThreshold) & (img >= lowThreshold))
    res[strong_i, strong_j] = 255
    res[weak_i, weak_j] = 50
    return res, 50, 255

def canny(image, low_thresh, high_thresh, blur_kernel=(5,5), blur_sigma=1.4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, blur_kernel, blur_sigma)
    Gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    G = np.hypot(Gx, Gy)
    G = (G / G.max()) * 255 if G.max() > 0 else G
    theta = np.arctan2(Gy, Gx)
    nms = non_max_suppression(G, theta)
    res, weak, strong = threshold(nms, low_thresh, high_thresh)
    final = hysteresis(res, weak, strong)
    return final

def roberts(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64)
    H, W = gray.shape
    Gx = np.zeros((H, W), dtype=np.float64)
    Gy = np.zeros((H, W), dtype=np.float64)
    for i in range(H-1):
        for j in range(W-1):
            a = gray[i, j]
            b = gray[i, j+1]
            c = gray[i+1, j]
            d = gray[i+1, j+1]
            Gx[i, j] = a - d
            Gy[i, j] = b - c
    G = np.sqrt(Gx**2 + Gy**2)
    G = (G / G.max()) * 255 if G.max() > 0 else G
    return G.astype(np.uint8)

image1 = cv2.imread("image1.jpg")
image1_noise = cv2.imread("image1_noise.jpg")
image2 = cv2.imread("image2.jpg")

my_image1_canny = canny(image1, low_thresh=0.055, high_thresh=0.11)
my_image1_roberts = roberts(image1)

my_image1_noise_canny = canny(image1_noise, low_thresh=0.08, high_thresh=0.16)
my_image1_noise_roberts = roberts(image1_noise)

my_image2_canny = canny(image2, low_thresh=0.08, high_thresh=0.16)
my_image2_roberts = roberts(image2)

fig = plt.figure(figsize=(20, 60))

plt.subplot(12, 1, 1)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title('Image1 - Originals attels', fontsize=20, pad=20)
plt.axis('off')

plt.subplot(12, 1, 2)
plt.imshow(my_image1_canny, cmap='gray')
plt.title('Image1 - Canny malas', fontsize=20, pad=20)
plt.axis('off')

plt.subplot(12, 1, 3)
plt.imshow(my_image1_roberts, cmap='gray')
plt.title('Image1 - Roberts operators', fontsize=20, pad=20)
plt.axis('off')

plt.subplot(12, 1, 4)
plt.imshow(cv2.cvtColor(image1_noise, cv2.COLOR_BGR2RGB))
plt.title('Image1 Noise - Originals attels', fontsize=20, pad=20)
plt.axis('off')

plt.subplot(12, 1, 5)
plt.imshow(my_image1_noise_canny, cmap='gray')
plt.title('Image1 Noise - Canny malas', fontsize=20, pad=20)
plt.axis('off')

plt.subplot(12, 1, 6)
plt.imshow(my_image1_noise_roberts, cmap='gray')
plt.title('Image1 Noise - Roberts operators', fontsize=20, pad=20)
plt.axis('off')

plt.subplot(12, 1, 7)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title('Image2 - Originals attels', fontsize=20, pad=20)
plt.axis('off')

plt.subplot(12, 1, 8)
plt.imshow(my_image2_canny, cmap='gray')
plt.title('Image2 - Canny malas', fontsize=20, pad=20)
plt.axis('off')

plt.subplot(12, 1, 9)
plt.imshow(my_image2_roberts, cmap='gray')
plt.title('Image2 - Roberts operators', fontsize=20, pad=20)
plt.axis('off')

plt.suptitle('Malas Detekcijas Salidzinajums - Canny vs Roberts', fontsize=24, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()

cv2.imwrite("my_image1_canny.jpg", my_image1_canny)
cv2.imwrite("my_image1_roberts.jpg", my_image1_roberts)
cv2.imwrite("my_image1_noise_canny.jpg", my_image1_noise_canny)
cv2.imwrite("my_image1_noise_roberts.jpg", my_image1_noise_roberts)
cv2.imwrite("my_image2_canny.jpg", my_image2_canny)
cv2.imwrite("my_image2_roberts.jpg", my_image2_roberts)
