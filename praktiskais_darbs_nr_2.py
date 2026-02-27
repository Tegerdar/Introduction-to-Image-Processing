import numpy as np
import cv2
import matplotlib.pyplot as plt

def log_correction(image):
    result = image.astype(np.float32)
    for i in range(3):
        channel = result[:, :, i]
        c = 255.0 / (np.log(1.0 + np.max(channel)))
        result[:, :, i] = c * (np.log(channel + 1.0))
    return result.astype(np.uint8)

def linear_transformation_of_histogram(image):
    result = image.astype(np.float32)
    for i in range(3):
        channel = result[:, :, i]
        min_val = np.min(channel)
        max_val = np.max(channel)
        
        if max_val > min_val:
            result[:, :, i] = (channel - min_val) / (max_val - min_val) * 255.0
        else:
            result[:, :, i] = channel
            
    return result.astype(np.uint8)

def show_image_with_histogram(image_bgr, title):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.7)
        plt.xlim([0, 256])
        
    plt.title("Histogram: " + title)
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Count")
    plt.show()

bright_image = cv2.imread('bright_image.jpg')
dark_image = cv2.imread('dark_image.jpg')
low_contrast_image = cv2.imread('low_contrast_image.jpg')

images = {
    "Bright": bright_image,
    "Dark": dark_image,
    "Low Contrast": low_contrast_image
}

for name, img in images.items():
    if img is None:
        print(f"Error: Could not load {name} image. Check file path.")
        continue
    
    show_image_with_histogram(img, f"{name} - Original")
    
    img_log = log_correction(img)
    cv2.imwrite(f"{name.split()[0].lower()}_log.jpg", img_log)
    show_image_with_histogram(img_log, f"{name} - Logarithmic")
    
    img_lin = linear_transformation_of_histogram(img)
    cv2.imwrite(f"{name.split()[0].lower()}_linear.jpg", img_lin)
    show_image_with_histogram(img_lin, f"{name} - Linear")
