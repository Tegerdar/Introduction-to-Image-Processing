from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE = Path("/content")
JPEG_QUALITY = 50
NOISE_STD = 25
FILTER_KERNEL = (3, 3)


def load_images():
    images = {
        "attēls1": cv2.imread(str(BASE / "image1.webp")),
        "attēls2": cv2.imread(str(BASE / "image2.webp")),
    }
    if any(img is None for img in images.values()):
        raise FileNotFoundError("image1.webp or image2.webp not found in /content/")
    return images


def add_gaussian_noise(image, std=NOISE_STD):
    noise = np.random.normal(0, std, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def jpeg_degrade(image, quality=JPEG_QUALITY):
    _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_filters(image):
    return {
        "vidējais filtrs":  cv2.blur(image, FILTER_KERNEL),
        "mediānas filtrs":  cv2.medianBlur(image, 3),
        "gausa filtrs":     cv2.GaussianBlur(image, FILTER_KERNEL, 0),
    }


def show(images: dict, suptitle: str):
    for label, img in images.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(label)
        ax.axis("off")
        fig.suptitle(suptitle)
        plt.tight_layout()
        plt.show()


def process(name, image):
    noisy    = add_gaussian_noise(image)
    degraded = jpeg_degrade(image)

    cv2.imwrite(str(BASE / f"{name}_originalais.png"), image)
    cv2.imwrite(str(BASE / f"{name}_gausa_troksnis.png"), noisy)
    cv2.imwrite(str(BASE / f"{name}_jpeg_artifakti.jpg"), degraded)

    for fname, filtered in apply_filters(noisy).items():
        cv2.imwrite(str(BASE / f"{name}_gausa_{fname}.png"), filtered)

    for fname, filtered in apply_filters(degraded).items():
        cv2.imwrite(str(BASE / f"{name}_jpeg_{fname}.jpg"), filtered)

    show({"oriģināls": image, "gausa troksnis": noisy, "JPEG artifakti": degraded}, f"{name}: pārskats")
    show({"gausa troksnis": noisy,    **apply_filters(noisy)},    f"{name}: filteri uz gausa trokšņa")
    show({"JPEG artifakti": degraded, **apply_filters(degraded)}, f"{name}: filteri uz JPEG artifaktiem")


def main():
    np.random.seed(42)
    for name, image in load_images().items():
        process(name, image)

main()
