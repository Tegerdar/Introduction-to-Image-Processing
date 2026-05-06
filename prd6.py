import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def uzlabot_kontrastu_kombinets(img, slieksnis_tumšam=100):
    # 1. Transformācija uz YUV un kanālu atdalīšana
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_img)
    y_float = y.astype(np.float32)

    # 2. Analīze
    videjais_gaisums = np.mean(y)
    
    # 3. Zarošanās
    if videjais_gaisums < slieksnis_tumšam:
        # A ZARS (Logaritmiskā korekcija)
        c = 255.0 / np.log(1 + np.max(y_float) if np.max(y_float) > 0 else 1)
        y_apstradats = c * np.log(1 + y_float)
        y_apstradats = np.clip(y_apstradats, 0, 255).astype(np.uint8)
        metode = "YUV + Logaritmiskā korekcija"
    else:
        # B ZARS (Lineārā stiepšana)
        y_apstradats = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX)
        metode = "YUV + Min-Max stiepšana"

    # 4. Rekonstrukcija
    yuv_uzlabots = cv2.merge((y_apstradats, u, v))
    rezultats = cv2.cvtColor(yuv_uzlabots, cv2.COLOR_YUV2BGR)
    
    return rezultats, y, y_apstradats, metode

# Attēlu saraksts
faili = ["img1.png", "img2.png", "img3.png", "img4.png", "img5.png"]

for faila_nosaukums in faili:
    if not os.path.exists(faila_nosaukums):
        print(f"Brīdinājums: Fails {faila_nosaukums} netika atrasts.")
        continue

    # Ielasa attēlu
    original_bgr = cv2.imread(faila_nosaukums)
    
    # Veic apstrādi
    uzlabots_bgr, y_orig, y_uzlabots, izmantota_metode = uzlabot_kontrastu_kombinets(original_bgr)
    
    # Sagatavo vizualizāciju
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Analīze: {faila_nosaukums} | Izvēlētais zars: {izmantota_metode}", fontsize=14)

    # Oriģinālais attēls
    axs[0, 0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Oriģināls")
    axs[0, 0].axis('off')

    # Uzlabotais attēls
    axs[0, 1].imshow(cv2.cvtColor(uzlabots_bgr, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title(f"Uzlabots ({izmantota_metode})")
    axs[0, 1].axis('off')

    # Oriģinālā histogramma
    axs[1, 0].hist(y_orig.ravel(), 256, [0, 256], color='black')
    axs[1, 0].set_title("Oriģinālā histogramma (Y kanāls)")
    axs[1, 0].set_xlim([0, 256])

    # Uzlabotā histogramma
    axs[1, 1].hist(y_uzlabots.ravel(), 256, [0, 256], color='blue')
    axs[1, 1].set_title("Uzlabotā histogramma (Y kanāls)")
    axs[1, 1].set_xlim([0, 256])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Saglabā rezultātu ar jaunu nosaukumu, lai nesajauktu ar vecajiem
    jauns_nosaukums = f"jaunais_rezultats_{faila_nosaukums}"
    plt.savefig(jauns_nosaukums)
    plt.show()
