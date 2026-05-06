import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def uzlabot_kontrastu_kombinets(img, slieksnis_tumšam=100):
    """
    Apstrādes loģika: YUV transformācija + adaptīvā logaritmiskā korekcija.
    """
    # 1. Transformācija uz YUV un kanālu atdalīšana
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_img)
    y_float = y.astype(np.float32)

    # 2. Nosacījuma pārbaude un logaritmiskā korekcija
    # Formula: $S = c \cdot \log(1 + R)$
    videjais_gaisums = np.mean(y)
    
    if videjais_gaisums < slieksnis_tumšam:
        # Agresīvāks mērogošanas koeficients tumšiem attēliem
        c = 255.0 / np.log(1 + np.max(y_float) if np.max(y_float) > 0 else 1)
    else:
        # Standarta koeficients gaišākiem attēliem
        c = 255.0 / np.log(1 + 255.0)
    
    y_apstradats = c * np.log(1 + y_float)
    y_apstradats = np.clip(y_apstradats, 0, 255).astype(np.uint8)

    # 3. Rekonstrukcija
    yuv_uzlabots = cv2.merge((y_apstradats, u, v))
    rezultats = cv2.cvtColor(yuv_uzlabots, cv2.COLOR_YUV2BGR)
    
    return rezultats, y, y_apstradats

# Attēlu saraksts
faili = ["img1.png", "img2.png", "img3.png", "img4.png", "img5.png"]

for faila_nosaukums in faili:
    if not os.path.exists(faila_nosaukums):
        print(f"Brīdinājums: Fails {faila_nosaukums} netika atrasts. Izlaižu...")
        continue

    # Ielasa attēlu
    original_bgr = cv2.imread(faila_nosaukums)
    
    # Veic apstrādi
    uzlabots_bgr, y_orig, y_uzlabots = uzlabot_kontrastu_kombinets(original_bgr)
    
    # Sagatavo vizualizāciju (2x2 režģis)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Analīze: {faila_nosaukums}", fontsize=16)

    # 1. Oriģinālais attēls (konvertēts uz RGB rādīšanai)
    axs[0, 0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Oriģināls")
    axs[0, 0].axis('off')

    # 2. Uzlabotais attēls
    axs[0, 1].imshow(cv2.cvtColor(uzlabots_bgr, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title("Uzlabots (YUV + Log)")
    axs[0, 1].axis('off')

    # 3. Oriģinālā histogramma (tikai Y kanālam)
    axs[1, 0].hist(y_orig.ravel(), 256, [0, 256], color='black')
    axs[1, 0].set_title("Oriģinālā histogramma (Luminance)")
    axs[1, 0].set_xlim([0, 256])

    # 4. Uzlabotā histogramma
    axs[1, 1].hist(y_uzlabots.ravel(), 256, [0, 256], color='blue')
    axs[1, 1].set_title("Uzlabotā histogramma (Luminance)")
    axs[1, 1].set_xlim([0, 256])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Saglabā rezultātu kā jaunu failu salīdzināšanai
    plt.savefig(f"analize_{faila_nosaukums}")
    plt.show()
