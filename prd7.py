import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

def uzlabot_kontrastu_kombinets(image_rgb):
    yuv_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv_img)
    
    vid_gaisums = np.mean(y)
    y_float = y.astype(np.float32)
    
    if vid_gaisums < 100:
        c = 255.0 / np.log(1.0 + np.max(y_float))
        y_apstr = c * np.log(1.0 + y_float)
        y_apstr = np.clip(y_apstr, 0, 255).astype(np.uint8)
        izmantota_metode = "Logaritmiskā korekcija"
    else:
        min_val = np.min(y_float)
        max_val = np.max(y_float)
        if max_val > min_val:
            y_apstr = 255.0 * ((y_float - min_val) / (max_val - min_val))
        else:
            y_apstr = y_float
        y_apstr = np.clip(y_apstr, 0, 255).astype(np.uint8)
        izmantota_metode = "Min-Max stiepšana"
        
    yuv_apstr = cv2.merge([y_apstr, u, v])
    apstr_rgb = cv2.cvtColor(yuv_apstr, cv2.COLOR_YUV2RGB)
    
    return apstr_rgb, izmantota_metode

def pasliktinat_sakura(img):
    img_float = img.astype(np.float32)
    img_compressed = 80 + (img_float / 255.0) * (180 - 80)
    return np.clip(img_compressed, 0, 255).astype(np.uint8)

def pasliktinat_korgi(img):
    img_dark = img.astype(np.float32) / 3.5
    return np.clip(img_dark, 0, 255).astype(np.uint8)

def pasliktinat_car(img):
    img_washed = np.where(img < 100, 100, img)
    return img_washed.astype(np.uint8)

def aprekinat_metrikas(orig_img, test_img):
    p = psnr(orig_img, test_img, data_range=255)
    s = ssim(orig_img, test_img, data_range=255, channel_axis=2)
    m = mse(orig_img, test_img)
    rmse = np.sqrt(m)
    return p, s, m, rmse

def ieladet_attelu(celsh):
    img = cv2.imread(celsh)
    if img is None:
        raise FileNotFoundError(f"Nevarēja atrast failu: {celsh}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

failu_dati = [
    {"nosaukums": "sakura_park.png", "pasliktinat_fn": pasliktinat_sakura, "nosaukums_viz": "Sakuras Parks"},
    {"nosaukums": "korgi+girl.png", "pasliktinat_fn": pasliktinat_korgi, "nosaukums_viz": "Meitene un Korgijs"},
    {"nosaukums": "red_car.png", "pasliktinat_fn": pasliktinat_car, "nosaukums_viz": "Sporta Auto"}
]

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("7. Praktiskais Darbs: Attēlu Apstrādes Novērtēšana", fontsize=16, weight='bold')

for i, dati in enumerate(failu_dati):
    try:
        orig_img = ieladet_attelu(dati["nosaukums"])
    except FileNotFoundError as e:
        print(e)
        continue
    
    pasl_img = dati["pasliktinat_fn"](orig_img)
    psnr_pasl, ssim_pasl, mse_pasl, rmse_pasl = aprekinat_metrikas(orig_img, pasl_img)
    
    apstr_img, metode = uzlabot_kontrastu_kombinets(pasl_img)
    psnr_apstr, ssim_apstr, mse_apstr, rmse_apstr = aprekinat_metrikas(orig_img, apstr_img)
    
    ax = axes[i, 0]
    ax.imshow(orig_img)
    ax.set_title(f"Oriģināls: {dati['nosaukums_viz']}")
    ax.axis('off')
    
    ax = axes[i, 1]
    ax.imshow(pasl_img)
    ax.set_title(f"Pasliktināts\nPSNR: {psnr_pasl:.2f} | SSIM: {ssim_pasl:.3f}\nMSE: {mse_pasl:.1f} | RMSE: {rmse_pasl:.1f}")
    ax.axis('off')
    
    ax = axes[i, 2]
    ax.imshow(apstr_img)
    ax.set_title(f"Apstrādāts ({metode})\nPSNR: {psnr_apstr:.2f} | SSIM: {ssim_apstr:.3f}\nMSE: {mse_apstr:.1f} | RMSE: {rmse_apstr:.1f}")
    ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
