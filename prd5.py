import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from collections import deque

KONFIGURACIJA = {
    'img1.png': {'block_size': None, 'C': None, 'tolerance': None, 'seklas': None, 'n_seklas': 6,  'max_pixels': None},
    'img2.png': {'block_size': None, 'C': None, 'tolerance': None, 'seklas': None, 'n_seklas': 4,  'max_pixels': None},
    'img3.png': {'block_size': None, 'C': None, 'tolerance': 8,    'seklas': None, 'n_seklas': 5,  'max_pixels': 50000},
}

def auto_parametri(attels):
    std = np.std(attels)
    tolerance = int(std * 0.2)
    block_size = max(3, attels.shape[0] // 30)
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    C = max(2, int(std * 0.05))
    return {'block_size': block_size, 'C': C, 'tolerance': tolerance}

def validet_bloka_izmeru(bs):
    bs = max(3, bs)
    return bs if bs % 2 == 1 else bs + 1

def slieksnosana(attels, cfg):
    bs = validet_bloka_izmeru(cfg['block_size'])
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    attels_clahe = clahe.apply(attels)
    th = cv2.adaptiveThreshold(
        attels_clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bs, cfg['C']
    )
    k = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    return th

def seklas_no_slieksna(th_maska, n):
    konturas, _ = cv2.findContours(th_maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    konturas = sorted(konturas, key=cv2.contourArea, reverse=True)
    seklas = []
    for cnt in konturas:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            seklas.append((cy, cx))
        if len(seklas) >= n:
            break
    return seklas

def audzesana(attels, seklas, tolerance, max_pixels=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    attels_e = clahe.apply(attels)

    h, w = attels_e.shape
    maska = np.zeros((h, w), dtype=np.uint8)
    apmeklets = np.zeros((h, w), dtype=bool)
    kopejais = 0
    for sy, sx in seklas:
        if not (0 <= sy < h and 0 <= sx < w) or apmeklets[sy, sx]:
            continue
        sakuma_verts = int(attels_e[sy, sx])
        rinda = deque([(sy, sx)])
        apmeklets[sy, sx] = True
        while rinda:
            if max_pixels and kopejais >= max_pixels:
                break
            y, x = rinda.popleft()
            maska[y, x] = 255
            kopejais += 1
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not apmeklets[ny, nx]:
                    apmeklets[ny, nx] = True
                    if abs(int(attels_e[ny, nx]) - sakuma_verts) <= tolerance:
                        rinda.append((ny, nx))
    return maska

def notirit_mazus_rezultatus(maska, min_area=100):
    konturas, _ = cv2.findContours(maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tira_maska = np.zeros_like(maska)
    for cnt in konturas:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(tira_maska, [cnt], -1, 255, -1)
    return tira_maska

for faila_nosaukums, cfg in KONFIGURACIJA.items():
    originals = cv2.imread(faila_nosaukums)
    attels = cv2.imread(faila_nosaukums, cv2.IMREAD_GRAYSCALE)

    auto_cfg = auto_parametri(attels)
    for key in ['block_size', 'C', 'tolerance']:
        if cfg.get(key) is None:
            cfg[key] = auto_cfg[key]

    rez_slieksnis = slieksnosana(attels, cfg)

    if cfg['seklas'] is not None:
        seklas = cfg['seklas']
    else:
        seklas = seklas_no_slieksna(rez_slieksnis, cfg['n_seklas'])

    rez_audzesana = notirit_mazus_rezultatus(
        audzesana(attels, seklas, cfg['tolerance'], cfg.get('max_pixels'))
    )

    cv2_imshow(originals)
    cv2_imshow(rez_slieksnis)
    cv2_imshow(rez_audzesana)
