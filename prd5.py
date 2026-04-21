from __future__ import annotations

import os
from collections import deque

import cv2
import numpy as np

try:
    from google.colab.patches import cv2_imshow  # type: ignore
except Exception:
    def cv2_imshow(attels):
        """Fallback for non-Colab environments."""
        print("cv2_imshow pieejams tikai Colab vidē.")

KONFIGURACIJA = {
    "img1.png": {
        "block_size": None,
        "C": None,
        "tolerance": None,
        "seklas": None,
        "n_seklas": 6,
        "max_pixels": None,
    },
    "img2.png": {
        "block_size": None,
        "C": None,
        "tolerance": None,
        "seklas": None,
        "n_seklas": 4,
        "max_pixels": None,
    },
    "img3.png": {
        "block_size": None,
        "C": None,
        "tolerance": 8,
        "seklas": None,
        "n_seklas": 5,
        "max_pixels": 50000,
    },
}


KAIMINI_8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def validet_bloka_izmeru(bs: int, h: int, w: int) -> int:
    """Nodrošina derīgu (nepāra) block size adaptive threshold metodei."""
    max_bs = min(h, w)
    if max_bs % 2 == 0:
        max_bs -= 1
    max_bs = max(3, max_bs)
    bs = int(bs)
    bs = max(3, bs)
    if bs % 2 == 0:
        bs += 1
    return min(bs, max_bs)


def auto_parametri(attels: np.ndarray) -> dict:
    """Aprēķina drošus automātiskos parametrus konkrētam attēlam."""
    h, w = attels.shape
    std = np.std(attels)
    min_dim = max(3, min(h, w))
    block_size = max(3, min_dim // 12)
    block_size = validet_bloka_izmeru(block_size, h, w)
    C = int(np.clip(std * 0.06, 1, 30))
    tolerance = int(np.clip(std * 0.2, 2, 35))
    return {"block_size": block_size, "C": C, "tolerance": tolerance}


def apvienot_konfiguraciju(cfg: dict, auto_cfg: dict) -> dict:
    """Atgriež per-attēla konfigurāciju bez globālās KONFIGURACIJA mutēšanas."""
    rez = dict(cfg)
    for key, value in auto_cfg.items():
        if rez.get(key) is None:
            rez[key] = value
    return rez


def slieksnosana(attels: np.ndarray, cfg: dict) -> np.ndarray:
    """Mean-neighborhood thresholding: T(x,y) = neighborhood average - C."""
    h, w = attels.shape
    bs = validet_bloka_izmeru(cfg["block_size"], h, w)
    C = int(np.clip(cfg["C"], 0, 50))
    th = cv2.adaptiveThreshold(
        attels,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        bs,
        C,
    )

    morph_kernel = int(cfg.get("morph_kernel", 3))
    morph_kernel = validet_bloka_izmeru(morph_kernel, h, w)
    k = np.ones((morph_kernel, morph_kernel), np.uint8)
    if cfg.get("do_open", True):
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    if cfg.get("do_close", True):
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    return th


def seklas_no_slieksna(th_maska: np.ndarray, n: int, cfg: dict) -> list[tuple[int, int]]:
    """Automātiski izvēlas sēklas no sliekšņa maskas, filtrējot pēc kontūru laukuma."""
    konturas, _ = cv2.findContours(th_maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = th_maska.shape
    img_area = h * w
    min_area = int(cfg.get("seed_min_area", max(20, img_area * 0.001)))
    max_area = int(cfg.get("seed_max_area", img_area * 0.7))
    filtr_konturas = [cnt for cnt in konturas if min_area <= cv2.contourArea(cnt) <= max_area]
    filtr_konturas = sorted(filtr_konturas, key=cv2.contourArea, reverse=True)

    seklas = []
    for cnt in filtr_konturas:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            seklas.append((cy, cx))
        if len(seklas) >= n:
            break
    return seklas


def audzesana(
    attels: np.ndarray,
    seklas: list[tuple[int, int]],
    tolerance: int,
    max_pixels: int | None = None,
) -> np.ndarray:
    """
    Region growing ar BFS.
    max_pixels ir globāls limits (kopējais pikseļu skaits pāri visām sēklām).
    """
    h, w = attels.shape
    maska = np.zeros((h, w), dtype=np.uint8)
    apmeklets = np.zeros((h, w), dtype=np.uint8)
    tolerance = max(1, int(tolerance))
    if max_pixels is not None:
        max_pixels = max(1, int(max_pixels))

    kopejais = 0
    for sy, sx in seklas:
        if max_pixels is not None and kopejais >= max_pixels:
            break
        if not (0 <= sy < h and 0 <= sx < w) or apmeklets[sy, sx]:
            continue

        sakuma_verts = int(attels[sy, sx])
        rinda = deque([(sy, sx)])
        apmeklets[sy, sx] = 1
        maska[sy, sx] = 255
        kopejais += 1
        summas = sakuma_verts
        pikselu_skaits = 1

        while rinda:
            if max_pixels is not None and kopejais >= max_pixels:
                break
            y, x = rinda.popleft()
            region_mean = summas / pikselu_skaits
            for dy, dx in KAIMINI_8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not apmeklets[ny, nx]:
                    apmeklets[ny, nx] = 1
                    if abs(int(attels[ny, nx]) - region_mean) <= tolerance:
                        pikselis = int(attels[ny, nx])
                        summas += pikselis
                        pikselu_skaits += 1
                        kopejais += 1
                        maska[ny, nx] = 255
                        rinda.append((ny, nx))
                        if max_pixels is not None and kopejais >= max_pixels:
                            break
    return maska


def notirit_mazus_rezultatus(maska: np.ndarray, min_area: int = 100) -> np.ndarray:
    konturas, _ = cv2.findContours(maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tira_maska = np.zeros_like(maska)
    for cnt in konturas:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(tira_maska, [cnt], -1, 255, -1)
    return tira_maska


def izveidot_overlay(originals: np.ndarray, maska: np.ndarray) -> np.ndarray:
    """Pārklāj segmentācijas masku uz oriģinālā attēla."""
    if originals.ndim == 2:
        overlay = cv2.cvtColor(originals, cv2.COLOR_GRAY2BGR)
    else:
        overlay = originals.copy()
    overlay[maska > 0] = (0, 0, 255)
    return cv2.addWeighted(originals if originals.ndim == 3 else cv2.cvtColor(originals, cv2.COLOR_GRAY2BGR), 0.7, overlay, 0.3, 0)


def main() -> None:
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)

    for faila_nosaukums, cfg in KONFIGURACIJA.items():
        originals = cv2.imread(faila_nosaukums)
        attels = cv2.imread(faila_nosaukums, cv2.IMREAD_GRAYSCALE)
        if originals is None or attels is None:
            print(f"Nevar ielādēt failu: {faila_nosaukums}")
            continue

        auto_cfg = auto_parametri(attels)
        merged_cfg = apvienot_konfiguraciju(cfg, auto_cfg)

        rez_slieksnis = slieksnosana(attels, merged_cfg)

        if merged_cfg["seklas"] is not None:
            seklas = merged_cfg["seklas"]
        else:
            seklas = seklas_no_slieksna(rez_slieksnis, merged_cfg["n_seklas"], merged_cfg)

        rez_audzesana = notirit_mazus_rezultatus(
            audzesana(attels, seklas, merged_cfg["tolerance"], merged_cfg.get("max_pixels"))
        )

        bāze = os.path.splitext(os.path.basename(faila_nosaukums))[0]
        overlay = izveidot_overlay(originals, rez_audzesana)

        cv2.imwrite(os.path.join(out_dir, f"{bāze}_original.png"), originals)
        cv2.imwrite(os.path.join(out_dir, f"{bāze}_threshold.png"), rez_slieksnis)
        cv2.imwrite(os.path.join(out_dir, f"{bāze}_region.png"), rez_audzesana)
        cv2.imwrite(os.path.join(out_dir, f"{bāze}_overlay.png"), overlay)

        cv2_imshow(originals)
        cv2_imshow(rez_slieksnis)
        cv2_imshow(rez_audzesana)


if __name__ == "__main__":
    main()
