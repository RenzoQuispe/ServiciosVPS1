"""
Canvas-a-Productos — extracción de productos desde catálogos Canva.

- Captura screenshots por página (Selenium/Chrome), detecta tarjetas con OpenAI.
- Bbox solo sobre la imagen del producto (sin título, descripción ni precio en el crop).
- Recorte con padding mínimo (2% por defecto); smart crop desactivado por defecto.
- Opcional: two-pass (refina texto por crop), enrich (CSV enriquecido).

Uso CLI:
  python script.py --url "https://tu-sitio.canva.site" --out salida
  python script.py --url "..." --out salida --two-pass --enrich

La API (app.py) llama a process_url_to_csv_openai() con parámetros desde env.
"""

import os
import re
import csv
import time
import base64
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

# OpenAI
from openai import OpenAI
from pydantic import BaseModel, Field


# =========================
# Utils
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def safe_lower(s: str) -> str:
    return (s or "").strip().lower()

def norm_price(s: str) -> str:
    s = norm_spaces(s)
    s = re.sub(r"\bS/\s+", "S/", s, flags=re.IGNORECASE)
    return s

def image_to_data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png"
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"
    elif ext == ".gif":
        mime = "image/gif"

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def file_suffix(fn: str) -> str:
    m = re.search(r"(\d+)", fn)
    return m.group(1) if m else fn

def laplacian_sharpness(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# =========================
# Hash de imagen
# =========================

def dhash_bgr(img_bgr: np.ndarray, hash_size: int = 8) -> int:
    if img_bgr is None or img_bgr.size == 0:
        return 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    h = 0
    bit = 1
    for v in diff.flatten():
        if v:
            h |= bit
        bit <<= 1
    return h

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# =========================
# Screenshot normalization
# =========================

def trim_black_bars(img_bgr: np.ndarray, thr: int = 18) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    mask = (gray > thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    ys, xs = np.where(mask > 0)
    if len(xs) < 50 or len(ys) < 50:
        return img_bgr, (0, 0, w, h)

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    pad = int(0.005 * min(w, h))
    x1 = clamp(x1 - pad, 0, w - 1)
    y1 = clamp(y1 - pad, 0, h - 1)
    x2 = clamp(x2 + pad, 1, w)
    y2 = clamp(y2 + pad, 1, h)

    cropped = img_bgr[y1:y2, x1:x2].copy()
    return cropped, (x1, y1, x2, y2)

def normalize_screenshot_inplace(path: str, do_trim_blackbars: bool = True, thr: int = 18) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        return None

    if do_trim_blackbars:
        trimmed, _ = trim_black_bars(img, thr=thr)
        cv2.imwrite(path, trimmed)
        return trimmed

    return img


# =========================
# Selenium (Canva paginado)
# =========================

def setup_chrome(headless: bool = True):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1600,900")
    opts.add_argument("--hide-scrollbars")
    opts.add_argument("--lang=es-PE")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    return webdriver.Chrome(options=opts)

def close_common_overlays(driver) -> None:
    try:
        buttons = driver.find_elements(
            By.XPATH,
            "//button[contains(., 'Aceptar') or contains(., 'Accept') or contains(., 'OK') or contains(., 'Entendido')]"
        )
        if buttons:
            try:
                buttons[0].click()
                time.sleep(0.4)
            except Exception:
                pass
    except Exception:
        pass

def find_canvas_like_element(driver):
    candidates = driver.find_elements(By.XPATH, "//*")
    best = None
    best_area = 0
    for el in candidates:
        try:
            r = el.rect
            w = int(r.get("width", 0))
            h = int(r.get("height", 0))
            if w < 300 or h < 300:
                continue
            area = w * h
            if area > best_area:
                best_area = area
                best = el
        except Exception:
            continue
    return best

def ensure_canva_controls_visible(driver, canvas_el=None) -> None:
    try:
        actions = ActionChains(driver)
        if canvas_el is None:
            canvas_el = find_canvas_like_element(driver)

        if canvas_el is not None:
            actions.move_to_element_with_offset(canvas_el, 20, 20).perform()
            time.sleep(0.12)
            actions.move_to_element_with_offset(canvas_el, 250, 250).perform()
            time.sleep(0.12)

            rect = canvas_el.rect
            off_x = 80
            off_y = int(rect.get("height", 600)) - 60
            off_y = max(20, off_y)
            actions.move_to_element_with_offset(canvas_el, off_x, off_y).perform()
            time.sleep(0.12)
        else:
            body = driver.find_element(By.TAG_NAME, "body")
            actions.move_to_element_with_offset(body, 20, 20).perform()
            time.sleep(0.1)
            actions.move_to_element_with_offset(body, 300, 700).perform()
            time.sleep(0.1)
    except Exception:
        pass

def try_click_next_canva(driver, canvas_el=None) -> bool:
    ensure_canva_controls_visible(driver, canvas_el=canvas_el)

    xpaths = [
        "//button[contains(@aria-label,'Next') or contains(@aria-label,'Siguiente') or contains(@aria-label,'next') or contains(@aria-label,'siguiente')]",
        "//*[@role='button' and (contains(@aria-label,'Next') or contains(@aria-label,'Siguiente') or contains(@aria-label,'next') or contains(@aria-label,'siguiente'))]",
        "//*[@role='button' and (contains(., 'Next') or contains(., 'Siguiente'))]",
    ]

    for xp in xpaths:
        try:
            els = driver.find_elements(By.XPATH, xp)
            for el in els[:6]:
                try:
                    driver.execute_script("arguments[0].click();", el)
                    return True
                except Exception:
                    try:
                        el.click()
                        return True
                    except Exception:
                        continue
        except Exception:
            continue

    try:
        body = driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.ARROW_RIGHT)
        return True
    except Exception:
        return False

def capture_canva_paginated_screens(
    driver,
    out_dir: str,
    max_pages: Optional[int] = None,
    wait_after_nav: float = 1.0,
    dup_stop_limit: int = 4,
    hash_near_threshold: int = 0,
    do_trim_blackbars: bool = True,
    trim_thr: int = 18,
) -> List[str]:
    ensure_dir(out_dir)
    paths: List[str] = []

    canvas_el = find_canvas_like_element(driver)

    last_hash: Optional[int] = None
    dup_count = 0
    idx = 1

    def read_indicator() -> Optional[Tuple[int, int]]:
        ensure_canva_controls_visible(driver, canvas_el=canvas_el)
        candidates = driver.find_elements(By.XPATH, "//*[contains(normalize-space(.), '/')]")
        for el in candidates:
            try:
                txt = (el.text or "").strip()
                m = re.search(r"^\s*(\d+)\s*/\s*(\d+)\s*$", txt)
                if m:
                    return int(m.group(1)), int(m.group(2))
            except Exception:
                continue
        return None

    ind = read_indicator()
    total_from_ind = ind[1] if ind else None
    limit = max_pages if (max_pages is not None) else (total_from_ind if total_from_ind is not None else 999999)

    while idx <= limit:
        ensure_canva_controls_visible(driver, canvas_el=canvas_el)
        time.sleep(0.2)

        out_path = os.path.join(out_dir, f"screen_{idx:03d}.png")
        driver.save_screenshot(out_path)

        img_norm = normalize_screenshot_inplace(out_path, do_trim_blackbars=do_trim_blackbars, thr=trim_thr)
        if img_norm is None:
            clicked = try_click_next_canva(driver, canvas_el=canvas_el)
            time.sleep(wait_after_nav)
            if not clicked:
                print("[STOP] No pude avanzar en Canva. Cortando.")
                break
            idx += 1
            continue

        cur_hash = dhash_bgr(img_norm)

        if last_hash is not None:
            dist = hamming_distance(cur_hash, last_hash)
            if dist <= hash_near_threshold:
                dup_count += 1
            else:
                dup_count = 0
        last_hash = cur_hash

        if dup_count >= 1:
            try:
                os.remove(out_path)
            except Exception:
                pass

            if dup_count >= dup_stop_limit:
                print(f"[STOP] Screenshot no cambia tras {dup_count} intentos. Cortando para evitar gasto.")
                break

            clicked = try_click_next_canva(driver, canvas_el=canvas_el)
            time.sleep(wait_after_nav + (0.5 * dup_count))
            if not clicked:
                print("[STOP] No pude avanzar en Canva. Cortando.")
                break
            continue

        paths.append(out_path)

        ind2 = read_indicator()
        if ind2 and ind2[0] >= ind2[1]:
            break

        clicked = try_click_next_canva(driver, canvas_el=canvas_el)
        time.sleep(wait_after_nav)
        if not clicked:
            print("[STOP] No pude avanzar en Canva. Cortando.")
            break

        idx += 1

    return paths

def fullpage_screenshots(
    url: str,
    out_dir: str,
    wait_sec: float = 3.0,
    max_pages: Optional[int] = None,
    dup_stop_limit: int = 4,
    hash_near_threshold: int = 0,
    do_trim_blackbars: bool = True,
    trim_thr: int = 18,
) -> List[str]:
    ensure_dir(out_dir)
    driver = setup_chrome(headless=True)
    try:
        driver.get(url)
        time.sleep(wait_sec)
        close_common_overlays(driver)
        driver.set_window_size(1600, 1200)
        time.sleep(0.6)

        return capture_canva_paginated_screens(
            driver,
            out_dir=out_dir,
            max_pages=max_pages,
            wait_after_nav=1.0,
            dup_stop_limit=dup_stop_limit,
            hash_near_threshold=hash_near_threshold,
            do_trim_blackbars=do_trim_blackbars,
            trim_thr=trim_thr,
        )
    finally:
        driver.quit()


# =========================
# OpenAI schemas - V3 ULTRA-PRECISO
# =========================

class BBoxNorm(BaseModel):
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

class ProductDetected(BaseModel):
    nombre: str = Field(default="")
    variante: str = Field(default="")
    precio: str = Field(default="")
    bbox: BBoxNorm
    confianza: float = Field(default=0.0, ge=0.0, le=1.0)

class ProductsPage(BaseModel):
    productos: List[ProductDetected] = Field(default_factory=list)

class ProductExtract(BaseModel):
    nombre: str = Field(default="")
    variante: str = Field(default="")
    descripcion: str = Field(default="")
    precio: str = Field(default="")


def _usage_from_response(resp) -> Dict[str, int]:
    """Extrae input/output/total tokens del response de OpenAI (responses API)."""
    u = getattr(resp, "usage", None)
    if u is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "total_tokens": getattr(u, "total_tokens", 0) or 0,
    }


def openai_page_detect(client: OpenAI, screenshot_path: str, model: str, max_retries: int = 3) -> Tuple[ProductsPage, Dict[str, int]]:
    """
    VERSIÓN 3.0 ULTRA-PRECISA - No incluye texto lateral
    """
    data_url = image_to_data_url(screenshot_path)

    system = (
        "Eres un extractor ULTRA-PRECISO de imágenes de producto en catálogos.\n\n"
        "OBJETIVO CRÍTICO: El bbox debe contener EXCLUSIVAMENTE la imagen/foto del producto.\n"
        "NO INCLUIR texto, precio, descripción ni botones que estén AL LADO de la imagen.\n\n"
        "═══════════════════════════════════════════════════════════════\n"
        "REGLAS ESTRICTAS PARA BBOX (x1, y1, x2, y2 normalizados 0-1):\n"
        "═══════════════════════════════════════════════════════════════\n\n"
        "1. LAYOUT VERTICAL (imagen arriba, texto abajo):\n"
        "   - bbox cubre SOLO la parte superior donde está la imagen\n"
        "   - NO incluir título/nombre debajo de la imagen\n"
        "   - NO incluir precio debajo de la imagen\n"
        "   - Ejemplo: si imagen ocupa 60% superior → bbox = (0.1, 0.05, 0.9, 0.65)\n\n"
        "2. LAYOUT HORIZONTAL (imagen izquierda, texto derecha):\n"
        "   - bbox cubre SOLO la parte izquierda donde está la imagen\n"
        "   - NO incluir texto/descripción a la derecha\n"
        "   - NO incluir precio a la derecha\n"
        "   - Ejemplo: si imagen ocupa 40% izquierdo → bbox = (0.05, 0.1, 0.45, 0.9)\n\n"
        "3. LAYOUT MIXTO (imagen central con texto alrededor):\n"
        "   - bbox cubre SOLO la zona central de la imagen\n"
        "   - Ignora texto lateral/superior/inferior\n"
        "   - Enfócate en la foto/ilustración del producto\n\n"
        "4. MARGEN MÍNIMO:\n"
        "   - Solo 1-2% de margen alrededor de la imagen\n"
        "   - NO dejar espacio blanco/fondo excesivo\n"
        "   - El bbox debe estar AJUSTADO a los bordes visuales del producto\n\n"
        "5. CASOS ESPECIALES:\n"
        "   - Si la imagen tiene forma circular/irregular: bbox = rectángulo mínimo\n"
        "   - Si hay sombras/reflejos DENTRO de la imagen: inclúyelos\n"
        "   - Si hay sombras/efectos FUERA separados: NO incluirlos\n\n"
        "6. VALIDACIÓN ASPECT RATIO:\n"
        "   - Si la tarjeta es horizontal pero la imagen es cuadrada/vertical:\n"
        "     → bbox debe ser cuadrado/vertical (no todo el ancho)\n"
        "   - Si detectas aspect ratio >3:1 o >1:3 → probablemente incluiste texto lateral\n\n"
        "7. EJEMPLOS DE LO QUE NO DEBES HACER:\n"
        "   ❌ Incluir \"Nombre del producto\" que está debajo/al lado\n"
        "   ❌ Incluir \"S/199\" que está abajo/derecha\n"
        "   ❌ Incluir botón \"Comprar\" que está debajo\n"
        "   ❌ Incluir texto descriptivo lateral\n"
        "   ❌ Dejar >5% de margen blanco alrededor\n\n"
        "═══════════════════════════════════════════════════════════════\n"
        "CAMPOS A DEVOLVER:\n"
        "═══════════════════════════════════════════════════════════════\n\n"
        "- bbox: Coordenadas PRECISAS de solo la imagen (sin texto lateral/inferior)\n"
        "- nombre: Extrae del texto visible FUERA del bbox (puede estar debajo/al lado)\n"
        "- variante: Color/talla/modelo si se ve en el texto\n"
        "- precio: Precio exacto si se ve en el texto\n"
        "- confianza: 0.8-1.0 si bbox muy claro, 0.5-0.7 si hay ambigüedad, <0.5 si dudoso\n\n"
        "PRIORIDAD: Es MEJOR un bbox más pequeño que incluya solo la imagen,\n"
        "que un bbox grande que incluya texto lateral.\n\n"
        "Devuelve SOLO el schema JSON, sin explicaciones."
    )

    user = (
        "Analiza esta imagen de catálogo y detecta todas las tarjetas de producto.\n\n"
        "Para cada producto:\n"
        "1. Identifica dónde está la IMAGEN/FOTO del producto (ignorando texto)\n"
        "2. Define un bbox QUE CONTENGA SOLO ESA IMAGEN\n"
        "3. Si la tarjeta tiene layout horizontal (imagen izq, texto der):\n"
        "   → El bbox debe cubrir solo la mitad izquierda donde está la imagen\n"
        "4. Si la tarjeta tiene layout vertical (imagen arriba, texto abajo):\n"
        "   → El bbox debe cubrir solo la mitad superior donde está la imagen\n"
        "5. Extrae nombre/precio/variante del TEXTO (aunque esté fuera del bbox)\n\n"
        "RECUERDA: bbox SOLO para la imagen, NO incluir texto que esté al lado o debajo."
    )

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "input_text", "text": user},
                        {"type": "input_image", "image_url": data_url, "detail": "high"},
                    ]},
                ],
                text_format=ProductsPage,
            )
            return resp.output_parsed, _usage_from_response(resp)
        except Exception as e:
            last_err = e
            time.sleep(0.8 * attempt)

    print(f"[WARN] Pass1 failed for {screenshot_path}: {last_err}")
    return ProductsPage(), {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

def openai_refine_crop(client: OpenAI, crop_path: str, model: str, max_retries: int = 3) -> Tuple[ProductExtract, Dict[str, int]]:
    data_url = image_to_data_url(crop_path)

    system = "Eres un extractor de producto desde una imagen de una tarjeta/crop. Devuelve SOLO el schema. No inventes."
    user = (
        "Extrae nombre, variante/color, descripcion (corta si existe) y precio exactamente como aparece. "
        "Si no se ve claro, devuelve string vacío."
    )

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "input_text", "text": user},
                        {"type": "input_image", "image_url": data_url, "detail": "high"},
                    ]},
                ],
                text_format=ProductExtract,
            )
            return resp.output_parsed, _usage_from_response(resp)
        except Exception as e:
            last_err = e
            time.sleep(0.8 * attempt)

    print(f"[WARN] Pass2 failed for {crop_path}: {last_err}")
    return ProductExtract(), {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


# =========================
# Smart Crop V3 - ULTRA PRECISO
# =========================

def detect_text_regions(region_bgr: np.ndarray) -> np.ndarray:
    """
    Detecta regiones con texto usando análisis de varianza y bordes.
    Retorna máscara donde 255 = probablemente texto, 0 = no texto.
    """
    h, w = region_bgr.shape[:2]
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detectar bordes (texto tiene muchos bordes)
    edges = cv2.Canny(gray, 50, 150)
    
    # Análisis de densidad de bordes en ventanas
    kernel = np.ones((20, 20), np.uint8)
    edge_density = cv2.filter2D(edges.astype(float), -1, kernel / kernel.sum())
    
    # Regiones con alta densidad de bordes = probablemente texto
    text_mask = (edge_density > 15).astype(np.uint8) * 255
    
    # Morfología para limpiar
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, k)
    
    return text_mask

def bbox_norm_to_px(b: BBoxNorm, w: int, h: int, pad_rel: float = 0.02) -> Tuple[int, int, int, int]:
    """Convierte bbox normalizado a píxeles con padding mínimo."""
    x1 = int((b.x1 - pad_rel) * w)
    y1 = int((b.y1 - pad_rel) * h)
    x2 = int((b.x2 + pad_rel) * w)
    y2 = int((b.y2 + pad_rel) * h)
    x1 = clamp(x1, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    x2 = clamp(x2, 1, w)
    y2 = clamp(y2, 1, h)
    return x1, y1, x2, y2

def estimate_bg_color(region_bgr: np.ndarray) -> np.ndarray:
    """Estima color de fondo desde los bordes."""
    h, w = region_bgr.shape[:2]
    b = max(3, int(0.03 * min(h, w)))
    strips = [
        region_bgr[0:b, :, :],
        region_bgr[h - b:h, :, :],
        region_bgr[:, 0:b, :],
        region_bgr[:, w - b:w, :],
    ]
    border = np.concatenate([s.reshape(-1, 3) for s in strips], axis=0).astype(np.float32)
    bg = np.percentile(border, 75, axis=0)
    return bg

def smart_refine_crop_v3(
    img_bgr: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    max_area_ratio: float = 0.55,
    min_content_threshold: float = 25.0,
    exclude_text_regions: bool = True
) -> Tuple[int, int, int, int]:
    """
    V3: Refinamiento ultra-preciso con exclusión de texto.
    """
    h, w = img_bgr.shape[:2]
    region = img_bgr[y1:y2, x1:x2].copy()
    rh, rw = region.shape[:2]
    
    if rh < 60 or rw < 60:
        return x1, y1, x2, y2

    # Estimar fondo
    bg = estimate_bg_color(region)
    diff = np.abs(region.astype(np.float32) - bg.reshape(1, 1, 3))
    score = diff[:, :, 0] + diff[:, :, 1] + diff[:, :, 2]

    # Threshold adaptativo MÁS conservador
    thr = max(min_content_threshold, np.percentile(score, 45) * 0.65)
    content_mask = (score > thr).astype(np.uint8) * 255

    # Detectar y EXCLUIR regiones de texto
    if exclude_text_regions:
        text_mask = detect_text_regions(region)
        # Invertir: queremos contenido que NO sea texto
        # Pero solo excluir texto que está en los bordes
        border_margin = int(0.15 * min(rw, rh))
        
        # Máscara de bordes (donde podría haber texto lateral/inferior)
        border_mask = np.zeros((rh, rw), dtype=np.uint8)
        border_mask[:border_margin, :] = 255  # Top
        border_mask[rh-border_margin:, :] = 255  # Bottom
        border_mask[:, :border_margin] = 255  # Left
        border_mask[:, rw-border_margin:] = 255  # Right
        
        # Texto en bordes = probablemente labels/precio
        text_in_borders = cv2.bitwise_and(text_mask, border_mask)
        
        # Excluir ese texto
        content_mask = cv2.bitwise_and(content_mask, cv2.bitwise_not(text_in_borders))

    # Morfología más conservadora
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, k_open, iterations=1)

    ys, xs = np.where(content_mask > 0)
    
    if len(xs) < 400 or len(ys) < 400:
        return x1, y1, x2, y2

    cx1, cx2 = int(xs.min()), int(xs.max())
    cy1, cy2 = int(ys.min()), int(ys.max())

    # Padding MUY reducido
    content_w = cx2 - cx1
    content_h = cy2 - cy1
    pad = int(0.02 * max(content_w, content_h))  # 2% del contenido
    
    cx1 = clamp(cx1 - pad, 0, rw - 1)
    cy1 = clamp(cy1 - pad, 0, rh - 1)
    cx2 = clamp(cx2 + pad, 1, rw)
    cy2 = clamp(cy2 + pad, 1, rh)

    # Coordenadas globales
    gx1 = clamp(x1 + cx1, 0, w - 1)
    gy1 = clamp(y1 + cy1, 0, h - 1)
    gx2 = clamp(x1 + cx2, 1, w)
    gy2 = clamp(y1 + cy2, 1, h)

    # Validación de área máxima
    max_area = max_area_ratio * (w * h)
    area = (gx2 - gx1) * (gy2 - gy1)
    if area > max_area:
        return x1, y1, x2, y2

    # Validación: no reducir demasiado
    orig_area = (x2 - x1) * (y2 - y1)
    new_area = area
    if new_area < 0.25 * orig_area:
        return x1, y1, x2, y2

    # Validación de aspect ratio (evitar bbox muy anchos/altos)
    new_w = gx2 - gx1
    new_h = gy2 - gy1
    aspect = new_w / max(new_h, 1)
    
    # Si el bbox es muy horizontal (>2.5:1) probablemente incluye texto lateral
    if aspect > 2.5:
        # Reducir ancho al 60% desde el lado derecho (donde suele estar texto)
        reduction = int(new_w * 0.4)
        gx2 = max(gx1 + int(new_w * 0.6), gx1 + 50)
    
    # Si es muy vertical (>3:1) podría incluir texto abajo
    if aspect < 0.4:
        # Reducir altura al 70% desde abajo
        reduction = int(new_h * 0.3)
        gy2 = max(gy1 + int(new_h * 0.7), gy1 + 50)

    return gx1, gy1, gx2, gy2

def crop_by_bbox_norm_fallback(img_bgr: np.ndarray, bbox: BBoxNorm, pad: float = 0.005) -> np.ndarray:
    """Fallback con padding ULTRA mínimo (0.5%)"""
    h, w = img_bgr.shape[:2]
    x1 = int((bbox.x1 - pad) * w)
    y1 = int((bbox.y1 - pad) * h)
    x2 = int((bbox.x2 + pad) * w)
    y2 = int((bbox.y2 + pad) * h)

    x1 = clamp(x1, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    x2 = clamp(x2, 1, w)
    y2 = clamp(y2, 1, h)

    if x2 <= x1 or y2 <= y1:
        return img_bgr.copy()
    return img_bgr[y1:y2, x1:x2].copy()

def crop_product_smart_v3(
    img_bgr: np.ndarray,
    bbox: BBoxNorm,
    pad_rel: float = 0.02,
    use_smart: bool = True,
    max_area_ratio: float = 0.55,
    exclude_text: bool = True
) -> np.ndarray:
    """
    V3: Recorte ultra-preciso que excluye texto lateral.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_norm_to_px(bbox, w, h, pad_rel=pad_rel)

    if not use_smart:
        return img_bgr[y1:y2, x1:x2].copy()

    try:
        rx1, ry1, rx2, ry2 = smart_refine_crop_v3(
            img_bgr, x1, y1, x2, y2,
            max_area_ratio=max_area_ratio,
            exclude_text_regions=exclude_text
        )
        
        if rx2 <= rx1 or ry2 <= ry1:
            return crop_by_bbox_norm_fallback(img_bgr, bbox)
        
        crop = img_bgr[ry1:ry2, rx1:rx2].copy()
        
        if crop.shape[0] < 50 or crop.shape[1] < 50:
            return crop_by_bbox_norm_fallback(img_bgr, bbox)
        
        return crop
    except Exception as e:
        print(f"[WARN] Smart crop V3 falló: {e}, usando fallback")
        return crop_by_bbox_norm_fallback(img_bgr, bbox)


# =========================
# Output rows (base)
# =========================

@dataclass
class CsvRow:
    pagina: str
    nombre: str
    variante: str
    precio: str
    imagen: str
    confianza: float
    pass2: int

def needs_pass2(p: ProductDetected, min_conf_two_pass: float, require_fields: bool = True) -> bool:
    if float(p.confianza) < min_conf_two_pass:
        return True
    if require_fields and (not norm_spaces(p.nombre) or not norm_spaces(p.precio)):
        return True
    return False


# =========================
# Enriquecimiento distintivo (batch)
# =========================

class EnrichedItem(BaseModel):
    imagen: str = Field(..., description="Nombre de archivo del crop")
    valid: bool = Field(..., description="true si es tarjeta de producto")
    nombre: str = Field(default="", description="Nombre del producto")
    distintivo: str = Field(default="", description="Rasgo único observable")
    variante: str = Field(default="", description="Variante/color/talla")
    precio: str = Field(default="", description="Precio")
    descripcion: str = Field(default="", description="Descripción corta")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confianza")

class EnrichBatch(BaseModel):
    items: List[EnrichedItem] = Field(default_factory=list)

def openai_enrich_crops_batch(
    client: OpenAI,
    model: str,
    crop_paths: List[str],
    crop_filenames: List[str],
    max_retries: int = 3
) -> Tuple[EnrichBatch, Dict[str, int]]:
    system = (
        "Eres un extractor y normalizador de productos en tarjetas (cards) de catálogo.\n"
        "Objetivo: MINIMIZAR NOMBRES DUPLICADOS cuando los productos cambian por alguna característica.\n\n"
        "Para cada crop devuelve:\n"
        "- valid: true SOLO si el recorte es una tarjeta de producto (foto + nombre y/o precio y/o botón comprar).\n"
        "- distintivo: OBLIGATORIO si existe algo que diferencie (elige 1-3 rasgos observables):\n"
        "  color, talla, pack (x2/x3), modelo, código/SKU, capacidad (ml/GB), estampado, tipo, versión.\n"
        "  Si NO hay texto distintivo, usa un descriptor visual corto y no inventado (ej: 'con capucha', 'sin mangas').\n"
        "- nombre: nombre limpio del producto e INYECTA el distintivo cuando aplique.\n"
        "  Formato requerido cuando aplique: 'Nombre base - <distintivo>' (ej: 'Polo deportivo - Negro').\n"
        "- variante: color/talla/modelo si se ve (no la dejes vacía si es visible).\n"
        "- precio: exactamente como aparece (ej: S/119). Si no se ve, vacío.\n"
        "- descripcion: 1 sola frase corta en español, sin inventar. Solo lo visible.\n"
        "- confidence: tu confianza (0..1).\n\n"
        "Reglas anti-ruido:\n"
        "- Si ves texto gigante vertical/lateral y NO es tarjeta, valid=false.\n"
        "- Si el recorte tiene dos productos completos, igual valid=true pero describe el principal.\n"
        "- NO inventes marca ni material si no se aprecia.\n"
        "Devuelve SOLO el schema."
    )

    content = [{"type": "input_text", "text": (
        "Procesa estos crops en el mismo orden. "
        "La clave de cada uno es su nombre de archivo. "
        "Devuelve items[] con imagen=<filename> para cada uno."
    )}]
    for p in crop_paths:
        content.append({"type": "input_image", "image_url": image_to_data_url(p), "detail": "high"})

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                text_format=EnrichBatch,
            )
            batch = resp.output_parsed

            got = {it.imagen: it for it in batch.items}
            fixed: List[EnrichedItem] = []
            for fn in crop_filenames:
                if fn in got:
                    fixed.append(got[fn])
                else:
                    fixed.append(EnrichedItem(
                        imagen=fn, valid=False, nombre="", distintivo="", variante="", precio="", descripcion="", confidence=0.0
                    ))
            return EnrichBatch(items=fixed), _usage_from_response(resp)
        except Exception as e:
            last_err = e
            time.sleep(0.8 * attempt)

    print(f"[WARN] Enrich batch failed: {last_err}")
    return EnrichBatch(items=[
        EnrichedItem(imagen=fn, valid=False, nombre="", distintivo="", variante="", precio="", descripcion="", confidence=0.0)
        for fn in crop_filenames
    ]), {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


# =========================
# CSV IO
# =========================

def read_csv_dicts(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows

def write_csv(path: str, fieldnames: List[str], rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


# =========================
# Grouping / Dedupe
# =========================

def make_group_key(nombre: str, precio: str, variante: str) -> str:
    n = re.sub(r"[^a-z0-9]+", "", safe_lower(nombre))
    p = re.sub(r"[^0-9/]+", "", safe_lower(precio))
    v = re.sub(r"[^a-z0-9]+", "", safe_lower(variante))
    if v:
        return f"{n}|{p}|{v}"
    return f"{n}|{p}"

def choose_best_in_group(items: List[Dict], crops_dir: str) -> Dict:
    best = None
    best_score = -1e18
    for it in items:
        conf = float(it.get("Confianza", "0") or 0)
        img_name = it.get("Imagen", "")
        sp = os.path.join(crops_dir, img_name)
        sharp = 0.0
        if os.path.exists(sp):
            bgr = cv2.imread(sp)
            sharp = laplacian_sharpness(bgr)
        score = (conf * 100.0) + (math.log1p(sharp) * 3.0)
        if score > best_score:
            best_score = score
            best = it
    return best or items[0]


# =========================
# Extraction (URL/screens -> base CSV)
# =========================

def process_screenshots_folder_openai(
    input_dir: str,
    out_dir: str,
    csv_name: str,
    model_page: str,
    two_pass: bool = False,
    model_crop: Optional[str] = None,
    min_conf_keep: float = 0.25,
    min_conf_two_pass: float = 0.45,
    max_pages: int = 0,
    max_openai_calls: int = 0,
    do_trim_blackbars: bool = True,
    trim_thr: int = 18,
    use_smart_crop: bool = True,
    smart_pad_rel: float = 0.02,
    smart_max_area_ratio: float = 0.55,
    exclude_text: bool = True,
) -> Dict[str, object]:
    ensure_dir(out_dir)
    crops_dir = os.path.join(out_dir, "crops")
    ensure_dir(crops_dir)

    client = OpenAI()
    openai_calls = 0
    usage_total: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    files.sort()
    if max_pages and max_pages > 0:
        files = files[:max_pages]

    rows: List[CsvRow] = []
    crop_idx = 1

    for fname in files:
        spath = os.path.join(input_dir, fname)

        img = normalize_screenshot_inplace(spath, do_trim_blackbars=do_trim_blackbars, thr=trim_thr)
        if img is None:
            continue

        if max_openai_calls and openai_calls >= max_openai_calls:
            print(f"[STOP] max_openai_calls alcanzado ({max_openai_calls}).")
            break

        page, usage = openai_page_detect(client, spath, model=model_page)
        openai_calls += 1
        for k in usage_total:
            usage_total[k] += usage.get(k, 0)

        products: List[ProductDetected] = []
        for p in page.productos:
            if float(p.confianza) < min_conf_keep:
                continue
            if (p.bbox.x2 - p.bbox.x1) < 0.05 or (p.bbox.y2 - p.bbox.y1) < 0.05:
                continue
            products.append(p)

        products.sort(key=lambda p: (p.bbox.y1, p.bbox.x1))

        for p in products:
            crop = crop_product_smart_v3(
                img_bgr=img,
                bbox=p.bbox,
                pad_rel=smart_pad_rel,
                use_smart=use_smart_crop,
                max_area_ratio=smart_max_area_ratio,
                exclude_text=exclude_text
            )

            crop_name = f"prod_{crop_idx:04d}.png"
            crop_path = os.path.join(crops_dir, crop_name)
            cv2.imwrite(crop_path, crop)

            nombre = norm_spaces(p.nombre)
            variante = norm_spaces(p.variante)
            precio = norm_price(p.precio)
            conf = float(p.confianza)
            did_pass2 = 0

            if two_pass and needs_pass2(p, min_conf_two_pass=min_conf_two_pass, require_fields=True):
                if max_openai_calls and openai_calls >= max_openai_calls:
                    print(f"[STOP] max_openai_calls alcanzado ({max_openai_calls}) en two-pass.")
                else:
                    refine_model = model_crop or model_page
                    refined, usage = openai_refine_crop(client, crop_path, model=refine_model)
                    openai_calls += 1
                    for k in usage_total:
                        usage_total[k] += usage.get(k, 0)
                    did_pass2 = 1

                    r_nombre = norm_spaces(refined.nombre)
                    r_var = norm_spaces(refined.variante)
                    r_precio = norm_price(refined.precio)

                    if r_nombre:
                        nombre = r_nombre
                    if r_var:
                        variante = r_var
                    if r_precio:
                        precio = r_precio

            rows.append(CsvRow(
                pagina=fname,
                nombre=nombre,
                variante=variante,
                precio=precio,
                imagen=crop_name,
                confianza=conf,
                pass2=did_pass2
            ))
            crop_idx += 1

        print(f"[OK] {fname}: detectados={len(page.productos)} usados={len(products)}")

    csv_path = os.path.join(out_dir, csv_name)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["Pagina", "Nombre", "Variante", "Precio", "Imagen", "Confianza", "TwoPass"])
        for r in rows:
            wr.writerow([r.pagina, r.nombre, r.variante, r.precio, r.imagen, f"{r.confianza:.2f}", r.pass2])

    print(f"\nOK -> CSV base: {csv_path}")
    print(f"OK -> Recortes: {crops_dir}")
    print(f"Total productos (base): {len(rows)}")
    print(f"OpenAI calls (extract): {openai_calls}")
    print(f"Tokens usados: input={usage_total['input_tokens']} output={usage_total['output_tokens']} total={usage_total['total_tokens']}")

    return {
        "csv_path": csv_path,
        "crops_dir": crops_dir,
        "openai_calls": openai_calls,
        "rows_count": len(rows),
        "usage": dict(usage_total),
    }

def process_url_to_csv_openai(
    url: str,
    out_dir: str,
    csv_name: str,
    model_page: str,
    two_pass: bool = False,
    model_crop: Optional[str] = None,
    max_pages: Optional[int] = None,
    dup_stop_limit: int = 4,
    hash_near_threshold: int = 0,
    min_conf_keep: float = 0.25,
    min_conf_two_pass: float = 0.45,
    max_openai_calls: int = 0,
    do_trim_blackbars: bool = True,
    trim_thr: int = 18,
    use_smart_crop: bool = True,
    smart_pad_rel: float = 0.02,
    smart_max_area_ratio: float = 0.55,
    exclude_text: bool = True,
) -> Dict[str, object]:
    screens_dir = os.path.join(out_dir, "screens")
    ensure_dir(out_dir)
    ensure_dir(screens_dir)

    _ = fullpage_screenshots(
        url=url,
        out_dir=screens_dir,
        wait_sec=4.0,
        max_pages=max_pages,
        dup_stop_limit=dup_stop_limit,
        hash_near_threshold=hash_near_threshold,
        do_trim_blackbars=do_trim_blackbars,
        trim_thr=trim_thr,
    )

    return process_screenshots_folder_openai(
        input_dir=screens_dir,
        out_dir=out_dir,
        csv_name=csv_name,
        model_page=model_page,
        two_pass=two_pass,
        model_crop=model_crop,
        min_conf_keep=min_conf_keep,
        min_conf_two_pass=min_conf_two_pass,
        max_pages=0,
        max_openai_calls=max_openai_calls,
        do_trim_blackbars=do_trim_blackbars,
        trim_thr=trim_thr,
        use_smart_crop=use_smart_crop,
        smart_pad_rel=smart_pad_rel,
        smart_max_area_ratio=smart_max_area_ratio,
        exclude_text=exclude_text,
    )


# =========================
# Enrich stage
# =========================

def enrich_from_out_dir(
    out_dir: str,
    csv_in_name: str,
    csv_out_name: str,
    model_enrich: str,
    batch_size: int = 6,
    max_enrich_calls: int = 50,
    min_valid_confidence: float = 0.55,
    max_items: int = 0,
    dedupe: bool = False,
) -> Dict[str, object]:
    csv_in = os.path.join(out_dir, csv_in_name)
    crops_dir = os.path.join(out_dir, "crops")
    csv_out = os.path.join(out_dir, csv_out_name)

    if not os.path.exists(csv_in):
        raise SystemExit(f"No existe: {csv_in}")
    if not os.path.isdir(crops_dir):
        raise SystemExit(f"No existe carpeta crops/: {crops_dir}")

    base_rows = read_csv_dicts(csv_in)

    work = []
    for r in base_rows:
        img = (r.get("Imagen") or "").strip()
        if not img:
            continue
        p = os.path.join(crops_dir, img)
        if os.path.exists(p):
            work.append((r, p, img))

    if max_items and max_items > 0:
        work = work[:max_items]

    if not work:
        raise SystemExit("No encontré crops para procesar.")

    client = OpenAI()
    batch_size = max(1, min(12, int(batch_size)))
    max_calls = max(1, int(max_enrich_calls))

    enriched_rows: List[Dict] = []
    openai_calls = 0

    seen_names: Dict[str, int] = {}

    for i in range(0, len(work), batch_size):
        if openai_calls >= max_calls:
            break

        chunk = work[i:i + batch_size]
        crop_paths = [c[1] for c in chunk]
        crop_fns = [c[2] for c in chunk]

        batch, _ = openai_enrich_crops_batch(
            client=client,
            model=model_enrich,
            crop_paths=crop_paths,
            crop_filenames=crop_fns
        )
        openai_calls += 1

        for (orig, _, fn), it in zip(chunk, batch.items):
            base_nombre = norm_spaces(orig.get("Nombre", ""))
            base_var = norm_spaces(orig.get("Variante", ""))
            base_precio = norm_price(orig.get("Precio", ""))

            if it.valid:
                nombre2 = norm_spaces(it.nombre) if it.nombre else base_nombre
                var2 = norm_spaces(it.variante) if it.variante else base_var
                precio2 = norm_price(it.precio) if it.precio else base_precio

                dist = norm_spaces(getattr(it, "distintivo", ""))
                if dist and dist.lower() not in nombre2.lower():
                    if not var2 or dist.lower() != var2.lower():
                        nombre2 = f"{nombre2} - {dist}"
            else:
                nombre2 = base_nombre
                var2 = base_var
                precio2 = base_precio

            key = re.sub(r"\s+", " ", (nombre2 or "").strip().lower())
            if key:
                seen_names[key] = seen_names.get(key, 0) + 1
                if seen_names[key] > 1:
                    nombre2 = f"{nombre2} #{file_suffix(fn)}"

            group_id = make_group_key(nombre2, precio2, var2) if it.valid else ""

            lowq = 0
            if it.valid and float(it.confidence) < float(min_valid_confidence):
                lowq = 1

            row = {
                "Pagina": orig.get("Pagina", ""),
                "Nombre": nombre2,
                "Variante": var2,
                "Precio": precio2,
                "Descripcion": norm_spaces(it.descripcion) if it.valid else "",
                "Imagen": fn,
                "Confianza": orig.get("Confianza", ""),
                "TwoPass": orig.get("TwoPass", ""),
                "ValidProduct": 1 if it.valid else 0,
                "EnrichConfidence": f"{float(it.confidence):.2f}",
                "LowQuality": lowq,
                "GroupId": group_id,
            }
            enriched_rows.append(row)

    if openai_calls >= max_calls:
        processed = min(len(work), ((openai_calls) * batch_size))
        for j in range(processed, len(work)):
            orig, _, fn = work[j]
            nombre_fallback = norm_spaces(orig.get("Nombre", ""))
            key = re.sub(r"\s+", " ", (nombre_fallback or "").strip().lower())
            if key:
                seen_names[key] = seen_names.get(key, 0) + 1
                if seen_names[key] > 1:
                    nombre_fallback = f"{nombre_fallback} #{file_suffix(fn)}"

            enriched_rows.append({
                "Pagina": orig.get("Pagina", ""),
                "Nombre": nombre_fallback,
                "Variante": norm_spaces(orig.get("Variante", "")),
                "Precio": norm_price(orig.get("Precio", "")),
                "Descripcion": "",
                "Imagen": fn,
                "Confianza": orig.get("Confianza", ""),
                "TwoPass": orig.get("TwoPass", ""),
                "ValidProduct": 0,
                "EnrichConfidence": "0.00",
                "LowQuality": 1,
                "GroupId": "",
            })

    final_rows = enriched_rows
    if dedupe:
        groups: Dict[str, List[Dict]] = {}
        others: List[Dict] = []
        for r in enriched_rows:
            if r.get("ValidProduct") == 1 and r.get("GroupId"):
                groups.setdefault(r["GroupId"], []).append(r)
            else:
                others.append(r)

        picked: List[Dict] = []
        for gid, items in groups.items():
            best = choose_best_in_group(items, crops_dir=crops_dir)
            best2 = dict(best)
            best2["GroupSize"] = len(items)
            picked.append(best2)

        picked.sort(key=lambda x: (x.get("Pagina", ""), x.get("Nombre", ""), x.get("Variante", "")))
        others.sort(key=lambda x: (x.get("Pagina", ""), x.get("Nombre", ""), x.get("Variante", "")))

        final_rows = picked + others

    fieldnames = [
        "Pagina", "Nombre", "Variante", "Precio", "Descripcion",
        "Imagen", "Confianza", "TwoPass",
        "ValidProduct", "EnrichConfidence", "LowQuality", "GroupId"
    ]
    if dedupe and any("GroupSize" in r for r in final_rows):
        fieldnames.append("GroupSize")

    write_csv(csv_out, fieldnames, final_rows)

    print(f"OK -> CSV enriquecido: {csv_out}")
    print(f"OpenAI calls (enrich): {openai_calls}/{max_calls}")
    print(f"Filas salida: {len(final_rows)}")

    return {
        "csv_out": csv_out,
        "openai_calls": openai_calls,
        "rows_count": len(final_rows),
    }


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Canva Scraper V3 - Ultra-preciso (sin texto lateral)")

    ap.add_argument("--url", default="", help="URL de Canva Site")
    ap.add_argument("--input", default="", help="Carpeta con screenshots")
    ap.add_argument("--out", default="salida_v3", help="Carpeta de salida")
    ap.add_argument("--csv", default="productos.csv", help="CSV base")

    ap.add_argument("--model-page", default="gpt-4o", help="Modelo Pass1")
    ap.add_argument("--model-crop", default="gpt-4o-mini", help="Modelo Pass2")
    ap.add_argument("--two-pass", action="store_true", help="Activar Pass2")

    ap.add_argument("--max-pages", type=int, default=0, help="Límite páginas")
    ap.add_argument("--dup-stop", type=int, default=4, help="Stop duplicados")
    ap.add_argument("--hash-near", type=int, default=0, help="Tolerancia hash")

    ap.add_argument("--max-openai-calls", type=int, default=0, help="Límite OpenAI")

    ap.add_argument("--min-conf-keep", type=float, default=0.25, help="Confianza mínima")
    ap.add_argument("--min-conf-two-pass", type=float, default=0.45, help="Umbral Pass2")

    ap.add_argument("--no-trim", action="store_true", help="Desactivar trim")
    ap.add_argument("--trim-thr", type=int, default=18, help="Umbral negro")

    ap.add_argument("--no-smart-crop", action="store_true", help="Desactivar Smart Crop")
    ap.add_argument("--no-exclude-text", action="store_true", help="Desactivar exclusión de texto")
    ap.add_argument("--smart-pad-rel", type=float, default=0.02, help="Padding bbox")
    ap.add_argument("--smart-max-area", type=float, default=0.55, help="Área máxima")

    ap.add_argument("--enrich", action="store_true", help="Enriquecer")
    ap.add_argument("--skip-extract", action="store_true", help="Omitir extracción")
    ap.add_argument("--model-enrich", default="gpt-4o-mini", help="Modelo enrich")
    ap.add_argument("--batch-size", type=int, default=6, help="Batch size")
    ap.add_argument("--max-enrich-calls", type=int, default=50, help="Límite enrich")
    ap.add_argument("--min-valid-confidence", type=float, default=0.55, help="Umbral LowQuality")
    ap.add_argument("--max-items", type=int, default=0, help="Límite items")
    ap.add_argument("--write", default="productos_enriquecidos.csv", help="CSV enrich")
    ap.add_argument("--dedupe", action="store_true", help="Deduplicar")

    args = ap.parse_args()

    model_crop = args.model_crop.strip() or None
    do_trim_blackbars = not args.no_trim
    use_smart_crop = not args.no_smart_crop
    exclude_text = not args.no_exclude_text

    print("\n" + "="*70)
    print("CANVA SCRAPER V3.0 - ULTRA PRECISO")
    print("="*70)
    print(f"Smart Crop: {'✓ ON' if use_smart_crop else '✗ OFF'}")
    print(f"Exclusión texto: {'✓ ON' if exclude_text else '✗ OFF'}")
    print(f"Padding: {args.smart_pad_rel*100:.1f}%")
    print(f"Área máx: {args.smart_max_area*100:.1f}%")
    print("="*70 + "\n")

    if not args.skip_extract:
        if args.url.strip():
            process_url_to_csv_openai(
                url=args.url.strip(),
                out_dir=args.out,
                csv_name=args.csv,
                model_page=args.model_page,
                two_pass=args.two_pass,
                model_crop=model_crop,
                max_pages=(None if args.max_pages == 0 else args.max_pages),
                dup_stop_limit=args.dup_stop,
                hash_near_threshold=args.hash_near,
                min_conf_keep=args.min_conf_keep,
                min_conf_two_pass=args.min_conf_two_pass,
                max_openai_calls=args.max_openai_calls,
                do_trim_blackbars=do_trim_blackbars,
                trim_thr=args.trim_thr,
                use_smart_crop=use_smart_crop,
                smart_pad_rel=args.smart_pad_rel,
                smart_max_area_ratio=args.smart_max_area,
                exclude_text=exclude_text,
            )
        elif args.input.strip():
            process_screenshots_folder_openai(
                input_dir=args.input.strip(),
                out_dir=args.out,
                csv_name=args.csv,
                model_page=args.model_page,
                two_pass=args.two_pass,
                model_crop=model_crop,
                min_conf_keep=args.min_conf_keep,
                min_conf_two_pass=args.min_conf_two_pass,
                max_pages=0,
                max_openai_calls=args.max_openai_calls,
                do_trim_blackbars=do_trim_blackbars,
                trim_thr=args.trim_thr,
                use_smart_crop=use_smart_crop,
                smart_pad_rel=args.smart_pad_rel,
                smart_max_area_ratio=args.smart_max_area,
                exclude_text=exclude_text,
            )
        else:
            if not args.enrich:
                raise SystemExit("Usa --url o --input")

    if args.enrich:
        enrich_from_out_dir(
            out_dir=args.out,
            csv_in_name=args.csv,
            csv_out_name=args.write,
            model_enrich=args.model_enrich,
            batch_size=args.batch_size,
            max_enrich_calls=args.max_enrich_calls,
            min_valid_confidence=args.min_valid_confidence,
            max_items=args.max_items,
            dedupe=args.dedupe,
        )


if __name__ == "__main__":
    main()