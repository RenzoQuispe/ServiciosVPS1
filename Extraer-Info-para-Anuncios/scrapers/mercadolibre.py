"""
Scraper de resultados de búsqueda — Falabella Perú.

Reemplaza al scraper original de MercadoLibre que ya no funciona
por bloqueo anti-bot (JS challenge).  Se mantiene el nombre del
módulo para no romper imports existentes.
"""

import re
import logging
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.falabella.com.pe/falabella-pe/search"
_BASE_URL = "https://www.falabella.com.pe"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.6478.127 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-PE,es;q=0.9,en;q=0.8",
    "sec-ch-ua": '"Chromium";v="126", "Google Chrome";v="126"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "upgrade-insecure-requests": "1",
}


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _normalize_image(url: str) -> str:
    """Intenta obtener la imagen en mejor resolución."""
    url = url.strip().split("?")[0]
    # Falabella usa /width=xxx/ en la URL — quitamos ese resize
    url = re.sub(r"/widt[^/]*/", "/", url)
    return url


def _extract_result(el: BeautifulSoup) -> dict | None:
    """Extrae datos de un pod de Falabella."""
    # Título
    title_el = el.select_one("b.pod-subTitle") or el.select_one("[class*='subTitle']")
    if not title_el:
        title_el = el.select_one("b")
    title = _clean(title_el.get_text()) if title_el else ""
    if not title:
        return None

    # URL — construir desde el testId o data-key
    pod_link = el.select_one("a.pod-link") or el.select_one("a[id*='pod']")
    href = ""
    if pod_link:
        href = pod_link.get("href", "")
        if not href:
            # Construir URL desde el id del pod
            pod_id = pod_link.get("id", "")
            # testId-pod-116992404 → /falabella-pe/product/116992404
            match = re.search(r"(\d{6,})", pod_id)
            if match:
                href = f"/falabella-pe/product/{match.group(1)}"
    url = f"{_BASE_URL}{href}" if href and not href.startswith("http") else href

    # Precio
    price_el = el.select_one("[class*='price']")
    price_text = _clean(price_el.get_text()) if price_el else ""
    price = ""
    if price_text:
        # Extraer el primer precio (formato: "S/  148   -54%S/  319")
        match = re.search(r"S/\s*([\d,.]+)", price_text)
        if match:
            price = f"S/ {match.group(1)}"

    # Imagen
    img = el.select_one("img[src*='falabella']") or el.select_one("img[src*='media']") or el.select_one("img")
    image_url = None
    if img:
        src = img.get("src") or img.get("data-src") or ""
        if src and src.startswith("http"):
            image_url = _normalize_image(src)

    # Descripción: intentar extraer texto descriptivo del pod
    description = ""
    desc_selectors = [
        "[class*='pod-details']",
        "[class*='description']",
        "[class*='brandName']",
        "[class*='pod-body']",
    ]
    for sel in desc_selectors:
        desc_el = el.select_one(sel)
        if desc_el:
            desc_text = _clean(desc_el.get_text())
            if desc_text and len(desc_text) > 5:
                description = desc_text
                break

    # Si no se encontró descripción, usar el título
    if not description:
        description = title

    features: list[str] = []

    # Badges y etiquetas
    badge_els = el.select("[class*='badge']")
    for badge in badge_els[:3]:
        t = _clean(badge.get_text())
        if t and len(t) > 3 and "%" not in t:
            features.append(t)

    return {
        "title": title,
        "url": url,
        "price": price,
        "image_url": image_url,
        "description": description,
        "features": features,
    }


async def search_mercadolibre(
    product_name: str,
    timeout: float = 15.0,
    max_results: int = 3,
) -> dict:
    """
    Busca un producto en Falabella Perú.
    Mantiene el nombre search_mercadolibre para compatibilidad.
    """
    result = {"images": [], "texts": [], "videos": []}

    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=timeout,
        ) as client:
            resp = await client.get(
                _SEARCH_URL,
                params={"Ntt": product_name},
            )
            resp.raise_for_status()
    except Exception as e:
        logger.warning("Falabella HTTP error for '%s': %s", product_name, e)
        return result

    soup = BeautifulSoup(resp.text, "html.parser")

    # Buscar pods de producto
    items = soup.select("[class*='grid-pod']")
    if not items:
        items = soup.select("[class*='pod-4_GRID']")
    if not items:
        items = soup.select("div.pod")

    logger.info(
        "Falabella: %d results for '%s' (status=%d, body=%d chars)",
        len(items), product_name, resp.status_code, len(resp.text),
    )

    seen_images: set[str] = set()
    for item in items:
        if len(result["texts"]) >= max_results:
            break

        # Saltar patrocinados marcados como sponsored
        if item.get("data-sponsored") == "true":
            continue

        data = _extract_result(item)
        if not data:
            continue

        result["texts"].append({
            "title": data["title"],
            "description": data.get("description", ""),
            "features": data.get("features", []),
            "price": data["price"],
            "url": data["url"],
        })

        if data["image_url"] and data["image_url"] not in seen_images:
            seen_images.add(data["image_url"])
            result["images"].append({
                "url": data["image_url"],
                "alt": data["title"],
                "source_url": data["url"],
            })

    return result
