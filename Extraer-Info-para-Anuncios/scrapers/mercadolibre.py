"""Scraper de resultados de búsqueda de MercadoLibre Perú."""

import re
import logging
import httpx
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

_BASE_URL = "https://listado.mercadolibre.com.pe"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.6478.127 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "es-PE,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "sec-ch-ua": '"Chromium";v="126", "Google Chrome";v="126", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
}


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _normalize_image(url: str) -> str:
    """Convierte thumbnail de MercadoLibre a mejor resolución."""
    url = url.strip().split("?")[0]
    # MercadoLibre usa -I.jpg para alta res, -O.jpg para thumb
    url = re.sub(r"-[A-Z]\.jpg$", "-I.jpg", url)
    return url


def _is_product_image(url: str) -> bool:
    low = url.lower()
    if not low.startswith("http"):
        return False
    if any(x in low for x in ["sprite", "icon", "nav", "logo", "pixel", "transparent"]):
        return False
    return "http2.mlstatic.com" in low or "mlstatic.com" in low or "mercadolibre" in low


def _extract_result(el: BeautifulSoup) -> dict | None:
    """Extrae datos de un resultado de búsqueda de MercadoLibre."""
    # Título
    title_el = (
        el.select_one("h2.ui-search-item__title")
        or el.select_one("a.ui-search-item__group__element h2")
        or el.select_one("[class*='item__title']")
    )
    title = _clean(title_el.get_text()) if title_el else ""
    if not title:
        return None

    # URL
    link = (
        el.select_one("a.ui-search-link")
        or el.select_one("a.ui-search-item__group__element")
        or el.select_one("a[href*='mercadolibre']")
    )
    href = link.get("href", "") if link else ""
    # Limpiar tracking params
    if href and "#" in href:
        href = href.split("#")[0]
    url = href

    # Precio
    price = ""
    price_el = el.select_one("span.andes-money-amount__fraction")
    if price_el:
        price_text = _clean(price_el.get_text())
        if price_text:
            # Verificar moneda
            currency_el = el.select_one("span.andes-money-amount__currency-symbol")
            currency = _clean(currency_el.get_text()) if currency_el else "S/"
            price = f"{currency} {price_text}"

    # Imagen
    image_url = None
    img = (
        el.select_one("img.ui-search-result-image__element")
        or el.select_one("img[data-src*='mlstatic']")
        or el.select_one("img[src*='mlstatic']")
    )
    if img:
        src = img.get("data-src") or img.get("src") or ""
        if src and _is_product_image(src):
            image_url = _normalize_image(src)

    # Descripción: atributos y detalles
    description = ""
    desc_selectors = [
        ".ui-search-item__group--attributes",
        "[class*='item__subtitle']",
        "[class*='item__details']",
        "[class*='item__variations']",
    ]
    for sel in desc_selectors:
        desc_el = el.select_one(sel)
        if desc_el:
            desc_text = _clean(desc_el.get_text())
            if desc_text and len(desc_text) > 5:
                description = desc_text
                break
    if not description:
        description = title

    # Features: badges de envío, estado, etc.
    features: list[str] = []
    tag_selectors = [
        ".ui-search-item__highlight-label",
        "[class*='fulfillment']",
        "[class*='shipping']",
        ".ui-search-reviews__amount",
    ]
    for sel in tag_selectors:
        tag_els = el.select(sel)
        for tag in tag_els[:2]:
            t = _clean(tag.get_text())
            if t and len(t) > 3:
                features.append(t)

    return {
        "title": title,
        "url": url,
        "price": price,
        "image_url": image_url,
        "description": description,
        "features": features[:3],
    }


async def search_mercadolibre(
    product_name: str,
    timeout: float = 15.0,
    max_results: int = 3,
) -> dict:
    """Busca un producto en MercadoLibre Perú."""
    result = {"images": [], "texts": [], "videos": []}

    # MercadoLibre usa el nombre del producto en la URL separado por guiones
    search_term = quote_plus(product_name.strip())
    search_url = f"{_BASE_URL}/{search_term}"

    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=timeout,
        ) as client:
            resp = await client.get(search_url)
            resp.raise_for_status()
    except Exception as e:
        logger.warning("MercadoLibre HTTP error for '%s': %s", product_name, e)
        return result

    soup = BeautifulSoup(resp.text, "html.parser")

    # Selectores de resultados de MercadoLibre
    items = soup.select("li.ui-search-layout__item")
    if not items:
        items = soup.select("[class*='ui-search-layout'] > li")
    if not items:
        items = soup.select("ol.ui-search-layout > li")

    logger.info(
        "MercadoLibre: %d results for '%s' (status=%d, body=%d chars)",
        len(items), product_name, resp.status_code, len(resp.text),
    )

    seen_images: set[str] = set()
    for item in items:
        if len(result["texts"]) >= max_results:
            break

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
