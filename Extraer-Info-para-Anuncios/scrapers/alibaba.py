"""
Scraper de resultados de búsqueda — Amazon México (.com.mx).

Reemplaza al scraper original de Alibaba que ya no funciona
(contenido renderizado por JavaScript, no disponible en HTML estático).
Se mantiene el nombre del módulo para no romper imports existentes.
Usa Amazon México como fuente alternativa de productos en español.
"""

import re
import logging
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.amazon.com.mx/s"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.6478.127 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "es-MX,es;q=0.9,en;q=0.7",
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
    """Convierte thumbnail de Amazon a alta resolución."""
    url = url.strip().split("?")[0]
    url = re.sub(r"\._[^./]+_\.", ".", url)
    return url


def _is_product_image(url: str) -> bool:
    low = url.lower()
    if not low.startswith("http"):
        return False
    if any(x in low for x in ["sprite", "icon", "nav", "logo", "pixel", "transparent"]):
        return False
    return "images" in low or "m.media-amazon" in low


def _extract_result(el: BeautifulSoup) -> dict | None:
    """Extrae datos de un resultado de búsqueda de Amazon México."""
    title_el = el.select_one("h2 a span") or el.select_one("h2 span")
    title = _clean(title_el.get_text()) if title_el else ""
    if not title:
        return None

    # URL: intentar h2 a, luego links con /dp/, luego construir desde data-asin
    link = el.select_one("h2 a")
    href = link.get("href", "") if link else ""
    if not href:
        dp_link = el.select_one('a[href*="/dp/"]')
        href = dp_link.get("href", "") if dp_link else ""
    if not href:
        asin = el.get("data-asin", "")
        if asin:
            href = f"/dp/{asin}"
    url = f"https://www.amazon.com.mx{href}" if href.startswith("/") else href

    price_whole = el.select_one("span.a-price-whole")
    price_frac = el.select_one("span.a-price-fraction")
    price = ""
    if price_whole:
        whole = _clean(price_whole.get_text()).rstrip(".")
        frac = _clean(price_frac.get_text()) if price_frac else "00"
        price = f"MX${whole}.{frac}"

    img = el.select_one("img.s-image")
    image_url = None
    if img:
        src = img.get("src", "")
        if _is_product_image(src):
            image_url = _normalize_image(src)

    description = ""
    desc_el = el.select_one(".a-size-base-plus.a-color-base.a-text-normal")
    if not desc_el:
        desc_el = el.select_one(".a-size-medium.a-color-base")
    if desc_el:
        desc_text = _clean(desc_el.get_text())
        if desc_text and desc_text.lower()[:40] != title.lower()[:40]:
            description = desc_text

    features = []
    bullets = el.select("span.a-text-bold + span, .a-row .a-size-base, .a-color-secondary .a-size-base")
    for b in bullets[:4]:
        t = _clean(b.get_text())
        if t and len(t) > 5 and t.lower() != title.lower()[:len(t)]:
            features.append(t)

    return {
        "title": title,
        "url": url,
        "price": price,
        "image_url": image_url,
        "description": description,
        "features": features[:3],
    }


async def search_alibaba(
    product_name: str,
    timeout: float = 15.0,
    max_results: int = 3,
) -> dict:
    """
    Busca un producto en Amazon México.
    Mantiene el nombre search_alibaba para compatibilidad.
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
                params={"k": product_name},
            )
            resp.raise_for_status()
    except Exception as e:
        logger.warning("Amazon MX HTTP error for '%s': %s", product_name, e)
        return result

    soup = BeautifulSoup(resp.text, "html.parser")

    items = soup.select('div[data-component-type="s-search-result"]')
    logger.info(
        "Amazon MX: %d results for '%s' (status=%d, body=%d chars)",
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
            "features": data["features"],
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
