"""Scraper de resultados de búsqueda de Alibaba.com."""

import re
import logging
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.alibaba.com/trade/search"
_BASE_URL = "https://www.alibaba.com"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.6478.127 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
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
    """Convierte thumbnail de Alibaba a mejor resolución."""
    url = url.strip().split("?")[0]
    # Alibaba usa _50x50.jpg o _220x220.jpg — intentar quitar el resize
    url = re.sub(r"_\d+x\d+\.", ".", url)
    if url.startswith("//"):
        url = "https:" + url
    return url


def _is_product_image(url: str) -> bool:
    low = url.lower()
    if not any(low.startswith(p) for p in ["http", "//"]):
        return False
    if any(x in low for x in ["sprite", "icon", "nav", "logo", "pixel", "transparent", "svg"]):
        return False
    return True


def _extract_result(el: BeautifulSoup) -> dict | None:
    """Extrae datos de un resultado de búsqueda de Alibaba."""
    # Título: buscar en múltiples selectores comunes de Alibaba
    title_el = (
        el.select_one("h2.search-card-e-title a")
        or el.select_one("[class*='title'] a")
        or el.select_one("h2 a")
        or el.select_one("a[title]")
    )
    title = ""
    href = ""
    if title_el:
        title = _clean(title_el.get("title") or title_el.get_text())
        href = title_el.get("href", "")
    if not title:
        return None

    # URL
    if href and not href.startswith("http"):
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = _BASE_URL + href
    url = href

    # Precio
    price = ""
    price_el = (
        el.select_one("[class*='price']")
        or el.select_one("[class*='Price']")
    )
    if price_el:
        price_text = _clean(price_el.get_text())
        # Extraer precio en formato US$ o $
        match = re.search(r"\$\s*([\d,.]+)", price_text)
        if match:
            price = f"US${match.group(1)}"

    # Imagen
    image_url = None
    img = el.select_one("img[src], img[data-src]")
    if img:
        src = img.get("src") or img.get("data-src") or ""
        if src and _is_product_image(src):
            image_url = _normalize_image(src)

    # Descripción
    description = ""
    desc_selectors = [
        "[class*='description']",
        "[class*='attr']",
        "[class*='info']",
    ]
    for sel in desc_selectors:
        desc_el = el.select_one(sel)
        if desc_el:
            desc_text = _clean(desc_el.get_text())
            if desc_text and len(desc_text) > 10 and desc_text != title:
                description = desc_text
                break
    if not description:
        description = title

    # Features
    features: list[str] = []
    tag_els = el.select("[class*='tag'], [class*='attr'] span")
    for tag in tag_els[:4]:
        t = _clean(tag.get_text())
        if t and len(t) > 3 and t.lower() != title.lower()[:len(t)]:
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
    """Busca un producto en Alibaba.com."""
    result = {"images": [], "texts": [], "videos": []}

    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=timeout,
        ) as client:
            resp = await client.get(
                _SEARCH_URL,
                params={"SearchText": product_name},
            )
            resp.raise_for_status()
    except Exception as e:
        logger.warning("Alibaba HTTP error for '%s': %s", product_name, e)
        return result

    soup = BeautifulSoup(resp.text, "html.parser")

    # Selectores para los cards de producto de Alibaba
    items = soup.select("[class*='organic-list'] [class*='list-card']")
    if not items:
        items = soup.select("[class*='search-card-e']")
    if not items:
        items = soup.select("[class*='J-offer-wrapper']")
    if not items:
        # Fallback genérico: cualquier div con data-content
        items = soup.select("div[data-content]")

    logger.info(
        "Alibaba: %d results for '%s' (status=%d, body=%d chars)",
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
