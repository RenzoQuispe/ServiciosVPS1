"""Scraper de resultados de búsqueda de Amazon."""

import re
import httpx
from bs4 import BeautifulSoup

_SEARCH_URL = "https://www.amazon.com/s"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-ES,es;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
}


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _normalize_image(url: str) -> str:
    """Convierte thumbnail de Amazon a alta resolución."""
    url = url.strip().split("?")[0]
    # Remover sufijos de tamaño como ._AC_UL320_.jpg
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
    """Extrae datos de un resultado de búsqueda de Amazon."""
    # Título
    title_el = el.select_one("h2 a span") or el.select_one("h2 span")
    title = _clean(title_el.get_text()) if title_el else ""
    if not title:
        return None

    # URL
    link = el.select_one("h2 a")
    href = link.get("href", "") if link else ""
    url = f"https://www.amazon.com{href}" if href.startswith("/") else href

    # Precio
    price_whole = el.select_one("span.a-price-whole")
    price_frac = el.select_one("span.a-price-fraction")
    price = ""
    if price_whole:
        whole = _clean(price_whole.get_text()).rstrip(".")
        frac = _clean(price_frac.get_text()) if price_frac else "00"
        price = f"US${whole}.{frac}"

    # Imagen
    img = el.select_one("img.s-image")
    image_url = None
    if img:
        src = img.get("src", "")
        if _is_product_image(src):
            image_url = _normalize_image(src)

    # Descripción breve (línea debajo del título en resultados)
    description = ""
    desc_el = el.select_one(".a-size-base-plus.a-color-base.a-text-normal")
    if not desc_el:
        desc_el = el.select_one(".a-size-medium.a-color-base")
    if desc_el:
        desc_text = _clean(desc_el.get_text())
        # Solo usar si es diferente al título
        if desc_text and desc_text.lower()[:40] != title.lower()[:40]:
            description = desc_text

    # Features / bullets visibles en el listado
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


async def search_amazon(
    product_name: str,
    timeout: float = 15.0,
    max_results: int = 3,
) -> dict:
    """
    Busca un producto en Amazon y retorna imágenes y textos.

    Returns:
        {
            "images": [{"url": str, "alt": str}],
            "texts": [{"title": str, "description": str, "features": [], "price": str, "url": str}],
            "videos": []
        }
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
                params={"k": product_name, "language": "es"},
            )
            resp.raise_for_status()
    except Exception:
        return result

    soup = BeautifulSoup(resp.text, "html.parser")

    # Resultados principales
    items = soup.select('div[data-component-type="s-search-result"]')

    seen_images: set[str] = set()
    for item in items:
        if len(result["texts"]) >= max_results:
            break

        # Saltar resultados patrocinados sin info útil
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
