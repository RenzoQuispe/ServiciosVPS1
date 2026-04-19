"""Scraper de resultados de búsqueda de MercadoLibre."""

import re
import httpx
from bs4 import BeautifulSoup

_SEARCH_URL = "https://listado.mercadolibre.com.pe/{query}"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-PE,es;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _build_search_url(product_name: str) -> str:
    query = product_name.strip().replace(" ", "-")
    return _SEARCH_URL.format(query=query)


def _extract_image(el: BeautifulSoup) -> str | None:
    """Extrae URL de imagen de alta resolución de un resultado."""
    img = el.select_one("img.poly-component__picture")
    if not img:
        img = el.select_one("img")
    if not img:
        return None

    src = img.get("data-src") or img.get("src") or ""
    if not src.startswith("http"):
        return None

    # MercadoLibre usa sufijos de tamaño: I.jpg -> O.jpg (original)
    src = re.sub(r"-[A-Z]\.jpg", "-O.jpg", src)
    return src


def _extract_result(el: BeautifulSoup) -> dict | None:
    """Extrae datos de un resultado de búsqueda individual."""
    # Título
    title_el = el.select_one("h2.poly-component__title a")
    if not title_el:
        title_el = el.select_one("a.poly-component__title")
    if not title_el:
        title_el = el.select_one("h2 a")

    title = _clean(title_el.get_text()) if title_el else ""
    url = title_el.get("href", "") if title_el else ""
    if not title or not url:
        return None

    # Precio
    price_el = el.select_one("span.andes-money-amount__fraction")
    price = _clean(price_el.get_text()) if price_el else ""
    currency_el = el.select_one("span.andes-money-amount__currency-symbol")
    currency = _clean(currency_el.get_text()) if currency_el else "S/"

    # Imagen
    image_url = _extract_image(el)

    # Descripción / atributos visibles en el listado
    description = ""
    desc_el = el.select_one("p.poly-component__description")
    if not desc_el:
        desc_el = el.select_one(".ui-search-item__group__element--details")
    if desc_el:
        description = _clean(desc_el.get_text())

    # Atributos destacados (envío gratis, vendedor, etc.)
    features: list[str] = []
    attr_els = el.select("li.poly-component__highlight, span.poly-component__shipped-from")
    for attr_el in attr_els[:3]:
        t = _clean(attr_el.get_text())
        if t and len(t) > 3:
            features.append(t)

    return {
        "title": title,
        "url": url,
        "price": f"{currency} {price}" if price else "",
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
    Busca un producto en MercadoLibre y retorna imágenes y textos.

    Returns:
        {
            "images": [{"url": str, "alt": str}],
            "texts": [{"title": str, "description": str, "features": [], "price": str, "url": str}],
            "videos": []
        }
    """
    result = {"images": [], "texts": [], "videos": []}
    search_url = _build_search_url(product_name)

    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=timeout,
        ) as client:
            resp = await client.get(search_url)
            resp.raise_for_status()
    except Exception:
        return result

    soup = BeautifulSoup(resp.text, "html.parser")

    # Buscar items de resultado
    items = soup.select("li.ui-search-layout__item")
    if not items:
        items = soup.select("div.poly-card")
    if not items:
        items = soup.select("ol.ui-search-layout li")

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
