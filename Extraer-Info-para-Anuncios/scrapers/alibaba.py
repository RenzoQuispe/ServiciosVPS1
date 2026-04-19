"""Scraper de resultados de búsqueda de Alibaba."""

import re
import httpx
from bs4 import BeautifulSoup

_SEARCH_URL = "https://www.alibaba.com/trade/search"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _normalize_image(url: str) -> str:
    """Convierte thumbnail de Alibaba a mejor resolución."""
    url = url.strip().split("?")[0]
    # Remover sufijos de resize como _50x50.jpg, _220x220.jpg
    url = re.sub(r"_\d+x\d+\.", ".", url)
    if url.startswith("//"):
        url = "https:" + url
    return url


def _extract_result(el: BeautifulSoup) -> dict | None:
    """Extrae datos de un resultado de búsqueda de Alibaba."""
    # Título
    title_el = el.select_one("h2.search-card-e-title a")
    if not title_el:
        title_el = el.select_one(".organic-list-offer-title a")
    if not title_el:
        title_el = el.select_one("a[title]")

    title = ""
    url = ""
    if title_el:
        title = _clean(title_el.get("title") or title_el.get_text())
        url = title_el.get("href", "")
        if url.startswith("//"):
            url = "https:" + url
    if not title:
        return None

    # Precio
    price_el = el.select_one(".search-card-e-price-main")
    if not price_el:
        price_el = el.select_one(".element-offer-price-amount")
    price = _clean(price_el.get_text()) if price_el else ""

    # Imagen
    img = el.select_one("img.search-card-e-slider__img")
    if not img:
        img = el.select_one("img.seb-img-lazy")
    if not img:
        img = el.select_one("img")

    image_url = None
    if img:
        src = img.get("data-src") or img.get("src") or ""
        if src and ("alibaba" in src or "alicdn" in src or src.startswith("http")):
            image_url = _normalize_image(src)

    # Descripción / snippet debajo del título
    description = ""
    desc_el = el.select_one(".search-card-e-introduction")
    if not desc_el:
        desc_el = el.select_one(".search-card-e-desc-text")
    if desc_el:
        description = _clean(desc_el.get_text())

    # MOQ / info adicional / features
    features = []
    feature_els = el.select(".search-card-m-sale-features__item, .search-card-e-company")
    for feat_el in feature_els[:3]:
        t = _clean(feat_el.get_text())
        if t and len(t) > 3:
            features.append(t)

    return {
        "title": title,
        "url": url,
        "price": price,
        "image_url": image_url,
        "description": description,
        "features": features,
    }


async def search_alibaba(
    product_name: str,
    timeout: float = 15.0,
    max_results: int = 3,
) -> dict:
    """
    Busca un producto en Alibaba y retorna imágenes y textos.

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
                params={"SearchText": product_name},
            )
            resp.raise_for_status()
    except Exception:
        return result

    soup = BeautifulSoup(resp.text, "html.parser")

    # Resultados
    items = soup.select("div.organic-list-offer-inner")
    if not items:
        items = soup.select("div.fy23-search-card")
    if not items:
        items = soup.select("div.search-card-e-content")
    if not items:
        items = soup.select('[class*="search-card"]')

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
