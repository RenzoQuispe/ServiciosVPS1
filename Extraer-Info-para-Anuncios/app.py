"""
Extraer-Info-para-Anuncios — Servicio VPS para enriquecer anuncios
con contenido externo de Amazon, Alibaba y MercadoLibre,
y generar preguntas frecuentes con OpenAI.
"""

import asyncio
import json
import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel

from scrapers import search_mercadolibre, search_amazon, search_alibaba

# Cargar variables de entorno desde .env
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Extraer-Info-para-Anuncios", version="1.0.0")

SCRAPE_TIMEOUT = float(os.getenv("SCRAPE_TIMEOUT", "15"))
MAX_RESULTS_PER_SOURCE = int(os.getenv("MAX_RESULTS_PER_SOURCE", "3"))
MAX_PER_PRODUCT = 3  # máximo 3 items por tipo POR PRODUCTO

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ── Modelos ──────────────────────────────────────────────────

class ProductInput(BaseModel):
    id: int
    name: str
    description: str = ""


class EnrichRequest(BaseModel):
    products: list[ProductInput]


class EnrichResponse(BaseModel):
    images: list[dict]
    videos: list[dict]
    texts: list[dict]
    preguntas: list[dict]


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"ok": True, "service": "extraer-info-anuncios"}


@app.post("/enrich", response_model=EnrichResponse)
async def enrich(req: EnrichRequest):
    """
    Recibe lista de productos (id + nombre) y retorna:
    - imágenes, videos y textos extraídos (max 3 por producto)
    - preguntas y respuestas generadas por OpenAI (3 por producto)
    """
    products = req.products[:3]

    all_images: list[dict] = []
    all_videos: list[dict] = []
    all_texts: list[dict] = []

    # Scraping en paralelo por producto
    scrape_tasks = [_scrape_product(p.id, p.name) for p in products]
    scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

    for res in scrape_results:
        if isinstance(res, Exception):
            logger.warning("Scrape error: %s", res)
            continue
        all_images.extend(res["images"])
        all_videos.extend(res["videos"])
        all_texts.extend(res["texts"])

    # Generar Q&A con OpenAI en paralelo por producto (usando descripción del negocio)
    qa_tasks = [
        _generate_qa(p.id, p.name, p.description)
        for p in products
    ]
    qa_results = await asyncio.gather(*qa_tasks, return_exceptions=True)

    all_preguntas: list[dict] = []
    for res in qa_results:
        if isinstance(res, Exception):
            logger.warning("Q&A generation error: %s", res)
            continue
        all_preguntas.extend(res)

    return EnrichResponse(
        images=all_images,
        videos=all_videos,
        texts=all_texts,
        preguntas=all_preguntas,
    )


# ── Scraping ─────────────────────────────────────────────────

async def _scrape_product(product_id: int, product_name: str) -> dict:
    """Busca un producto en las 3 fuentes en paralelo y retorna max 3 de cada tipo.
    Prioridad: Amazon > Alibaba > MercadoLibre."""
    logger.info("Scraping product %d: %s", product_id, product_name)

    amz_task = search_amazon(
        product_name, timeout=SCRAPE_TIMEOUT, max_results=MAX_RESULTS_PER_SOURCE
    )
    ali_task = search_alibaba(
        product_name, timeout=SCRAPE_TIMEOUT, max_results=MAX_RESULTS_PER_SOURCE
    )
    ml_task = search_mercadolibre(
        product_name, timeout=SCRAPE_TIMEOUT, max_results=MAX_RESULTS_PER_SOURCE
    )

    amz_res, ali_res, ml_res = await asyncio.gather(
        amz_task, ali_task, ml_task, return_exceptions=True
    )

    raw_images: list[dict] = []
    raw_videos: list[dict] = []
    raw_texts: list[dict] = []

    for source_name, res in [("amazon", amz_res), ("alibaba", ali_res), ("mercadolibre", ml_res)]:
        if isinstance(res, Exception):
            logger.warning("Error scraping %s for '%s': %s", source_name, product_name, res)
            continue

        for img in res.get("images", []):
            raw_images.append({
                "product_id": product_id,
                "url": img["url"],
                "source": source_name,
                "alt": img.get("alt", product_name),
                "source_url": img.get("source_url", ""),
            })

        for vid in res.get("videos", []):
            raw_videos.append({
                "product_id": product_id,
                "url": vid["url"],
                "source": source_name,
                "thumbnail": vid.get("thumbnail", ""),
            })

        for txt in res.get("texts", []):
            raw_texts.append({
                "product_id": product_id,
                "title": txt.get("title", ""),
                "description": txt.get("description", ""),
                "features": txt.get("features", []),
                "source": source_name,
                "price": txt.get("price", ""),
                "url": txt.get("url", ""),
            })

    images = _dedup_by_key(raw_images, key="url", limit=MAX_PER_PRODUCT)
    videos = _dedup_by_key(raw_videos, key="url", limit=MAX_PER_PRODUCT)
    texts = _dedup_by_key(raw_texts, key="title", limit=MAX_PER_PRODUCT)

    logger.info(
        "Scrape results for product %d (%s): %d images, %d videos, %d texts (raw: %d/%d/%d)",
        product_id, product_name,
        len(images), len(videos), len(texts),
        len(raw_images), len(raw_videos), len(raw_texts),
    )

    return {"images": images, "videos": videos, "texts": texts}


def _dedup_by_key(items: list[dict], key: str, limit: int) -> list[dict]:
    """Deduplica por un campo y limita la cantidad."""
    seen: set[str] = set()
    result: list[dict] = []
    for item in items:
        val = (item.get(key) or "").strip().lower()[:120]
        if val and val in seen:
            continue
        if val:
            seen.add(val)
        result.append(item)
        if len(result) >= limit:
            break
    return result


# ── Q&A con OpenAI ───────────────────────────────────────────

async def _generate_qa(
    product_id: int,
    product_name: str,
    product_description: str,
) -> list[dict]:
    """
    Genera 3 preguntas y respuestas frecuentes para un producto
    usando OpenAI, basándose en el nombre y descripción propios del negocio.
    Si la descripción es corta o vacía, GPT genera preguntas genéricas.
    Retorna lista de dicts con product_id, question, answer.
    """
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY no configurada, omitiendo Q&A")
        return []

    desc = (product_description or "").strip()
    has_description = len(desc) > 20

    if has_description:
        context_block = f"""Descripción del producto proporcionada por el negocio:
{desc}"""
        extra_rule = "- Basa las preguntas y respuestas en la descripción proporcionada por el negocio"
    else:
        context_block = "No se proporcionó descripción detallada del producto."
        extra_rule = "- Como no hay descripción detallada, genera preguntas genéricas pero útiles para un comprador potencial de este tipo de producto"

    prompt = f"""Genera exactamente 3 preguntas frecuentes con sus respuestas sobre el producto "{product_name}".

{context_block}

REGLAS:
- Cada pregunta DEBE empezar con un emoji relevante al tema de la pregunta
- Las preguntas deben ser útiles para un comprador potencial
- Las respuestas deben ser informativas, concisas (2-5 oraciones), simpáticas(usa emojis) y en español neutro
- Temas sugeridos: características principales, usos/beneficios, compatibilidad o cuidados
{extra_rule}
- NO mencionar precios ni tiendas específicas
- NO inventar especificaciones técnicas exactas que no estén en la descripción

Responde SOLO con un JSON array de 3 objetos, sin markdown ni texto adicional:
[
  {{"question": "🔋 ¿Cuánto dura la batería?", "answer": "..."}},
  {{"question": "📦 ¿Qué incluye el paquete?", "answer": "..."}},
  {{"question": "🛡️ ¿Es resistente al agua?", "answer": "..."}}
]"""

    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente experto en productos que genera preguntas frecuentes útiles para compradores. Responde SOLO con JSON válido.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=800,
        )

        raw = response.choices[0].message.content.strip()
        # Limpiar posible markdown wrapping
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        qa_list = json.loads(raw)
        if not isinstance(qa_list, list):
            logger.warning("OpenAI Q&A response is not a list for product %d", product_id)
            return []

        result: list[dict] = []
        for item in qa_list[:3]:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                result.append({
                    "product_id": product_id,
                    "question": str(item["question"]),
                    "answer": str(item["answer"]),
                })

        logger.info("Generated %d Q&A for product %d: %s", len(result), product_id, product_name)
        return result

    except Exception as e:
        logger.error("OpenAI Q&A error for product %d: %s", product_id, e)
        return []
