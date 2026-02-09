"""
API HTTP para Canvas-a-Productos.
Expone extracción de productos desde URL de Canva vía script.py.
"""
import os
import uuid
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Importar funciones del script existente
from script import process_url_to_csv_openai, enrich_from_out_dir, read_csv_dicts

app = FastAPI(title="Canvas-a-Productos API", version="1.0.0")

# Directorio base para jobs (se crea por job_id)
OUTPUT_BASE = Path(os.environ.get("OUTPUT_BASE", "/app/output"))
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Estado de jobs en memoria
JOBS: dict[str, dict] = {}


class ExtractRequest(BaseModel):
    url: str
    max_pages: Optional[int] = None
    two_pass: bool = False
    enrich: bool = False


def run_extraction(job_id: str, url: str, max_pages: Optional[int], two_pass: bool, enrich: bool) -> None:
    """Ejecuta la extracción en segundo plano."""
    try:
        JOBS[job_id]["status"] = "running"
        out_dir = str(OUTPUT_BASE / job_id)
        os.makedirs(out_dir, exist_ok=True)

        model_page = os.environ.get("OPENAI_MODEL_PAGE", "gpt-4o")
        model_crop = os.environ.get("OPENAI_MODEL_CROP") or model_page
        model_enrich = os.environ.get("OPENAI_MODEL_ENRICH") or model_crop
        use_smart_crop = os.environ.get("SMART_CROP", "").lower() in ("1", "true", "yes")
        smart_pad_rel = float(os.environ.get("PAD_REL", "0.02"))
        min_conf_keep = float(os.environ.get("MIN_CONF_KEEP", "0.25"))
        batch_size = int(os.environ.get("ENRICH_BATCH_SIZE", "6"))
        max_enrich_calls = int(os.environ.get("ENRICH_MAX_CALLS", "50"))
        enrich_dedupe = os.environ.get("ENRICH_DEDUPE", "").lower() in ("1", "true", "yes")

        result = process_url_to_csv_openai(
            url=url,
            out_dir=out_dir,
            csv_name="productos.csv",
            model_page=model_page,
            model_crop=model_crop,
            two_pass=two_pass,
            max_pages=max_pages,
            use_smart_crop=use_smart_crop,
            smart_pad_rel=smart_pad_rel,
            min_conf_keep=min_conf_keep,
        )

        csv_path = Path(out_dir) / "productos.csv"
        if not csv_path.exists():
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = "No se generó el CSV de productos."
            return

        usage = dict(result.get("usage") or {})

        if enrich:
            try:
                enrich_result = enrich_from_out_dir(
                    out_dir=out_dir,
                    csv_in_name="productos.csv",
                    csv_out_name="productos_enriquecidos.csv",
                    model_enrich=model_enrich,
                    batch_size=batch_size,
                    max_enrich_calls=max_enrich_calls,
                    dedupe=enrich_dedupe,
                )
                enrich_usage = enrich_result.get("usage") or {}
                for k in ("input_tokens", "output_tokens", "total_tokens"):
                    usage[k] = usage.get(k, 0) + enrich_usage.get(k, 0)
                csv_path = Path(out_dir) / "productos_enriquecidos.csv"
            except Exception as e:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = f"Error en enriquecimiento: {e}"
                return

        JOBS[job_id]["usage"] = usage
        u_in = usage.get("input_tokens", 0)
        u_out = usage.get("output_tokens", 0)
        u_tot = usage.get("total_tokens", 0)
        print(f"[OK] Procesamiento completado. Tokens: input={u_in} output={u_out} total={u_tot}")

        rows = read_csv_dicts(str(csv_path))
        crops_dir = Path(out_dir) / "crops"

        products = []
        for r in rows:
            img_name = (r.get("Imagen") or "").strip()
            crop_path = crops_dir / img_name if img_name else None
            prod = {
                "nombre": (r.get("Nombre") or "").strip(),
                "variante": (r.get("Variante") or "").strip(),
                "precio": (r.get("Precio") or "").strip(),
                "imagen": img_name,
                "imagen_path": str(crop_path) if crop_path and crop_path.exists() else None,
            }
            if "Descripcion" in r:
                prod["descripcion"] = (r.get("Descripcion") or "").strip()
            products.append(prod)

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["products"] = products
        JOBS[job_id]["output_dir"] = out_dir
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)


@app.get("/health")
def health():
    return {"ok": True, "service": "canvas-a-productos"}


@app.post("/extract")
def start_extract(req: ExtractRequest, background_tasks: BackgroundTasks):
    """Inicia extracción de productos desde URL de Canva. Devuelve job_id para consultar estado."""
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "pending", "url": req.url}

    # Enrich activo por defecto si ENABLE_ENRICH=1|true|yes en el .env (p. ej. con docker compose)
    enable_enrich_by_env = os.environ.get("ENABLE_ENRICH", "").lower() in ("1", "true", "yes")
    enrich = req.enrich or enable_enrich_by_env

    background_tasks.add_task(
        run_extraction,
        job_id,
        req.url,
        req.max_pages,
        req.two_pass,
        enrich,
    )

    return {"job_id": job_id, "status": "pending"}


@app.get("/extract/status/{job_id}")
def get_status(job_id: str):
    """Consulta el estado de un job de extracción."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job no encontrado")

    job = JOBS[job_id]
    out = {"job_id": job_id, "status": job["status"]}

    if job["status"] == "completed":
        # Productos con imagen filename; el backend construye la URL completa
        products = []
        for p in job.get("products", []):
            prod = {k: v for k, v in p.items() if k != "imagen_path"}
            products.append(prod)
        out["products"] = products
        out["job_id"] = job_id  # Para que el backend construya /crops/{job_id}/{imagen}
        if job.get("usage"):
            out["usage"] = job["usage"]  # Tokens usados: input_tokens, output_tokens, total_tokens

    elif job["status"] == "failed":
        out["error"] = job.get("error", "Error desconocido")

    return out


@app.get("/crops/{job_id}/{filename}")
def serve_crop(job_id: str, filename: str):
    """Sirve un archivo de imagen crop del job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job no encontrado")

    crop_path = OUTPUT_BASE / job_id / "crops" / filename
    if not crop_path.exists():
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    return FileResponse(crop_path, media_type="image/png")
