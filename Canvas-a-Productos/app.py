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
from script import process_url_to_csv_openai, read_csv_dicts

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
        use_smart_crop = os.environ.get("SMART_CROP", "").lower() in ("1", "true", "yes")
        smart_pad_rel = float(os.environ.get("PAD_REL", "0.02"))
        min_conf_keep = float(os.environ.get("MIN_CONF_KEEP", "0.25"))
        process_url_to_csv_openai(
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

        rows = read_csv_dicts(str(csv_path))
        crops_dir = Path(out_dir) / "crops"

        products = []
        for r in rows:
            img_name = (r.get("Imagen") or "").strip()
            # Ruta local para servir la imagen
            crop_path = crops_dir / img_name if img_name else None
            products.append({
                "nombre": r.get("Nombre", "").strip(),
                "variante": r.get("Variante", "").strip(),
                "precio": r.get("Precio", "").strip(),
                "imagen": img_name,
                "imagen_path": str(crop_path) if crop_path and crop_path.exists() else None,
            })

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

    background_tasks.add_task(
        run_extraction,
        job_id,
        req.url,
        req.max_pages,
        req.two_pass,
        req.enrich,
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
