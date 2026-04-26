from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import os

from . import model
from typing import Optional

app = FastAPI(title='StyleGAN2 Lab API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend files
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/", include_in_schema=False)
def root():
    index_path = os.path.join(frontend_dir, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail='index.html not found')


class GenerateResponse(BaseModel):
    latent_id: str
    image: str


class ArithmeticRequest(BaseModel):
    id_a: str
    id_b: str
    operation: str  # 'add' | 'subtract_ab' | 'subtract_ba'


class InterpRequest(BaseModel):
    id_a: str
    id_b: str
    # Single weighted interpolation (slider drag / filmstrip click)
    weight: Optional[float] = None
    # Full filmstrip generation
    steps: Optional[int] = None


@app.post('/generate', response_model=GenerateResponse)
def generate():
    try:
        z_id, img_b64 = model.sample_and_generate()
        return {"latent_id": z_id, "image": img_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/arithmetic', response_model=GenerateResponse)
def arithmetic(req: ArithmeticRequest):
    try:
        new_id, img_b64 = model.arithmetic(req.id_a, req.id_b, req.operation)
        return {"latent_id": new_id, "image": img_b64}
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/health')
def health():
    return {"status": "ok"}


@app.post('/interpolate')
def interpolate(req: InterpRequest):
    try:
        if req.weight is not None:
            # Single weighted interpolation -> return single image + latent id
            new_id, img = model.interpolate_weight(req.id_a, req.id_b, weight=float(req.weight))
            return {"latent_id": new_id, "image": img}
        else:
            steps = req.steps if req.steps is not None else 7
            result = model.interpolate(req.id_a, req.id_b, steps=max(2, int(steps)))
            return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
