# model — latent-space lab backend
#
# Images are keyed by a deterministic SHA-256 hash of the raw latent bytes so
# the same mathematical vector always maps to the same cached PNG.
#
# Latents are held in an in-process dict (LATENT_CACHE) for the lifetime of
# the server.  No latent files are written to disk — only PNG images are
# persisted.
#
# Interpolation convention (used everywhere):
#   z_out = (1 - t) * z_A  +  t * z_B      (t in [0, 1])
#   t = 0  ->  pure A
#   t = 1  ->  pure B
#
# The slider "weight" w exposed to the frontend maps as:
#   w = 1 - t   ->   w = 1 means pure A, w = 0 means pure B
# So internally we always convert: t = 1 - w before calling _blend.

import os
import hashlib
import base64
import io
import numpy as np
import pickle as pkl

import torch
from PIL import Image

# ── Directories ───────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(ROOT, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
os.makedirs(IMAGE_DIR, exist_ok=True)

# ── In-process latent cache (z_id -> CPU tensor) ──────────────────────────────
LATENT_CACHE: dict = {}

# ── Load generator ────────────────────────────────────────────────────────────
print('Loading StyleGAN2 generator (this may take a while)...')
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
print(f'Using device: {device}')

with open('download/ffhq.pkl', 'rb') as f:
    G = pkl.load(f)['G_ema'].to(device)
G.eval()
print('Generator loaded: z_dim=', G.z_dim, 'resolution=', G.img_resolution)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tensor_hash(z: torch.Tensor) -> str:
    """Stable SHA-256 of raw float32 bytes -- same vector -> same hash."""
    raw = z.cpu().to(torch.float32).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _image_path(h: str) -> str:
    return os.path.join(IMAGE_DIR, f'{h}.png')


def _pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{b}'


def _synth_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(arr[0].transpose(1, 2, 0))


def _generate_pil(z: torch.Tensor) -> Image.Image:
    z = z.to(device)
    with torch.no_grad():
        w = G.mapping(z, None)
        return _synth_to_pil(G.synthesis(w))


def _get_or_generate(z: torch.Tensor):
    """
    Return (z_id, data_url).
    z_id is the SHA-256 hash of the latent tensor; it is stored in the
    in-process LATENT_CACHE so future calls (arithmetic / interpolation)
    can retrieve the tensor by ID without touching the filesystem.
    """
    z_cpu   = z.cpu().to(torch.float32)
    z_id    = _tensor_hash(z_cpu)
    img_pth = _image_path(z_id)

    # Always keep the tensor available in memory
    LATENT_CACHE[z_id] = z_cpu

    if os.path.exists(img_pth):
        with open(img_pth, 'rb') as f:
            b = base64.b64encode(f.read()).decode('ascii')
        return z_id, f'data:image/png;base64,{b}'

    pil = _generate_pil(z_cpu)
    pil.save(img_pth, format='PNG')
    return z_id, _pil_to_data_url(pil)


def _load_z(z_id: str) -> torch.Tensor:
    if z_id in LATENT_CACHE:
        return LATENT_CACHE[z_id]
    raise FileNotFoundError(f'latent id not found in session cache: {z_id}')


def _blend(z_a: torch.Tensor, z_b: torch.Tensor, t: float) -> torch.Tensor:
    """z_out = (1-t)*z_A + t*z_B.  t=0 -> pure A, t=1 -> pure B."""
    return (1.0 - t) * z_a + t * z_b


# ── Public API ────────────────────────────────────────────────────────────────

def sample_and_generate():
    """Draw a random latent and return (z_id, image_data_url)."""
    z = torch.randn([1, G.z_dim], device=device)
    return _get_or_generate(z)


def arithmetic(z_id_a: str, z_id_b: str, operation: str = 'add'):
    z_a = _load_z(z_id_a).to(device)
    z_b = _load_z(z_id_b).to(device)
    if   operation == 'add':         z_new = z_a + z_b
    elif operation == 'subtract_ab': z_new = z_a - z_b
    elif operation == 'subtract_ba': z_new = z_b - z_a
    else: raise ValueError(f'unsupported operation: {operation}')
    return _get_or_generate(z_new)


def interpolate(z_id_a: str, z_id_b: str, steps: int = 7):
    """
    Return a filmstrip of `steps` images from A (t=0) to B (t=1).
    ts values are returned so the frontend can convert to slider weight w = 1 - t.
    """
    z_a = _load_z(z_id_a).to(device)
    z_b = _load_z(z_id_b).to(device)
    ts  = list(np.linspace(0.0, 1.0, steps))   # t=0 -> A, t=1 -> B
    ids, imgs = [], []
    for t in ts:
        new_id, img_b64 = _get_or_generate(_blend(z_a, z_b, t))
        ids.append(new_id)
        imgs.append(img_b64)
    return {'latent_ids': ids, 'images': imgs, 'ts': ts}


def interpolate_weight(z_id_a: str, z_id_b: str, weight: float = 0.5):
    """
    Single weighted blend using the slider convention:
      w = 1 -> pure A,  w = 0 -> pure B
    Converts to t = 1 - w so _blend() is used and results are always
    served from the image cache (no cache miss vs the filmstrip).
    """
    z_a = _load_z(z_id_a).to(device)
    z_b = _load_z(z_id_b).to(device)
    t   = 1.0 - float(weight)      # slider weight -> blend param
    return _get_or_generate(_blend(z_a, z_b, t))
