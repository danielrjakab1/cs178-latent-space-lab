# model HANDOUT
import os
import uuid
import pickle as pkl
import base64
import io
import numpy as np

import torch
from PIL import Image

# Data dirs
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
LATENT_DIR = os.path.join(DATA_DIR, 'latents')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
os.makedirs(LATENT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load the pretrained StyleGAN2 generator once at import
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


def _synth_to_pil(img_tensor):
    img = (img_tensor * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    arr = img[0].transpose(1, 2, 0)
    return Image.fromarray(arr)


def _save_z_and_get_id(z_tensor):
    z_id = uuid.uuid4().hex
    path = os.path.join(LATENT_DIR, f"{z_id}.pth")
    torch.save(z_tensor.cpu(), path)
    return z_id


def _z_path(z_id):
    return os.path.join(LATENT_DIR, f"{z_id}.pth")


def _image_path(img_id):
    return os.path.join(IMAGE_DIR, f"{img_id}.png")


def _pil_to_data_url(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    b = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b}"


def sample_and_generate():
    z = torch.randn([1, G.z_dim], device=device) # TODO: sample z 
    z_id = _save_z_and_get_id(z)
    img_b64 = generate_from_z_tensor(z)
    return z_id, img_b64


def generate_from_z_tensor(z_tensor):
    z = z_tensor.to(device)
    with torch.no_grad():
        c = None
        w = G.mapping(z, c)
        img_tensor = G.synthesis(w)
        pil = _synth_to_pil(img_tensor)
        return _pil_to_data_url(pil)


def generate_from_z_id(z_id):
    path = _z_path(z_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f'latent id not found: {z_id}')
    z = torch.load(path).to(device)
    return generate_from_z_tensor(z)


def arithmetic(z_id_a, z_id_b, operation='add'):
    path_a = _z_path(z_id_a)
    path_b = _z_path(z_id_b)
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        raise FileNotFoundError('one or both latent ids not found')
    z_a = torch.load(path_a).to(device)
    z_b = torch.load(path_b).to(device)
    if operation == 'add':
        z_new = z_a + z_b # TODO 
    elif operation == 'subtract_ab':
        z_new = z_a - z_b # TODO 
    elif operation == 'subtract_ba':
        z_new = z_b - z_a # TODO 
    else:
        raise ValueError('unsupported operation')
    new_id = _save_z_and_get_id(z_new)
    img_b64 = generate_from_z_tensor(z_new)
    return new_id, img_b64


def interpolate(z_id_a, z_id_b, steps=7):
    path_a = _z_path(z_id_a)
    path_b = _z_path(z_id_b)
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        raise FileNotFoundError('one or both latent ids not found')
    z_a = torch.load(path_a).to(device)
    z_b = torch.load(path_b).to(device)
    imgs = []
    ids = []
    alphas = list(np.linspace(0.0, 1.0, steps))
    for i, a in enumerate(alphas):
        z_new = (1-a) * z_a + a * z_b # TODO 
        new_id = _save_z_and_get_id(z_new)
        ids.append(new_id)
        img_b64 = generate_from_z_tensor(z_new)
        imgs.append(img_b64)
    return {"latent_ids": ids, "images": imgs, "alphas": alphas}


def interpolate_weight(z_id_a, z_id_b, weight=0.5):
    """
    Interpolate a single weighted latent: final_z = weight * z_a + (1-weight) * z_b
    Returns (new_id, image_data_url)
    """
    path_a = _z_path(z_id_a)
    path_b = _z_path(z_id_b)
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        raise FileNotFoundError('one or both latent ids not found')
    z_a = torch.load(path_a).to(device)
    z_b = torch.load(path_b).to(device)
    w = float(weight)
    z_new = w * z_a + (1-w) * z_b # TODO 
    new_id = _save_z_and_get_id(z_new)
    img_b64 = generate_from_z_tensor(z_new)
    return new_id, img_b64
