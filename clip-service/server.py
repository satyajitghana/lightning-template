import requests
import torch
import io
import os
import redis
import hashlib

import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load dataset
dataset = load_dataset("jxie/flickr8k")

# load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# load image embeddings
img_embeds = torch.tensor(np.load("image_embeds.npy"))

r = redis.Redis(
    host=os.environ["CACHE_HOST"],
    port=os.environ["CACHE_PORT"]
)


@app.get("/text-to-image")
async def find_image(text: str):
    inp_hash = hashlib.md5(text.encode()).hexdigest()

    # check if input is in cache
    if r.exists(inp_hash):
        text_feats_enc_b = r.get(inp_hash)
        text_feats_np = np.load(io.BytesIO(
            text_feats_enc_b), allow_pickle=True)
        text_features = torch.tensor(text_feats_np)
    else:
        inputs = processor([text], padding=True, return_tensors="pt")
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)

        # save it in cache
        np_bytes = io.BytesIO()
        np.save(np_bytes, text_features.cpu().numpy(), allow_pickle=True)
        r.set(inp_hash, np_bytes.getvalue())

    image_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp()
    logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
    img_idx = torch.argmax(logits_per_text[0]).item()
    similar_img = dataset['train'][img_idx]['image']

    img_byte_arr = io.BytesIO()
    similar_img.save(img_byte_arr, format="JPEG")

    return Response(img_byte_arr.getvalue(), media_type="image/jpeg")


@app.get("/health")
async def health():
    return {"message": "ok"}
