import base64

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader

from src import attack_method
from src.utils import get_static_hash, numpy2base64, resize_image, save_image

router = APIRouter()
templates = Environment(loader=FileSystemLoader("web/templates"), cache_size=0)


@router.get("/")
def index() -> HTMLResponse:
    template = templates.get_template("index.html")
    content = template.render(version=get_static_hash())
    return HTMLResponse(content=content)


@router.post("/predict")
def predict(image: UploadFile = File(...)) -> JSONResponse:
    image = resize_image(save_image(image))
    _, encoded_image = cv2.imencode(".jpg", image)

    return JSONResponse({
        "status": "success",
        "image": numpy2base64(image),
        "prediction": attack_method.predict(image)
    })


@router.post("/attack")
def attack(input_image: UploadFile = File(...), target_image: UploadFile = File(...)) -> JSONResponse:
    input_image = resize_image(save_image(input_image))
    target_image = resize_image(save_image(target_image))
    attacked_image = attack_method.attack(input_image, target_image)

    return JSONResponse({
        "status": "success",
        "image": numpy2base64(attacked_image),
        "input_prediction": attack_method.predict(input_image),
        "target_prediction": attack_method.predict(target_image),
        "prediction": attack_method.predict(attacked_image)
    })
