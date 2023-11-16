import cv2
import torch
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader

import config
from src.attack import SLAEAttack
from src.dense_network import DenseNetwork
from src.utils import get_static_hash, numpy2base64, resize_image, save_image

router = APIRouter()
templates = Environment(loader=FileSystemLoader("web/templates"), cache_size=0)

device = torch.device("cuda")
model = DenseNetwork(config.INPUT_SIZE, config.SIZES, config.ACTIVATION).to(device)
model.load("models/model_epoch10.pth")
attack_method = SLAEAttack(model, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.IMAGE_DEPTH)


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
