import cv2
import torch
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader

import config
from src.attack import SLAEAttack
from src.utils import get_model, get_model_name, get_static_hash, numpy2base64, resize_image, save_image

router = APIRouter()
templates = Environment(loader=FileSystemLoader("web/templates"), cache_size=0)

device = torch.device("cuda")
model_name = get_model_name()
model = get_model().to(device)
model.load(f'models/{model_name}.pth')
attack_method = SLAEAttack(model, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.IMAGE_DEPTH)


@router.get("/")
def index() -> HTMLResponse:
    template = templates.get_template("index.html")
    content = template.render(version=get_static_hash(), dataset=config.DATASET, model_name=model_name)
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
def attack(input_image: UploadFile = File(...), target_image: UploadFile = File(...), method: str = Form(...), scale: float = Form(...)) -> JSONResponse:
    input_image = resize_image(save_image(input_image))
    target_image = resize_image(save_image(target_image))

    if method == "qp":
        attacked_image = attack_method.qp_attack(input_image, target_image, scale)
    elif method == "split_matrix":
        attacked_image = attack_method.split_matrix_attack(input_image, target_image)
    else:
        return JSONResponse({"status": "error", "message": f'неизвестный метод атаки "{method}"'})

    if attacked_image is None:
        return JSONResponse({"status": "error", "message": "не удалось провести атаку"})

    return JSONResponse({
        "status": "success",
        "image": numpy2base64(attacked_image),
        "prediction": attack_method.predict(attacked_image)
    })
