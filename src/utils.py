import base64
import hashlib
import os
import shutil
import tempfile

import cv2
import numpy as np
from fastapi import UploadFile

import config


def get_hash(filename: str) -> str:
    hash_md5 = hashlib.md5()

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def get_static_hash() -> str:
    hashes = []
    styles_dir = os.path.join(os.path.dirname(__file__), "..", "web", "styles")
    js_dir = os.path.join(os.path.dirname(__file__), "..", "web", "js")

    for filename in os.listdir(styles_dir):
        path = os.path.join(styles_dir, filename)

        if os.path.isdir(path):
            for sub_filename in os.listdir(path):
                hashes.append(get_hash(os.path.join(path, sub_filename)))
        else:
            hashes.append(get_hash(path))

    for filename in os.listdir(js_dir):
        path = os.path.join(js_dir, filename)

        if os.path.isdir(path):
            for sub_filename in os.listdir(path):
                hashes.append(get_hash(os.path.join(path, sub_filename)))
        else:
            hashes.append(get_hash(path))

    statis_hash = "_".join(hashes)
    hash_md5 = hashlib.md5()
    hash_md5.update(statis_hash.encode("utf-8"))

    return hash_md5.hexdigest()


def save_image(image: UploadFile) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp_dir:
        extension = image.filename.split(".")[-1]
        file_path = os.path.join(tmp_dir, f"image.{extension}")

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
        finally:
            image.file.close()

        return cv2.imread(file_path)


def resize_image(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 3 and config.IMAGE_DEPTH == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.shape[0] != config.IMAGE_HEIGHT or image.shape[1] != config.IMAGE_WIDTH:
        image = cv2.resize(image, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)

    return image.astype(np.float32) / 255


def numpy2base64(image: np.ndarray) -> str:
    image = (image * 255).astype(np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)
    return base64.b64encode(encoded_image).decode("utf-8")
