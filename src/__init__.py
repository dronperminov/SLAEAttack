import torch

import config
from src.dense_network import DenseNetwork
from src.attack import SLAEAttack

device = torch.device("cuda")
model = DenseNetwork(config.INPUT_SIZE, config.SIZES, config.ACTIVATION).to(device)
model.load("models/model_epoch10.pth")
attack_method = SLAEAttack(model, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.IMAGE_DEPTH)
