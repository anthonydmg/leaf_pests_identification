import torchvision.models as models
import torch.nn as nn


def build_model(pretrained = True, fine_tune = True, num_classes = 5):
    model = models.efficientnet_b0(pretrained)

    if fine_tune:
        print("[INFO]: Entrenando todas las capas...")
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Congelando capas ocultas...')
        for params in model.parameters():
            params.requires_grad = False

    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

    return model






