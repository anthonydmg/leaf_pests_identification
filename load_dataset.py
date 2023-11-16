import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os

ROOT_DATA = "C:/Users/ileon/Documents/Anthony/leaf_pests_identification/data"
TEST_SIZE = 0.25
IMAGE_SIZE = 224
NUM_WORKERS = 4
BATCH_SIZE = 8
# Image normalization
def nomalize_transform(pretrained = True):
    if pretrained: # Normalizacion para los pesos pre entrenados
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else: # Normalizacion para entrenamiento desde cero
        normalize = transforms.Normalize(
            mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5]
        )
    
    return normalize

# Train pre processing image Transforms
def get_train_preprocessing_transforms(IMAGE_SIZE, pretrained = True):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        nomalize_transform(pretrained)
    ])

    return train_transform

def get_datasets(pretrained):
    """
    Function to prepare the Datasets.
    """

    dataset_train = datasets.ImageFolder(
        f"{ROOT_DATA}/train",
        transform=(get_train_preprocessing_transforms(IMAGE_SIZE, pretrained))
    )

    dataset_test = datasets.ImageFolder(
        f"{ROOT_DATA}/test",
        transform=(get_train_preprocessing_transforms(IMAGE_SIZE, pretrained))
    )

    return dataset_train, dataset_test, dataset_train.classes


if __name__ == "__main__":
    dataset_train, dataset_test, classes = get_datasets(pretrained=True)


    train_loader = DataLoader(
            dataset_train, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=1
            )
    valid_loader = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=1
    )

