import torch
import argparse
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import build_model
from load_dataset import get_datasets

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
BATCH_SIZE = 8

def compute_metrics(y_true, predictions):
    f1 = f1_score(y_true, predictions, average="macro")
    recall = recall_score(y_true, predictions,  average="macro")
    precision = precision_score(y_true, predictions,  average="macro")
    accuracy = accuracy_score(y_true, predictions)
    return  {"f1": f1, "recall": recall, "precision": precision, "accuracy": accuracy}



def train_step(model, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0.0
    all_true_labels = []
    all_preds_labels = []
    progress_bar = tqdm(train_loader, total= len(train_loader))
    for data in progress_bar:
        #counter += 1
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        # Calcular perdida
        loss =  criterion(outputs, labels)

        train_loss += loss.item()
        # Calcular accuracy
        preds = torch.argmax(outputs,dim=1)
        
        progress_bar.set_postfix({"loss": loss.item()})

        all_preds_labels.extend(preds.detach().cpu().numpy())
        all_true_labels.extend(labels.detach().cpu().numpy())
        #_, preds = torch.max(outputs.data, 1)
        loss.backward()
        optimizer.step()
    ## Loss 
    epoch_loss = train_loss / len(train_loader)

    train_metrics = compute_metrics(all_true_labels, all_preds_labels)


    return epoch_loss, train_metrics, progress_bar


def validation_step(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    all_true_labels = []
    all_preds_labels = []
    for data in tqdm(val_loader, total= len(val_loader)):
        #counter += 1
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        # Calcular perdida
        loss =  criterion(outputs, labels)

        val_loss += loss.item()
        # Calcular accuracy
        preds = torch.argmax(outputs,dim=1)
        
        all_preds_labels.extend(preds.detach().cpu().numpy())
        all_true_labels.extend(labels.detach().cpu().numpy())
        #_, preds = torch.max(outputs.data, 1)
    ## Loss 
    epoch_loss = train_loss / len(train_loader)
    val_metrics = compute_metrics(all_true_labels, all_preds_labels)

    return epoch_loss, val_metrics

if __name__ == "__main__":
    epochs = 5
    dataset_train, dataset_test, classes = get_datasets(pretrained=True)


    train_loader = DataLoader(
            dataset_train, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=1
            )
    valid_loader = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=1
    )

    device = 'cuda' if torch.cuda.is_available() else "cpu"

    model = build_model(
        num_classes = len(classes)
        ).to(device)
    
    print("Model\n:", model)
    total_parameters =  sum(p.numel() for p in model.parameters())
    print("\ntotal de parametros:", total_parameters)

    total_trainable_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\ntotal de parametros entrenables :", total_trainable_params)


    ## Optimizer
    optimizer = optim.Adam(model.parameters())
    ## Loss function
    criterion = nn.CrossEntropyLoss()

    print("Training...")

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    #train_loss = 0

    ## Train epochs

    for epoch in range(epochs):
        print(f"[INFO]: Epoca {epoch+1} de {epochs}")
        train_loss, train_metrics, progress_bar = train_step(model, train_loader, optimizer, criterion)
        print("train_loss:", train_loss)
        print("train_metrics:", train_metrics)

        val_loss, val_metrics = validation_step(model, valid_loader, criterion)

        progress_bar.set_postfix({"loss": train_loss, 
                                  "train_accuracy": train_metrics["accuracy"],
                                  "train_f1_score":  train_metrics["f1"],
                                  "val_loss": val_loss, 
                                  "val_accuracy": val_metrics["accuracy"],
                                  "val_f1_score": val_metrics["f1"],
                                 })
        print("val_loss:", val_loss)
        print("val_metrics:", val_metrics)
        
        break

        ## train
        


        

