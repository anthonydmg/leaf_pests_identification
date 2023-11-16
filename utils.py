import torch
import matplotlib.pyplot as plt
import os
def save_plots(train_acc, valid_acc, train_loss, valid_loss,  train_f1, valid_f1):

    os.makedirs("./results", exist_ok= True)
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"./results/accuracy.png")


    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='orange', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./results/loss.png")


    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_f1, color='blue', linestyle='-', 
        label='train f1-score'
    )
    plt.plot(
        valid_f1, color='orange', linestyle='-', 
        label='validataion f1-score'
    )
    plt.xlabel('Epochs')
    plt.ylabel('F1-score')
    plt.legend()
    plt.savefig(f"./results/f1-score.png")




def save_model(epochs, model, optimizer, criterion, model_name):
    """
    Function to save the trained model to disk.
    """
    os.makedirs("./outputs", exist_ok= True)
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"./outputs/finituned_{model_name}.pth")
