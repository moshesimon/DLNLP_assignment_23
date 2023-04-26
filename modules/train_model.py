from modules.evaluate_model import eval_model
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from transformers import get_scheduler
from config import *

def fine_tune(model, train_dataloader, eval_dataloader, experiment, num_epochs=3, lr=5e-5):
    """
    Fine-tune the given model using the provided data loaders.

    Args:
        model (nn.Module): Pretrained model to fine-tune.
        train_dataloader (DataLoader): Training data loader.
        eval_dataloader (DataLoader): Evaluation data loader.
        experiment (int): Experiment number.
        num_epochs (int, optional): Number of training epochs. Defaults to 3.
        lr (float, optional): Learning rate. Defaults to 5e-5.

    Returns:
        nn.Module: Fine-tuned model.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
        
    # define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_epochs = num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    print("number of training steps: {}".format(num_training_steps))
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    training_losses = []
    eval_losses = []
    for epoch in range(num_epochs):
        losses = []
        for batch in train_dataloader:
            #move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            #zero gradients
            optimizer.zero_grad()
            outputs = model.forward(**batch)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)
            #record our loss per epoch
            losses.append(loss.cpu().detach())
            
        mean_losses = np.mean(losses)
        print(f'epoch {epoch} complete, train loss: {mean_losses}')
        training_losses.append(mean_losses)
        #evaluate our model performance on the validation set:
        eval_loss = eval_model(model, eval_dataloader)
        eval_losses.append(eval_loss)
    
    #plot the training and validation losses
    plot_train_eval_losses(training_losses, eval_losses, experiment)

    return model

def plot_train_eval_losses(training_losses, eval_losses, experiment):
    """
    Plot training and validation losses.

    Args:
        training_losses (list): List of training losses.
        eval_losses (list): List of evaluation losses.
        experiment (int): Experiment number.
    """
    plt.plot(training_losses, label='Training loss')
    plt.plot(eval_losses, label='Validation loss')
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    #save the plot
    plt.savefig(os.path.join(figures_path,f'experiment_{experiment}_train_val_losses.png'))
    #close the plot
    plt.close()