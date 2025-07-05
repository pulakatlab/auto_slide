import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2 as cv  # Alias cv2 as cv for consistency
from torchvision.transforms import v2 as T
from tqdm import tqdm, trange

from autoslide.pipeline.model.prediction_utils import initialize_model

#############################################################################
# Directory and Data Management Functions
#############################################################################


def setup_directories(data_dir=None):
    """
    Set up necessary directories for saving artifacts and plots.

    Args:
        data_dir (str, optional): Root data directory

    Returns:
        tuple: (plot_dir, artifacts_dir) - Paths to the plot and artifacts directories
    """
    from autoslide import config

    plot_dir = config['plot_dirs']
    artifacts_dir = config['artifacts_dir']

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    print('Creating directories for plots and artifacts...')
    print(f'Plot directory: {plot_dir}')
    print(f'Artifacts directory: {artifacts_dir}')

    return plot_dir, artifacts_dir

#############################################################################
# Model Initialization and Setup Functions
#############################################################################





#############################################################################
# Training and Evaluation Functions
#############################################################################


def setup_training(model, device):
    """
    Set up optimizer and other training parameters.

    Moves the model to the appropriate device (CPU/GPU) and
    configures the optimizer with appropriate learning rate and momentum.

    Args:
        model (torch.nn.Module): The Mask R-CNN model
        device (torch.device): Device to run the model on (CPU or GPU)

    Returns:
        torch.optim.Optimizer: Configured optimizer for training
    """
    print(f'Setting up training on device: {device}')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    print(f'Number of trainable parameters: {len(params)}')

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    lr = 0.005
    momentum = 0.9
    weight_decay = 0.0005

    print(f'Optimizer configuration:')
    print(f'  Learning rate: {lr}')
    print(f'  Momentum: {momentum}')
    print(f'  Weight decay: {weight_decay}')

    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    return optimizer


def train_model(model, train_dl, val_dl, optimizer, device, plot_dir, artifacts_dir, n_epochs=90):
    """
    Train the model and evaluate on validation set.

    Performs the full training loop:
    1. Trains the model for the specified number of epochs
    2. Evaluates on the validation set after each epoch
    3. Tracks and saves the best model based on validation loss
    4. Plots training and validation losses
    5. Saves model checkpoints and loss histories

    Args:
        model (torch.nn.Module): The Mask R-CNN model
        train_dl (torch.utils.data.DataLoader): DataLoader for training data
        val_dl (torch.utils.data.DataLoader): DataLoader for validation data
        optimizer (torch.optim.Optimizer): Optimizer for training
        device (torch.device): Device to run the model on (CPU or GPU)
        plot_dir (str): Directory to save plots
        artifacts_dir (str): Directory to save model checkpoints
        n_epochs (int): Number of epochs to train for

    Returns:
        tuple: (model, all_train_losses, all_val_losses, best_val_loss) -
               Trained model, training losses, validation losses, and best validation loss
    """
    run_test_plot_dir = os.path.join(plot_dir, 'run_test_plot')
    os.makedirs(run_test_plot_dir, exist_ok=True)

    best_model_path = os.path.join(
        artifacts_dir, 'best_val_mask_rcnn_model.pth')
    fin_model_path = os.path.join(artifacts_dir, 'final_mask_rcnn_model.pth')

    all_train_losses = []
    all_val_losses = []
    best_val_loss = float('inf')  # Track the best validation loss
    best_model = None
    flag = False

    for epoch in trange(n_epochs):
        train_epoch_loss = 0
        val_epoch_loss = 0
        model.train()
        n_train = len(train_dl)
        pbar = tqdm(train_dl)

        for i, dt in enumerate(pbar):
            pbar.set_description(
                f"Epoch {epoch}/{n_epochs}, Batch {i}/{n_train}")
            imgs = [dt[0][0].to(device), dt[1][0].to(device)]
            targ = [dt[0][1], dt[1][1]]

            # Plot example image and mask to make sure augmentation is working
            if i == 0:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(imgs[0].cpu().detach().numpy().transpose(1, 2, 0))
                ax[1].imshow(
                    np.sum(targ[0]['masks'].cpu().detach().numpy(), axis=0))
                # Plot boxes
                for box in targ[0]['boxes']:
                    ax[0].add_patch(
                        plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                      fill=False, color="y", linewidth=2))
                fig.savefig(run_test_plot_dir + f'/{epoch}_{i}_train.png')
                plt.close(fig)

            targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
            loss = model(imgs, targets)
            if not flag:
                print(loss)
                flag = True
            losses = sum([l for l in loss.values()])
            train_epoch_loss += losses.cpu().detach().numpy()
            if np.isnan(train_epoch_loss):
                # raise Exception('Loss is Nan')
                print('Loss is NaN, skipping batch')
                continue
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        all_train_losses.append(train_epoch_loss)

        # Validation loop
        with torch.no_grad():
            for j, dt in enumerate(val_dl):
                if len(dt) < 2:
                    continue
                imgs = [dt[0][0].to(device), dt[1][0].to(device)]
                targ = [dt[0][1], dt[1][1]]
                targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targ]
                loss = model(imgs, targets)
                losses = sum([l for l in loss.values()])
                val_epoch_loss += losses.cpu().detach().numpy()
            all_val_losses.append(val_epoch_loss)

        print(epoch, "  ", train_epoch_loss, "  ", val_epoch_loss)

        # Save model only if validation loss improves
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model = model.state_dict()
            torch.save(best_model, best_model_path)

        torch.save(model.state_dict(), fin_model_path)

        # Plot training progress
        plot_losses(all_train_losses, all_val_losses, plot_dir, best_val_loss)

    # Save loss histories
    np.save(artifacts_dir + '/train_losses.npy', all_train_losses)
    np.save(artifacts_dir + '/val_losses.npy', all_val_losses)

    return model, all_train_losses, all_val_losses, best_val_loss


def plot_losses(all_train_losses, all_val_losses, plot_dir, best_val_loss):
    """
    Plot training and validation losses.

    Creates a visualization of training and validation loss curves
    with the best validation loss highlighted.

    Args:
        all_train_losses (list): List of training losses for each epoch
        all_val_losses (list): List of validation losses for each epoch
        plot_dir (str): Directory to save the plot
        best_val_loss (float): Best validation loss achieved during training
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(all_train_losses)
    ax[1].plot(all_val_losses)
    ax[0].set_title("Train Loss")
    ax[1].set_title("Validation Loss")
    ax[1].axhline(y=best_val_loss, color='r',
                  linestyle='--', label='Best Val Loss')
    fig.savefig(plot_dir + '/train_val_loss.png')
    plt.close(fig)


def evaluate_model(model, val_imgs, val_masks,
                   img_dir, mask_dir, aug_img_dir, aug_mask_dir,
                   device, plot_dir):
    """
    Evaluate the model on validation data and generate prediction visualizations.

    Runs the model on validation and negative samples and creates visualizations
    of the predictions compared to ground truth masks.

    Args:
        model (torch.nn.Module): The trained Mask R-CNN model
        val_imgs (list): List of validation image filenames
        val_masks (list): List of validation mask filenames
        img_dir (str): Directory containing original images
        mask_dir (str): Directory containing original masks
        aug_img_dir (str): Directory containing augmented images
        aug_mask_dir (str): Directory containing augmented masks
        device (torch.device): Device to run the model on (CPU or GPU)
        plot_dir (str): Directory to save the visualizations
    """
    print('Starting model evaluation...')
    model.eval()

    pred_out_path = os.path.join(plot_dir, 'pred_plots')
    os.makedirs(pred_out_path, exist_ok=True)
    print(f'Prediction plots will be saved to: {pred_out_path}')

    transform = T.ToTensor()

    # Evaluate on validation set
    print(f'Evaluating on {len(val_imgs)} validation images...')
    predictions_with_masks = 0
    predictions_without_masks = 0

    for img_name, mask_name in tqdm(zip(val_imgs, val_masks), total=len(val_imgs), desc='Validation evaluation'):
        if 'aug_' in img_name:
            img = Image.open(aug_img_dir + img_name).convert("RGB")
            mask = Image.open(aug_mask_dir + mask_name)
        else:
            img = Image.open(img_dir + img_name).convert("RGB")
            mask = Image.open(mask_dir + mask_name)

        # Use the base transform for prediction to match training
        ig = transform(img)
        with torch.no_grad():
            pred = model([ig.to(device)])

        n_preds = len(pred[0]["masks"])
        if n_preds > 0:
            fig, ax = plt.subplots(1, n_preds+1, figsize=(5*n_preds, 5))
            ax[0].imshow(img)
            for i in range(n_preds):
                ax[i+1].imshow((pred[0]["masks"][i].cpu().detach().numpy()
                               * 255).astype("uint8").squeeze())
            fig.savefig(pred_out_path +
                        f'/{img_name.split(".")[0]}example_masks.png')
            plt.close(fig)

            all_preds = np.stack(
                [
                    (pred[0]["masks"][i].cpu().detach().numpy()
                     * 255).astype("uint8").squeeze()
                    for i in range(n_preds)
                ]
            )

            fig, ax = plt.subplots(1, 3, figsize=(10, 5))
            ax[0].imshow(img)
            ax[1].imshow(all_preds.mean(axis=0))
            ax[2].imshow(np.array(mask))
            plt.savefig(pred_out_path +
                        f'/{img_name.split(".")[0]}mean_example_mask.png')
            plt.close()
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.imshow(img)
            ax.set_title('No predicted mask')
            plt.savefig(pred_out_path +
                        f'/{img_name.split(".")[0]}_mean_example_mask.png')
            plt.close()
            predictions_without_masks += 1

    print(f'Validation evaluation complete:')
    print(f'  Images with predicted masks: {predictions_with_masks}')
    print(f'  Images without predicted masks: {predictions_without_masks}')
    print(f'Model evaluation finished. All plots saved to {pred_out_path}')


def load_model(model, path, device):
    """
    Load a saved model from disk.

    Args:
        model (torch.nn.Module): The Mask R-CNN model structure
        path (str): Path to the saved model weights
        device (torch.device): Device to load the model on (CPU or GPU)

    Returns:
        torch.nn.Module: Loaded model with weights from disk
    """
    model.load_state_dict(torch.load(path, map_location=device))
    return model
