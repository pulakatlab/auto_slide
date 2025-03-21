"""
Resources
- https://github.com/DatumLearning/Mask-RCNN-finetuning-PyTorch/blob/main/notebook.ipynb
- https://www.youtube.com/watch?v=vV9L71hK-RE
- https://www.youtube.com/watch?v=t1MrzuAUdoE

"""

#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import sys
import cv2

import torch
import torchvision
# from torchvision import transforms as T
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Add parent directory to path to import utils
autoslide_dir = '/home/abuzarmahmood/projects/auto_slide'
# autoslide_dir = '/home/exouser/project/auto_slide'

if "__file__" not in globals():
    __file__ = os.path.join(autoslide_dir, 'src/pipeline/model/training.py')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import augment_dataset, generate_negative_samples, generate_artificial_vessels

plot_dir = os.path.join(autoslide_dir, 'plots') 
artifacts_dir = os.path.join(autoslide_dir, 'artifacts')

os.makedirs(plot_dir, exist_ok = True)
os.makedirs(artifacts_dir, exist_ok = True)

labelled_data_dir = os.path.join(autoslide_dir, 'data/labelled_images')
img_dir = os.path.join(labelled_data_dir, 'images/') 
mask_dir = os.path.join(labelled_data_dir, 'masks/') 
image_names = sorted(os.listdir(img_dir))
mask_names = sorted(os.listdir(mask_dir))

for img_path, mask_path in zip(image_names, mask_names):
    assert img_path.split(".")[0] in mask_path

# Create random rotation strictly of 90 or 270 degrees
class RandomRotation90():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            angle = np.random.choice([90, 270])
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask


transform = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            # T.RandomRotation(degrees=30), # Rotation might remove masks close to edge
            RandomRotation90(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor()
        ])

idx = 4
img = Image.open(img_dir + image_names[idx]).convert("RGB")
mask = Image.open(mask_dir + mask_names[idx])

img, mask = transform(img, mask)

fig,axis = plt.subplots(1,2,figsize=(10,5))
axis[0].imshow(img.T)
axis[1].imshow(mask.T)
plt.show()

# Test negative image generation
img = Image.open(img_dir + image_names[idx]).convert("RGB")
mask = Image.open(mask_dir + mask_names[idx])
img = np.array(img)
mask = np.array(mask)
neg_img, neg_mask = generate_negative_samples(img, mask)

fig,axis = plt.subplots(1,2,figsize=(10,5))
axis[0].imshow(neg_img)
axis[1].imshow(neg_mask)
plt.show()

def get_mask_outline(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_outline = np.zeros_like(mask)
    mask_outline = cv2.drawContours(mask_outline, contours, -1, 255, 1)
    return mask_outline

# Test artificial vessel generation
art_img, art_mask = generate_artificial_vessels(img, mask)
art_mask_outline = get_mask_outline(art_mask)
fig,axis = plt.subplots(2,2,figsize=(10,10))
axis[0,0].imshow(img)
axis[0,1].imshow(mask)
axis[1,0].imshow(art_img)
axis[1,1].imshow(art_mask)
axis[1,0].scatter(*np.where(art_mask_outline)[::-1], c='y', s=1)
plt.show()

class CustDat(torch.utils.data.Dataset):
    def __init__(self, image_names, mask_names, transform=None):
        self.image_names = image_names
        self.mask_names = mask_names
        # Define augmentation transformations
        # Simple transform for validation/testing
        self.base_transform = T.ToTensor()
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.base_transform

    def __getitem__(self, idx):
        img = Image.open(img_dir + self.image_names[idx]).convert("RGB")
        mask = Image.open(mask_dir + self.mask_names[idx])
        # img = Image.open(img_dir + image_names[idx]).convert("RGB")
        # mask = Image.open(mask_dir + mask_names[idx])

        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(img)
        # ax[1].imshow(mask)
        # plt.show()

        # Apply transformations
        img_tensor, mask_tensor = self.transform(img, mask)
        # img_tensor, mask_tensor = transform(img, mask)

        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(img_tensor.permute(1, 2, 0).cpu().numpy())
        # ax[1].imshow(mask_tensor.permute(1, 2, 0).cpu().numpy())
        # plt.show()

        # Convert mask back to numpy array
        mask = mask_tensor.numpy()[0] * 255
        mask = mask.astype(np.uint8)


        # Something weird is happening (likely with the transform)
        # 255 is showing up in a very small number of pixels
        # Enfore that a mask has to cover at least 1% of the image

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        fin_objs = []
        for obj in obj_ids:
            if np.mean(mask == obj) > 0.005:
                fin_objs.append(obj)

        obj_ids = np.array(fin_objs)

        num_objs = len(obj_ids)
        masks = np.zeros((num_objs, mask.shape[0], mask.shape[1]))
        for i in range(num_objs):
            masks[i][mask == obj_ids[i]] = True


        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i]>0)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # fig, ax = plt.subplots(1, len(masks), figsize=(10, 5))
        # for i in range(num_objs):
        #     ax[i].imshow(masks[i])
        #     # Plot bounding box
        #     ax[i].add_patch(
        #             plt.Rectangle((boxes[i][0], boxes[i][1]), boxes[i][2]-boxes[i][0], boxes[i][3]-boxes[i][1],
        #                           color="y", linewidth=2, fill=False))
        # plt.show()
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(img_tensor.permute(1, 2, 0).cpu().numpy())
        # ax[1].imshow(np.sum(np.array(masks), axis=0))
        # # Also draw bounding boxes
        # for box in boxes:
        #     ax[0].add_patch(
        #             plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
        #                           fill=False, color="y", linewidth=2))
        # plt.show()

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        
        return img_tensor, target

    def __len__(self):
        return len(self.image_names)


model = torchvision.models.detection.maskrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features , 2)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask , hidden_layer , 2)


# No need for a separate transform variable as it's now in the CustDat class

def custom_collate(data):
  return data

num = int(0.9 * len(image_names))
num = num if num % 2 == 0 else num + 1
train_imgs_inds = np.random.choice(range(len(image_names)) , num , replace = False)
val_imgs_inds = np.setdiff1d(range(len(image_names)) , train_imgs_inds)
train_imgs = np.array(image_names)[train_imgs_inds]
val_imgs = np.array(image_names)[val_imgs_inds]
train_masks = np.array(mask_names)[train_imgs_inds]
val_masks = np.array(mask_names)[val_imgs_inds]


# Check if augmented images already exist
aug_img_dir = os.path.join(labelled_data_dir, 'augmented_images/')
aug_mask_dir = os.path.join(labelled_data_dir, 'augmented_masks/')

if os.path.exists(aug_img_dir) and os.path.exists(aug_mask_dir) \
        and len(os.listdir(aug_img_dir)) > 0 and len(os.listdir(aug_mask_dir)) > 0:
    print("Augmented images already exist. Skipping augmentation...")

    aug_img_names = sorted(os.listdir(aug_img_dir))
    aug_mask_names = sorted(os.listdir(aug_mask_dir))
else:
    # Create augmented dataset paths
    print("Creating augmented dataset...")
    n_augmented = len(train_imgs) * 10
    # Load a few images to augment
    aug_img_list = []
    aug_mask_list = []
    for i in np.random.choice(range(len(train_imgs)), n_augmented, replace=True):
        img = np.array(Image.open(img_dir + train_imgs[i]).convert("RGB"))
        mask = np.array(Image.open(mask_dir + train_masks[i]))
        aug_img_list.append(img)
        aug_mask_list.append(mask)

    # Augment the dataset
    aug_images, aug_masks = augment_dataset(aug_img_list, aug_mask_list, neg_ratio=0.3, art_ratio=0.5)

    # Save augmented images and masks
    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_mask_dir, exist_ok=True)

    aug_img_names = []
    aug_mask_names = []
    for i, (img, mask) in enumerate(tqdm(zip(aug_images, aug_masks))):
        img_name = f'aug_{i}.png'
        mask_name = f'aug_{i}_mask.png'
        
        # Save the augmented image and mask
        plt.imsave(os.path.join(aug_img_dir, img_name), img)
        plt.imsave(os.path.join(aug_mask_dir, mask_name), mask, cmap='gray')
        
        aug_img_names.append(img_name)
        aug_mask_names.append(mask_name)

# Also add negative images
neg_image_dir = os.path.join(labelled_data_dir, 'negative_images/')
neg_mask_dir = os.path.join(labelled_data_dir, 'negative_masks/')
neg_img_names = sorted(os.listdir(neg_image_dir))
neg_mask_names = sorted(os.listdir(neg_mask_dir))

# Add augmented images to both training and validation sets
# train_imgs = np.append(train_imgs, aug_img_names)
# train_masks = np.append(train_masks, aug_mask_names)
n_aug_train = int(0.9 * len(aug_img_names))
n_aug_val = len(aug_img_names) - n_aug_train
n_neg_train = int(0.9 * len(neg_img_names))
n_neg_val = len(neg_img_names) - n_neg_train
train_imgs = np.concatenate([train_imgs, aug_img_names[:n_aug_train], neg_img_names[:n_neg_train]])
train_masks = np.concatenate([train_masks, aug_mask_names[:n_aug_train], neg_mask_names[:n_neg_train]])
val_imgs = np.concatenate([val_imgs, aug_img_names[n_aug_train:], neg_img_names[n_neg_train:]])
val_masks = np.concatenate([val_masks, aug_mask_names[n_aug_train:], neg_mask_names[n_neg_train:]])

# Update img_dir and mask_dir to include augmented directories
orig_img_dir = img_dir
orig_mask_dir = mask_dir

test_plot_dir = os.path.join(plot_dir, 'train_val_split')
os.makedirs(test_plot_dir, exist_ok=True)


n_plot = 10
train_inds = np.random.choice(range(len(train_imgs)), n_plot, replace=False)
for img_name, mask_name in zip(train_imgs[train_inds], train_masks[train_inds]):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    if 'aug_' in img_name:
        img = Image.open(aug_img_dir + img_name).convert("RGB")
        mask = Image.open(aug_mask_dir + mask_name)
    elif 'neg_' in img_name:
        img = Image.open(neg_image_dir + img_name).convert("RGB")
        mask = Image.open(neg_mask_dir + mask_name)
    else:
        img = Image.open(img_dir + img_name).convert("RGB")
        mask = Image.open(mask_dir + mask_name)
    ax[0].imshow(img)
    ax[1].imshow(mask)
    # plt.show()
    fig.savefig(test_plot_dir + f'/{img_name.split(".")[0]}train.png')
    plt.close(fig)

val_inds = np.random.choice(range(len(val_imgs)), n_plot, replace=False)
for img_name, mask_name in zip(val_imgs[val_inds], val_masks[val_inds]):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    if 'aug_' in img_name:
        img = Image.open(aug_img_dir + img_name).convert("RGB")
        mask = Image.open(aug_mask_dir + mask_name)
    elif 'neg_' in img_name:
        img = Image.open(neg_image_dir + img_name).convert("RGB")
        mask = Image.open(neg_mask_dir + mask_name)
    else:
        img = Image.open(img_dir + img_name).convert("RGB")
        mask = Image.open(mask_dir + mask_name)
    ax[0].imshow(img)
    ax[1].imshow(mask)
    # plt.show()
    fig.savefig(test_plot_dir + f'/{img_name.split(".")[0]}val.png')
    plt.close(fig)

# Create a custom dataset that can handle both original and augmented images
class AugmentedCustDat(torch.utils.data.Dataset):
    def __init__(self, image_names, mask_names, transform=None):
        self.image_names = image_names
        self.mask_names = mask_names
        self.base_transform = T.ToTensor()
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.base_transform

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        mask_name = self.mask_names[idx]
        
        # Check if this is an augmented image
        if 'aug_' in img_name:
            img = Image.open(os.path.join(aug_img_dir, img_name)).convert("RGB")
            mask = Image.open(os.path.join(aug_mask_dir, mask_name))
        elif 'neg_' in img_name:
            img = Image.open(os.path.join(neg_image_dir, img_name)).convert("RGB")
            mask = Image.open(os.path.join(neg_mask_dir, mask_name))
        else:
            img = Image.open(orig_img_dir + img_name).convert("RGB")
            mask = Image.open(orig_mask_dir + mask_name)

        # Apply transformations
        img_tensor, mask_tensor = self.transform(img, mask)
        
        # Convert mask back to numpy array
        mask = mask_tensor.numpy()[0] * 255
        mask = mask.astype(np.uint8)

        # Filter objects by size
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        fin_objs = []
        for obj in obj_ids:
            if np.mean(mask == obj) > 0.005:
                fin_objs.append(obj)

        obj_ids = np.array(fin_objs)

        num_objs = len(obj_ids)
        masks = np.zeros((num_objs, mask.shape[0], mask.shape[1]))
        for i in range(num_objs):
            masks[i][mask == obj_ids[i]] = True

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i]>0)
            if len(pos[0]) == 0:  # Skip if mask is empty
                continue
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        if len(boxes) == 0:  # If no valid boxes, create a dummy box
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        
        return img_tensor, target

    def __len__(self):
        return len(self.image_names)

train_dl = torch.utils.data.DataLoader(AugmentedCustDat(train_imgs, train_masks, transform),
                                 batch_size = 2,
                                 shuffle = True,  # Changed to True for better training
                                 collate_fn = custom_collate,
                                 num_workers = 1,
                                 pin_memory = True if torch.cuda.is_available() else False,
                                 drop_last=True)
val_dl = torch.utils.data.DataLoader(AugmentedCustDat(val_imgs, val_masks, transform),
                                 batch_size = 2,
                                 shuffle = False,
                                 collate_fn = custom_collate,
                                 num_workers = 1,
                                 pin_memory = True if torch.cuda.is_available() else False,
                                 drop_last=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device : {device}')

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# for i, dt in enumerate(train_dl):
#     imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
#     targ = [dt[0][1] , dt[1][1]]
#     targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
#     break

run_test_plot_dir = os.path.join(plot_dir, 'run_test_plot')
os.makedirs(run_test_plot_dir, exist_ok=True)

n_epochs = 60
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
    for i , dt in enumerate(pbar): 
        pbar.set_description(f"Epoch {epoch}/{n_epochs}, Batch {i}/{n_train}")
        imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
        targ = [dt[0][1] , dt[1][1]]
        
        # Plot example image and mask to make sure augmentation is working
        if i==0:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(imgs[0].cpu().detach().numpy().transpose(1, 2, 0))
            ax[1].imshow(np.sum(targ[0]['masks'].cpu().detach().numpy(), axis=0))
            # Plot boxes
            for box in targ[0]['boxes']:
                ax[0].add_patch(
                        plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                      fill=False, color="y", linewidth=2))
            fig.savefig(run_test_plot_dir + f'/{epoch}_{i}_train.png')
            plt.close(fig)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
        loss = model(imgs , targets)
        if not flag:
            print(loss)
            flag = True
        losses = sum([l for l in loss.values()])
        train_epoch_loss += losses.cpu().detach().numpy()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    all_train_losses.append(train_epoch_loss)
    with torch.no_grad():
        for j , dt in enumerate(val_dl):
            if len(dt) < 2:
                continue
            imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
            targ = [dt[0][1] , dt[1][1]]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
            loss = model(imgs , targets)
            losses = sum([l for l in loss.values()])
            val_epoch_loss += losses.cpu().detach().numpy()
        all_val_losses.append(val_epoch_loss)
    print(epoch , "  " , train_epoch_loss , "  " , val_epoch_loss)
    
    # Save model only if validation loss improves
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        best_model = model.state_dict()
        # torch.save(model.state_dict(), os.path.join(artifacts_dir, 'mask_rcnn_model.pth'))
        # print(f"New best model saved with validation loss: {best_val_loss}")

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].plot(all_train_losses)
    ax[1].plot(all_val_losses)
    ax[0].set_title("Train Loss")
    ax[1].set_title("Validation Loss")
    ax[1].axhline(y=best_val_loss, color='r', linestyle='--', label='Best Val Loss')
    # plt.show()
    fig.savefig(plot_dir + '/train_val_loss.png')
    plt.close(fig)


# pbar = tqdm(train_dl)
# for i , dt in enumerate(pbar): 
#     pbar.set_description(f"Epoch {epoch}/{n_epochs}, Batch {i}/{n_train}")
#     imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
#     targ = [dt[0][1] , dt[1][1]]
#     fig, ax = plt.subplots(1, 2, figsize=(10,5))
#     ax[0].imshow(imgs[0].cpu().detach().numpy().transpose(1,2,0))
#     ax[1].imshow(np.sum(targ[0]['masks'].cpu().detach().numpy(), axis=0))
#     # Plot boxes
#     for box in targ[0]['boxes']:
#         ax[0].add_patch(
#                 plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
#                               fill=False, color="y", linewidth=2))
# plt.show()

torch.save(best_model, os.path.join(artifacts_dir, 'mask_rcnn_model.pth'))


# Save loss histories
np.save(artifacts_dir + '/train_losses.npy', all_train_losses)
np.save(artifacts_dir + '/val_losses.npy', all_val_losses)



# Plot example predicted mask from validation set
model.eval()

transform = T.ToTensor()
for img_path, mask_path in tqdm(zip(val_imgs, val_masks), total=len(val_imgs)):
    img = Image.open(img_dir + img_path).convert("RGB")
    mask = Image.open(mask_dir + mask_path)
    # Use the base transform for prediction to match training
    ig = transform(img)
    with torch.no_grad():
        pred = model([ig.to(device)])

    # fig, ax = plt.subplots(1, 2, figsize=(10,5))
    # ax[0].imshow(img)
    # ax[1].imshow(ig.cpu().detach().numpy().transpose(1,2,0))
    # plt.show()


    n_preds = len(pred[0]["masks"])
    if n_preds > 0:
        fig, ax = plt.subplots(1, n_preds+1, figsize=(5*n_preds,5))
        ax[0].imshow(img)
        for i in range(n_preds):
            ax[i+1].imshow((pred[0]["masks"][i].cpu().detach().numpy() * 255).astype("uint8").squeeze())
        # plt.show()
        fig.savefig(plot_dir + f'/{img_path.split(".")[0]}example_masks.png')
        plt.close(fig)


        all_preds = np.stack(
                [
                    (pred[0]["masks"][i].cpu().detach().numpy() * 255).astype("uint8").squeeze() \
                            for i in range(n_preds)
                            ]
                )


        fig, ax = plt.subplots(1, 3, figsize=(10,5))
        ax[0].imshow(img)
        ax[1].imshow(all_preds.mean(axis = 0))
        ax[2].imshow(np.array(mask))
        # plt.show()
        plt.savefig(plot_dir + f'/{img_path.split(".")[0]}mean_example_mask.png')
        plt.close()

