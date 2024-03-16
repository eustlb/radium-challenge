import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import wandb
import yaml
from easydict import EasyDict
from tqdm import tqdm
import monai
import torch.nn as nn
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry

from model import RadSam
from dataset import RadSamDataset

join = os.path.join


if __name__ == "__main__":

    # CONFIG =============================
    # Load YAML config file
    with open('config.yaml', 'r') as file:
        config_dict = yaml.safe_load(file)

    config = EasyDict(config_dict)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    device = config.training.device
    model_save_path = join(config.model.work_dir, config.model.task_name + "-" + run_id)
    os.makedirs(model_save_path, exist_ok=True)
    # ====================================


    # MODEL ==============================
    sam_model = sam_model_registry[config.training.model_type](checkpoint=config.training.checkpoint)
    medsam_model = RadSam(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        freeze_img_encoder=False
    ).to(device)

    print("Number of total parameters: ", sum(p.numel() for p in medsam_model.parameters())) 
    print("Number of trainable parameters: ", sum(p.numel() for p in medsam_model.parameters() if p.requires_grad))  # 93729252

    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=config.training.lr, weight_decay=config.training.weight_decay
    )

    # optimizer
    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=config.training.lr, weight_decay=config.training.weight_decay
    )

    # loss
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # ====================================


    # DATASET ============================
    num_epochs = config.training.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10
    train_dataset = RadSamDataset(config.training.tr_npy_path)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    # ====================================


    # TRAINING ===========================
    start_epoch = 0
    if config.training.resume is not None:
        if os.path.isfile(config.training.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(config.training.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

    if config.training.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            if config.training.use_amp:
                # AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                        medsam_pred, gt2D.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred = medsam_model(image, boxes_np)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1

        epoch_loss /= step
        losses.append(epoch_loss)
        # ====================================

        # LOGGING ============================
        if config.training.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        # save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        # save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))


    # plot loss
    plt.plot(losses)
    plt.title("Dice + Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(join(model_save_path, config.model.task_name + "train_loss.png"))
    plt.close()




    
    

