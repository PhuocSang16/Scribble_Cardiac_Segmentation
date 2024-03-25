import torch
from typing import List, Dict, Optional, Union, Tuple
from tqdm.auto import tqdm
import numpy as np
import sys
import torch.nn as nn
from typing import List, Dict, Optional, Union, Tuple
import random
from utils.metrics import AverageMeter

# Định nghĩa hàm huấn luyện cho VNet
def train(
        epoch: int,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        device: torch.device,
        criterion: Dict,
        weights: torch.tensor,
        prossesID: int = None
        ) -> Tuple[int, list]:

    model.train()

    prefix = 'Training'

    if prossesID is not None:
        prefix = "[{}]{}".format(prossesID, prefix)

    losses = AverageMeter()

    with tqdm(total=len(loader), ascii=True, desc=('{}: {:02d}'.format(prefix, epoch))) as pbar:
        for batch_i, sample in enumerate(loader, 0):

            # ================= Extract Data ==================
            batch_img = sample['image'].to(device)
            batch_label = sample['label'].to(device)

            # =================== forward =====================
            output = model(batch_img)
            loss = criterion(output, batch_label)

            # =================== backward ====================
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pbar.update()

            losses.update(loss.item(), batch_img.size(0))
            pbar.set_description(f"Epoch {epoch} - Trainig Loss: {losses.avg:.4f}")

    return losses

def validate(mode: str,
             epoch: int,
             loader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             device: torch.device,
             criterion: Dict,
             weights: torch.tensor,
             prossesID: int = None
             ) -> Tuple[int, list]:

    model.eval()

    if mode == 'validation':
        prefix = 'Validating'
    elif mode == 'test':
        prefix = 'Testing'
    if prossesID is not None:
        prefix = "[{}]{}".format(prossesID, prefix)

    losses = AverageMeter()

    with torch.inference_mode():
        with tqdm(total=len(loader), ascii=True, desc=('{}: {:02d}'.format(prefix, epoch))) as pbar:
            for batch_i, sample in enumerate(loader, 0):

                # ================= Extract Data ==================
                batch_img = sample['image'].to(device)
                batch_gt = sample['gt'].to(device)

                # =================== forward =====================
                main_output, _ = model(batch_img)
                output_soft = torch.softmax(main_output, dim=1)
                loss = criterion['ce_loss'](output_soft, batch_gt.unsqueeze(1).long())

                # =================== backward ====================
                pbar.update()

                losses.update(loss.item(), batch_img.size(0))
                pbar.set_description(f"Epoch {epoch} - Val Loss: {losses.avg:.4f}")
    return losses
