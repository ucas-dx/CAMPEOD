#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# @Project : CAMPEOD
# @Time    : 2023/8/10 12:59
# @Author  : Deng xun
# @Email   : 38694034@qq.com
# @File    : test.py
# @Software: PyCharm 
# -------------------------------------------------------------------------------
from model.net import CampeodNet
from utils import *
from data import *
import torch
import tqdm
def test_model(device='cpu', test_loader=test_loader):
    model = CampeodNet().to(device)
    model.load_state_dict(torch.load(r"campeod.pth",map_location=device))
    for epoch in range(1):
        model.eval()
        test_dice = 0
        test_iou = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader,colour="blue"):
                images = batch['image'].to(device)
                # print(images.shape)
                masks = batch['mask'].to(device)
                # print(masks.shape)
                outputs,_ = model(images)
                test_dice += dice_coefficient(torch.sigmoid(outputs), masks).item()
                test_iou += iou(torch.sigmoid(outputs), masks).item()
        test_dice /= len(test_loader)
        test_iou /= len(test_loader)
        print(f'Test-, Dice: {test_dice:.4f}, IoU: {test_iou:.4f}')
