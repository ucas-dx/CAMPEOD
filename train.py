#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# @Project : CAMPEOD
# @Time    : 2023/8/10 15:32
# @Author  : Deng xun
# @Email   : 38694034@qq.com
# @File    : train.py
# @Software: PyCharm 
# -------------------------------------------------------------------------------
from torch import optim
import copy
from utils import *
import model.net as net
from data import *

def train_loop(device="cpu", train_loader=train_loader, Val_loader=Val_loader,
               num_epochs=150, learning_rate=0.01, learni=100,step=50):
    model=net.CampeodNet().to(device)
    print(f"device:{device}-epochs:{num_epochs}-learning_rate:{learning_rate}")
    #model = nn.DataParallel(model,device_ids=[0,1]).cuda()
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = DiceLoss()
    criterion3 = FocalLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0)
    down_params1 = list(model.swindencode1.model.parameters())
    down_params2 = list(model.swin64.modelswin.parameters())
    optimizer.param_groups.append(
        {'params': down_params1, 'lr': learning_rate / learni, 'weight_decay': 0, 'momentum': 0, 'dampening': 0,
         'nesterov': True, 'maximize': False, 'foreach': None})
    optimizer.param_groups.append(
        {'params': down_params2, 'lr': learning_rate/ learni, 'weight_decay': 0, 'momentum': 0, 'dampening': 0,
         'nesterov': True, 'maximize': False, 'foreach': None})
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_dice = 0
        train_iou = 0
        train_num_acc=0
        for batch in tqdm.tqdm(train_loader):
            oimage = []
            imge = []
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            masks64 = F.interpolate(masks, size=(64, 64), mode='bilinear', align_corners=False)
            outputs,x64= model(images)
            x64=torch.mean(x64,dim=-1).unsqueeze(1)
            optimizer.zero_grad()
            outimg = copy.copy(outputs)
            outimg = np.squeeze(outimg, axis=1)  # 降维
            for k in masks:
                imagek = k.cpu().numpy() * 255
                oimage.append(imagek[0])
            for t in outimg:
                outimg1 = torch.sigmoid(t).detach().cpu().numpy()
                imge.append(outimg1)
            loss1 = criterion1(outputs, masks)
            loss2 = criterion2(outputs, masks)
            loss3 = criterion3(outputs, masks)
            loss11 = criterion1(x64, masks64)
            loss22 = criterion2(x64, masks64)
            loss33 = criterion3(x64, masks64)
            lossm = loss11 + loss22 + loss33
            loss = loss1 + loss2 +loss3  + 0.5 * lossm
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_dice += dice_coefficient(torch.sigmoid(outputs), masks).item()
            train_iou += iou(torch.sigmoid(outputs), masks).item()
        scheduler.step()
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)
        train_num_acc/=len(train_loader)

        def val(data_loader):
            model.eval()
            val_loss = 0
            val_dice = 0
            val_iou = 0
            with torch.no_grad():
                for batch in tqdm.tqdm(data_loader):
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                    outputs,_ = model(images)
                    optimizer.zero_grad()
                    loss1 = criterion1(outputs, masks)
                    loss2 = criterion2(outputs, masks)
                    loss3 = criterion3(outputs, masks)
                    loss11 = criterion1(x64, masks64)
                    loss22 = criterion2(x64, masks64)
                    loss33 = criterion3(x64, masks64)
                    lossm = loss11 + loss22 + loss33
                    loss = loss1 + loss2 + loss3 + 0.5 * lossm
                    val_loss += loss.item()
                    val_dice += dice_coefficient(torch.sigmoid(outputs), masks).item()
                    val_iou += iou(torch.sigmoid(outputs), masks).item()
            val_loss /= len(data_loader)
            val_dice /= len(data_loader)
            val_iou /= len(data_loader)
            return val_loss,val_dice,val_iou
        val_loss,val_dice,val_iou=val(Val_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}')
        print(f'Test  - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}')
        torch.save(model.state_dict(),
                       f'Epoch_{epoch + 1}_modeldict.pth')

