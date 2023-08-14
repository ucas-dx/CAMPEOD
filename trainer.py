#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# @Project : CAMPEOD
# @Time    : 2023/8/14 18:18
# @Author  : Deng xun
# @Email   : 38694034@qq.com
# @File    : trainer.py
# @Software: PyCharm 
# -------------------------------------------------------------------------------
import argparse
from train import train_loop
from data import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--learni", type=int, default=100, help="Your learni value")
    parser.add_argument("--step", type=int, default=50, help="Your step value")
    parser.add_argument("--Val_loader",default=Val_loader,help="Your validation-data")
    args = parser.parse_args()
    train_loop(**vars(args))

if __name__ == "__main__":
    main()

