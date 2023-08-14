#!/bin/bash
device="cuda"
num_epochs=200
learning_rate=0.001
learni=100
step=50
python trainer.py --device $device --num_epochs $num_epochs --learning_rate $learning_rate --learni $learni --step $step

