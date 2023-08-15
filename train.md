python -c "import train; train.train_loop(device='cuda:0', num_epochs=150, learning_rate=0.01)"
or
python trainer.py --learning_rate=0.01 --device="cuda:0" --num_epochs=150
or
./run_trainer.sh
