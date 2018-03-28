#!/bin/bash
for ((n=200;n<1001;n++))
do
echo running game $n
python train_boost_Galen.py -g $n -e '_linear_epoch_decay_lr' -d '../save_all_transition/' > ./training_tempt_out_dir/temp-$n-linear-epoch-decay-lr.out 2>&1 &
wait
echo finishing game $n
sleep 20s
done
exit 0
