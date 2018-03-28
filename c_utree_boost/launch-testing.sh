#!/bin/bash
for ((n=1;n<201;n++))
do
echo running game $n
python test_boost_Galen.py -g $n -a 'result-correlation-all-linear-epoch-decay-lr-st0-1000' -b 'result-mse-all-linear-epoch-decay-lr-st0-1000' -f 'result-mae-all-linear-epoch-decay-lr-st0-1000' -j 'result-rae-all-linear-epoch-decay-lr-st0-1000' -i 'result-rse-all-linear-epoch-decay-lr-st0-1000' >>temp-all-testing-st0-1000.out -e '_linear_epoch_decay_lr' 2>&1 &
wait
echo finishing game $n
sleep 10s
done
exit 0