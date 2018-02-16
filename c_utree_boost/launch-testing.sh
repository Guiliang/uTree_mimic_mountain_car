#!/bin/bash
for ((n=1;n<101;n++))
do
echo running game $n
python test_boost_Galen.py -g $n -a 'result-correlation-all-epoch-linear-st0' -b 'result-mse-all-epoch-linear-st0'>>temp-all-testing-st0.out -e '_linear_epoch_decay_lr' 2>&1 &
wait
echo finishing game $n
sleep 10s
done
exit 0