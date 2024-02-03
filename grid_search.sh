#!/bin/bash

mkdir -p data/models/
mkdir -p data/models/generators/
mkdir -p data/models/discriminator/

# specify hyperparameter values
lr_rmsprop=("0.0001 0.001 0.01 0.0005 0.005 0.05")
lr_sgd=("0.0001 0.001 0.01 0.0005 0.005 0.05")
lr_adam=("0.0001 0.001 0.01 0.0005 0.005 0.05")
epochs=10

# different execution modes
if [ $1 = local ]
then
    echo "[local execution]"
    cmd="src/grid_search.sge"
elif [ $1 = grid ]
then
    echo "[grid execution]"
    cmd="qsub grid_search.sge"
else
    echo "[ERROR! Argument not supported!]"
    exit 1
fi

# do the grid search
# RMSProp
for lr in $lr_rmsprop
do
  echo "RMSProp $lr"
  $cmd "RMSProp'$lr'" --RMSProp $lr
done

# SGD
for lr in $lr_sgd
do
  echo "SGD $lr"
  $cmd "SGD'$lr'" --SGD $lr
done

# Adam
for lr in $lr_adam
do
  echo "Adam $lr"
  $cmd "Adam'$lr'" --adam $lr
done
