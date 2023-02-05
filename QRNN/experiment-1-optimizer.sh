#!/usr/bin/env bash
# test which optimizer works within which regime
# this experiment is very neat and simple; keep

optimizers=( "sgd" "rmsprop" "adam" "lbfgs" )
learningrates=( 10.0 5.0 2.0 1.0 0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001 0.00005 0.00002 0.00001 )
seeds=( 112 113 114 115 116 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

trap "exit" INT
sleep $[ ($RANDOM % 10) + 1 ]s


PORT=27777


for optim in "${optimizers[@]}"; do
for lr in "${learningrates[@]}"; do
for sd in "${seeds[@]}"; do
    # increment port in case multiple runs on same machine
    ((PORT++)) 

    TAG="optimizer-$optim-$lr-$sd"

    LOCKFILE="$LOCKFILEFOLDER/experiment-$TAG.lock"
    DONEFILE="$LOCKFILEFOLDER/experiment-$TAG.done"
    FAILFILE="$LOCKFILEFOLDER/experiment-$TAG.fail"

    # check if any lockfiles present
    sync
    if [[ -f "$DONEFILE" || -f "$FAILFILE" || -f "$LOCKFILE" ]] ; then
        echo "skipping $TAG"
        continue
    fi

    # try to aquire lockfile
    exec 200>"$LOCKFILE"
    flock -n 200 || {
        echo "skipping $TAG"
        continue
    }
    
    echo "running $TAG"
     ./main.py \
        --tag experiment-$TAG \
        --num-shards 1 \
        --seed $sd \
        --port $PORT \
        --epochs 500 \
        --stop-at-loss 0.000005 \
        train \
        --dataset simple-seq \
        --batch-size 2 \
        --optimizer $optim \
        --learning-rate $lr \
        --workspace 5 \
        --stages 2 \
        --order 2 \
        --degree 3 \
        --initial-bias-spread 0.5 \
        --initial-weights-spread 0.5 \
        --initial-unitaries-spread 0.5 
    
    if  [[ $? -eq 0 ]] ; then
        touch "$DONEFILE"    
    else
        touch "$FAILFILE"    
        echo "failure running $TAG."
    fi 

    sync   
    sleep 10
    rm "$LOCKFILE"
    sync   
    sleep 10
done
done
done
