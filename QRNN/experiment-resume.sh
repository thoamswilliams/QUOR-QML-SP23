#!/usr/bin/env bash
# resuming experiments

seeds=( 28258 )
checkpoints=( "$@" )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"


trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s

PORT=42000

for sd in "${seeds[@]}"; do
for checkpoint in "${checkpoints[@]}"; do
    # increment port in case multiple runs on same machine
    ((PORT++))

    TAG="resume-mnist-$checkpoint-$sd"

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

    
    # run test
    echo "running $TAG"
    OMP_NUM_THREADS=2 ./main.py \
        --tag experiment-$TAG \
        --seed $sd \
        --port $PORT \
        --num-shards 2 \
        --epochs 10000 \
        resume \
        `ls -t results/best/mnist36.2/*$checkpoint* | head -1` \
        --override-learning-rate 0.005 \
        --override-batch-size 250

    
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
