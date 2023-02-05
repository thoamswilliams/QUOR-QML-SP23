#!/usr/bin/env bash
# test which network configuration is how good at the simple-seq memoization task
# this experiment is very neat and simple; keep

orders=( 1 2 3 )
degrees=( 1 2 3 )
stages=( 1 2 3 4 5 )
workspaces=( 1 2 3 4 5 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s


PORT=27777


for o in "${orders[@]}"; do
for d in "${degrees[@]}"; do
for s in "${stages[@]}"; do
for w in "${workspaces[@]}"; do
    # increment port in case multiple runs on same machine
    ((PORT++)) 

    TAG="topology-$o-$d-$s-$w"

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
        --seed 2349711 \
        --num-shards 2 \
        --epochs 500 \
        train \
        --dataset simple-seq \
        --workspace $w \
        --stages $s \
        --order $o \
        --degree $d \
        --optimizer rmsprop \
        --learning-rate 0.01 \
        --sentence-length 20 \
        --batch-size 1


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
done