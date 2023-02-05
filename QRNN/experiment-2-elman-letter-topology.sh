#!/usr/bin/env bash
# elman-letter

workspaces=( 3 4 5 6 7 8 )
stages=( 1 2 3 4 )
degrees=( 1 2 3 4 )
seeds=( 6936 5081 3774 7427 700 1664 7262 499 9736 6654 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s


for sd in "${seeds[@]}"
do
for ws in "${workspaces[@]}"
do
for st in "${stages[@]}"
do
for dg in "${degrees[@]}"
do
    TAG="elman-letter-topology-$sd-$ws-$st-$dg"

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
    ./main.py \
        --tag experiment-$TAG \
        --seed $sd \
        --num-shards 2 \
        --epochs 1000 \
        --stop-at-loss 0.01 \
        train \
        --dataset elman-letter \
        --workspace $ws \
        --stages $st \
        --order 2 \
        --degree $dg \
        --optimizer adam \
        --learning-rate 0.005 \
        --sentence-length 36 \
        --batch-size 4
    
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