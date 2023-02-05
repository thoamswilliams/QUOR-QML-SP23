#!/usr/bin/env bash
# see whether we can get simple-quotes

seeds=( 720 7292 4402 5427 4269 7928 3475 5114 3975 2733 1217 8443 2 2826 9432 6936 5081 3774 7427 700 1664 7262 499 9736 6654 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s

for sd in "${seeds[@]}"
do
    TAG="memoize-quotes-$sd"

    LOCKFILE="$LOCKFILEFOLDER/experiment-$TAG.lock"
    DONEFILE="$LOCKFILEFOLDER/experiment-$TAG.done"
    sync

    if [[ ! -f "$DONEFILE" ]] ; then
        {
            if flock -n 200 ; then
                echo "running $TAG"
                ./main.py \
                    --tag experiment-$TAG \
                    --seed $sd \
                    --num-shards 3 \
                    --epochs 1000 \
                    train \
                    --dataset simple-quotes \
                    --workspace 6 \
                    --stages 3 \
                    --order 2 \
                    --degree 3 \
                    --optimizer rmsprop \
                    --learning-rate 0.005 \
                    --sentence-length 36 \
                    --batch-size 2
                
                if  [[ $? -eq 0 ]] ; then
                    touch "$DONEFILE"                
                    sync
                else
                    echo "failure running $TAG."
                fi
                sleep 1

            else
                echo "skipping $TAG"
            fi
        } 200>"$LOCKFILE"
    fi
done