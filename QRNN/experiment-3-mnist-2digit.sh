#!/usr/bin/env bash
# mnist01 or mnist36

seeds=( 1 2 3 4 5 )
lrs=( 0.3 0.1 0.03 0.01 )
dgs=( 2 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

DATASET=$1

if [[ "$DATASET" != "mnist01" && "$DATASET" != "mnist01-ds" && "$DATASET" != "mnist36" && "$DATASET" != "mnist36-ds" && "$DATASET" != "mnist01-gen" && "$DATASET" != "mnist-even-odd" ]] ; then
    echo "invalid dataset $DATASET"
    exit 1
fi

trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s


PORT=27777
SEED=20042


for sd in "${seeds[@]}"; do
for lr in "${lrs[@]}"; do
for dg in "${dgs[@]}"; do
    # increment port in case multiple runs on same machine
    ((PORT++))
    # increment actual seed
    ((SEED++))

    TAG="$DATASET-$sd-$lr-$dg"

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
    cmd="OMP_NUM_THREADS=2 ./main.py \
        --tag experiment-$TAG \
        --seed $SEED \
        --port $PORT \
        --num-shards 2 \
        --epochs 5000 \
        --timeout 42300 \
        train \
        --dataset $DATASET \
        --workspace 8 \
        --stages 2 \
        --order 2 \
        --degree $dg \
        --optimizer adam \
        --learning-rate $lr \
        --batch-size 50"
        
    srun --pty -p skylake --time 12:00:00 --ntasks 1 --cpus-per-task 4 --threads-per-core 1 bash -c "$cmd"
    
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