#!/usr/bin/env bash
# mnist 8 digits

seeds=( 1 2 3 )
datasets=( "mnist-umap-d2-r2" "mnist-umap-d2-r3" "mnist-umap-d2-r4"  "mnist-umap-d3-r2" "mnist-umap-d3-r3" "mnist-umap-d3-r4"  "mnist-umap-d4-r2" "mnist-umap-d4-r3" "mnist-umap-d4-r4" )
lrs=( 0.1 0.03 0.01 )
workspaces=( 5 6 7 )
degrees=( 2 3 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"


trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s


PORT=37777
SEED=100042


for sd in "${seeds[@]}"; do
for dataset in "${datasets[@]}"; do
for lr in "${lrs[@]}"; do
for ws in "${workspaces[@]}"; do
for dg in "${degrees[@]}"; do
    # increment port in case multiple runs on same machine
    ((PORT++))
    # different actual seed every run
    ((SEED++))

    TAG="pool-$dataset-$sd-$lr-$ws-$dg"

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
        --epochs 500 \
        --timeout 7200 \
        train \
        --dataset $dataset \
        --workspace $ws \
        --stages 2 \
        --order 2 \
        --degree $dg \
        --optimizer adam \
        --learning-rate $lr \
        --batch-size 64"
    
    srun --pty -p skylake --time 2:05:00 --ntasks 1 --cpus-per-task 4 --threads-per-core 1 bash -c "$cmd"

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
done
