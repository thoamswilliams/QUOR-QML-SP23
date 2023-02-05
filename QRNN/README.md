# QRNN Pytorch Implementation

## Setup

The code requires only a rudimentary pytorch/tensorboard installation; training is (so far) only done on the CPU, so no CUDA setup is required.
For instance, the following command installs the basic necessities.

    pip install torch tensorboard

Further packages required

    pip install colorful pandas

Optional packages:

    pip install pytest                                   # for tests
    pip install jupyter matplotlib opentsne umap-learn   # to run jupyter notebooks for MNIST data augmentation


## Folder Structure

    ./RVQE/             # implementation of QRNN as pytorch model
    ./RVQE/datasets/    # datasets and necessary resources

    ./runs/
    ./locks/
    ./checkpoints/      # empty folders used for tensorboard logs, checkpoints, and for synchronizing experiments on multi-rank systems

    ./notebooks/        # contains jupyter notebook for MNIST t-SNE augmentation,
                        # model test set evaluations,
                        # and RNN/LSTM reference implementation for DNA sequence test
    ./results/          # results used in paper; contains mathematica notebooks for plots,
                        # training data in csv format used to produce plots, and
                        # pre-trained models that performed best

    ./main.py           # main training program

    ./*.sh              # various experiment presets used in the paper
                        # modify these to match your training environment



## Running

    ./main.py --help
    ./main.py train --help
    ./main.py resume --help

    pytest              # executes a series of tests

    ./main.py train     # executes a simple default training task (default parameters from --help)


For instance, to train the umap augmented MNIST dataset on an 8-core machine, execute the following:

    OMP_NUM_THREADS=2 ./main.py \
        --tag experiment-test \
        --seed 42 \
        --port 20023 \
        --num-shards 4 \
        --epochs 10000 \
        train \
        --dataset mnist-umap-d2-r4 \
        --workspace 6 \
        --stages 2 \
        --order 2 \
        --degree 3 \
        --optimizer adam \
        --learning-rate 0.005 \
        --batch-size 16

Note that memory requirements go linear in the number of shards (ranks); linear in the number of stages; and grow exponentially in the workspace size. There is more parameters that control training, stopping, etc..

When training is interrupted, a checkpoint can simply be restarted with

    ./main.py resume checkpoint-name.tar.gz

A few of the parameters can be overridden, e.g. a new learning rate can be set.


## Datasets

No external datasets are necessary; the implemented ones can be elected with the `--dataset` switch, and all required files are precomputed. Some datasets for memorizing sequences require a certain batch size or number of shards or combinations thereof; the program will complain if this is manually set to something invalid (e.g., `--dataset simple-seq` makes the QRNN learn precisely two sequences, so the setup is such that either `--num-shards 2 ... --batch-size 1` or `--num-shards 1 ... --batch-size 2` is allowed). Most datasets do not have such a restriction.

For MNIST, the batch size indicates how many samples _of each digit_ are presented.

MNIST data augmentation with t-SNE was too slow in mathematica alone; hence it is broken up into two steps, data preparation using the `./notebooks/mnist.nb` MM file, and `./notebooks/mnist-tsne.ipynb` file which performs the heavy lifting. The created data can then be re-imported in MM, and exported for use in training. All of these files are included for transparency, and do not have to be executed before using the code.


## Experiments

The root folder contains more experiments than were run for the paper. Many of them perform grid search over parameter ranges, or seeds, to get statistics on learning performance. They are often set up to match a certain cluster, so modify to suit your needs before execution. Training data produced with them was used directly in producing the datasets under `./results/`, and the best MNIST models are collected in `./results/best/`, to be evaluated on their respective test sets in the `./notebooks/` folder.