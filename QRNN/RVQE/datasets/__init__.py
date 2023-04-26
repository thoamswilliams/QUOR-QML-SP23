from .shakespeare import DataShakespeare
from .simple import DataSimpleSequences, DataSimpleQuotes
from .elman import DataElmanXOR, DataElmanLetter
from .mnist import *
from .mnistpca import *
from .mnistumap import *
from .dna import *
from .twitter import TwitterSentiment
from .amazon import AmazonSentiment
from .twitter_reduced import TwitterSentimentReduced

all_datasets = {
    "simple-seq": DataSimpleSequences,
    "simple-quotes": DataSimpleQuotes,
    "elman-xor": DataElmanXOR,
    "elman-letter": DataElmanLetter,
    "mnist01": DataMNIST01,
    "mnist36": DataMNIST36,
    "mnist8": DataMNIST8,
    "mnist01-ds": DataMNIST01ds,
    "mnist36-ds": DataMNIST36ds,
    "mnist8-ds": DataMNIST8ds,
    "mnist01-ds-lrg": DataMNIST01ds_lrg,
    "mnist36-ds-lrg": DataMNIST36ds_lrg,
    "mnist8-ds-lrg": DataMNIST8ds_lrg,
    "mnist-pca-r2-p2": DataMNISTPCA_r2p2,
    "mnist-pca-r2-p5": DataMNISTPCA_r2p5,
    "mnist-pca-r2-p8": DataMNISTPCA_r2p8,
    "mnist-pca-r3-p2": DataMNISTPCA_r3p2,
    "mnist-pca-r3-p5": DataMNISTPCA_r3p5,
    "mnist-pca-r3-p8": DataMNISTPCA_r3p8,
    "mnist-pca-r4-p2": DataMNISTPCA_r4p2,
    "mnist-pca-r4-p5": DataMNISTPCA_r4p5,
    "mnist-pca-r4-p8": DataMNISTPCA_r4p8,
    "mnist-umap-d2-r2": DataMNISTumap_d2r2,
    "mnist-umap-d2-r3": DataMNISTumap_d2r3,
    "mnist-umap-d2-r4": DataMNISTumap_d2r4,
    "mnist-umap-d2-r5": DataMNISTumap_d2r5,
    "mnist-umap-d2-r8": DataMNISTumap_d2r8,
    "mnist-umap-d3-r2": DataMNISTumap_d3r2,
    "mnist-umap-d3-r3": DataMNISTumap_d3r3,
    "mnist-umap-d3-r4": DataMNISTumap_d3r4,
    "mnist-umap-d3-r5": DataMNISTumap_d3r5,
    "mnist-umap-d3-r8": DataMNISTumap_d3r8,
    "mnist-umap-d4-r2": DataMNISTumap_d4r2,
    "mnist-umap-d4-r3": DataMNISTumap_d4r3,
    "mnist-umap-d4-r4": DataMNISTumap_d4r4,
    "mnist-umap-d4-r5": DataMNISTumap_d4r5,
    "mnist-umap-d4-r8": DataMNISTumap_d4r8,
    "mnist-even-odd": DataMNIST_EvenOdd,
    "mnist01-gen": DataMNIST01_Gen,
    "shakespeare": DataShakespeare,
    "dna": DataDNA,
    "twitter": TwitterSentiment,
    "twitter-short": TwitterSentimentReduced,
    "amazon": AmazonSentiment,
}
