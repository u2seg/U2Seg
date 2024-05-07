# Clustering for Masked Instances

This repository provides code to perform clustering on masked instances dataset.

## Getting Started

To run the clustering, execute the following command:

```
cd selective_labeling
python usl-imagenet.py --cfg configs/ImageNet_usl_dino_0.2.yaml
```

The results of the clustering will be saved in the following file:

```
saved/imagenet_usl_dino_0.2/cluster_labels_decode.json
```

The clustering categories could be set using `NUM_SELECTED_SAMPLES` in the config file `configs/ImageNet_usl_dino_0.2.yaml`. And the feature calculating batchsize could be set using `BATCH_SIZE`.

