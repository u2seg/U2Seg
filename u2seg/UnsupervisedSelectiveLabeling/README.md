# Unsupervised Selective Labeling for More Effective Semi-Supervised Learning
by [Xudong Wang*](https://people.eecs.berkeley.edu/~xdwang/), [Long Lian*](https://tonylian.com/), and [Stella X. Yu](http://www1.icsi.berkeley.edu/~stellayu/) at UC Berkeley/ICSI. (*: co-first authors)

[Arxiv Paper](https://arxiv.org/abs/2110.03006) | [ECCV Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900423.pdf) | [**Poster**](https://people.eecs.berkeley.edu/~longlian/usl_poster.pdf) | [**Video**](https://people.eecs.berkeley.edu/~longlian/usl_video.html) | [Citation](#citation)

*European Conference on Computer Vision* (ECCV), 2022.

This work is also presented in [CV in the Wild workshop](https://computer-vision-in-the-wild.github.io/eccv-2022/) in ECCV 2022.

![Teaser](assets/teaser.png)

This repository contains the code for USL on CIFAR. Other implementations are coming soon.

## Use pre-selected labeled split in your SSL method for a **free boost**
Note that even if you only work on proposing new SSL methods, **you can try our unsupervised selected samples as the labeled data with your SSL methods to get a free boost with the same number of labeled samples.** We have pre-extracted the labeled split and nothing is needed to run.

Our unsupervised selection (USL and USL-T) is available in a plug-and-play format for CIFAR and ImageNet [here](#samples-selected-by-usl-and-usl-t). We have shown that our method works off-the-shelf with SimCLR, SimCLR-CLD, FixMatch, MixMatch, CoMatch, etc and it should also work for other newer methods.

For further information regarding the paper, please contact [Xudong Wang](mailto:xdwang@eecs.berkeley.edu). For information regarding the code and implementation, please contact [Long Lian](mailto:longlian@berkeley.edu).

## News
* **USL-T is fully supported** in this implementation. Pretraining weights and trained models are available.
* Selected sample indices on USL-T are added for reference (note that USL-T is training-based and thus gives different results on different runs)
* **Poster** and **video** are added (see above)
* ImageNet scripts, intermediate results, final results, and FixMatch checkpoints are added
* Provided CLD pretrained model and reference selections
* Initial Implementation

## Supported Methods
- [x] USL
- [x] USL-T

## Supported SSL Methods
- [x] FixMatch
- [x] SimCLRv2
- [x] SimCLRv2-CLD

## Supported Datasets
- [x] CIFAR-10
- [x] CIFAR-100
- [x] ImageNet100
- [x] ImageNet

## Preparation
Install the required packages:
```
pip install -r requirements.txt
```

You also need to install `clip` if you want to use `clip` models:
```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

For ImageNet, you need to change the data path to your data path in the config. For CIFAR, it will download the data automatically.

## Run USL
### CIFAR-10/100
#### Download our CLD pretrained model on CIFAR for USL
```shell
mkdir selective_labeling/pretrained
cd selective_labeling/pretrained
# CLD checkpoint on CIFAR-10
wget https://people.eecs.berkeley.edu/~longlian/files/cifar10_ckpt_epoch_200.pth
# CLD checkpoint on CIFAR-100
wget https://people.eecs.berkeley.edu/~longlian/files/cifar100_ckpt_epoch_200.pth
```

#### Perform USL on CIFAR-10/100
```shell
cd selective_labeling
# CIFAR-10
python usl-cifar.py --cfg configs/cifar10_usl.yaml
# CIFAR-100
python usl-cifar.py --cfg configs/cifar100_usl.yaml
```

#### Evaluate USL on CIFAR-10/100 with SimCLRv2-CLD
You can also find the config for SimCLR in `semisup-simclrv2-cld/configs`. The implementation for FixMatch is in `semisup-fixmatch-cifar`.
```shell
cd semisup-simclrv2-cld
# CIFAR-10
python fine_tune.py --cfg semisup-simclrv2-cld/configs/cifar10_usl-t_finetune.yaml
# CIFAR-100
python fine_tune.py --cfg semisup-simclrv2-cld/configs/cifar100_usl-t_finetune.yaml
```

### ImageNet
#### Download our CLD pretrained model on ImageNet for USL (USL-MoCo only)
```
mkdir selective_labeling/pretrained
cd selective_labeling/pretrained
# MoCov2 checkpoint on ImageNet (with EMAN as normalization)
wget https://eman-cvpr.s3.amazonaws.com/models/res50_moco_eman_800ep.pth.tar
```

CLIP models will be downloaded at the first run with USL-CLIP config.

#### Use pre-computed intermediate results (Recommended)
This step is optional but recommended to refrain from recomputing the feature from the dataset. Furthermore, it relieves the non-deterministic behavior from obtaining feature, kNN, and clustering, since GPU ops lead to non-deterministic behavior (even though seed is set), which is more prominent for large datasets where more compute is used.

We provide intermediate results after obtaining feature, kNN, k-Means, and the final selected indices in numpy and csv format. Intermediate results can be obtained by:
<details>
<summary>USL-MoCo experiments</summary>

```
mkdir -p selective_labeling/saved/imagenet_usl_moco_0.2
cd selective_labeling/saved/imagenet_usl_moco_0.2
wget https://people.eecs.berkeley.edu/~longlian/files/usl_imagenet_moco_0.2_intermediate.zip
unzip usl_imagenet_moco_0.2_intermediate.zip
cd ../../..
```

Please also download the precomputed MoCov2 feature [here](https://drive.google.com/file/d/1r8hJ_tuQ7Eta2eVTmZQT1W59FzLkvDZ3/view?usp=share_link) and unzip `memory_feats_list.npy` into `selective_labeling/saved/imagenet_usl_moco_0.2`.

</details>

<details>
<summary>USL-CLIP experiments</summary>

```
mkdir -p selective_labeling/saved/imagenet_usl_clip_0.2
cd selective_labeling/saved/imagenet_usl_clip_0.2
wget https://people.eecs.berkeley.edu/~longlian/files/usl_imagenet_clip_0.2_intermediate.zip
unzip usl_imagenet_clip_0.2_intermediate.zip
cd ../../..
```

Please also download the precomputed CLIP feature [here](https://drive.google.com/file/d/1V47BFvWs9uQYO_sOGDqj3RrslMwjaEsO/view?usp=share_link) and unzip `memory_feats_list.npy` into `selective_labeling/saved/imagenet_usl_clip_0.2`.

</details>

You can also obtain the intermediate and final results [here](https://people.eecs.berkeley.edu/~longlian/files/usl_imagenet.html).

If this step is skipped (i.e., compute everything from scratch), `RECOMPUTE_ALL` and `RECOMPUTE_NUM_DEP` need to be set to True in the config.

#### Perform USL on ImageNet
##### USL-MoCo experiments
```
cd selective_labeling
python usl-imagenet.py
```

##### USL-CLIP experiments
```
cd selective_labeling
python usl-imagenet.py --cfg configs/ImageNet_usl_clip_0.2.yaml
```

#### Evaluate USL on ImageNet with FixMatch
Use the script [here](https://github.com/amazon-science/exponential-moving-average-normalization). The output `csv` file is compatible with the split of labeled dataset.

## Run USL-T
To make our results more comparable, we release our selected indices and trained models, see [model zoo](#model-zoo) below.
### CIFAR-10/100
#### Download models trained with unsupervised learning methods
```
mkdir -p usl-t_pretraining/pretrained
wget https://people.eecs.berkeley.edu/~longlian/files/pretrained_kNN_cifar10.tgz
tar zxvf pretrained_kNN_cifar10.tgz

wget https://people.eecs.berkeley.edu/~longlian/files/pretrained_kNN_cifar100.tgz
tar zxvf pretrained_kNN_cifar100.tgz
```
#### Perform USL-T (training and sample selection)
* USL-T Training (also called pretraining because it is prior to training in semi-supervised learning):
```
cd usl-t_pretraining
# CIFAR-10
python usl-t-cifar-pretrain.py --cfg configs/cifar10_usl-t_pretrain.yaml
# CIFAR-100
python usl-t-cifar-pretrain.py --cfg configs/cifar100_usl-t_pretrain.yaml
cd ..
```
* Sample selection:
```
cd ../selective_labeling
# CIFAR-10
python usl-t-cifar.py --cfg configs/cifar10_usl-t.yaml
# CIFAR-100
python usl-t-cifar.py --cfg configs/cifar100_usl-t.yaml
cd ..
```
#### Evaluate USL-T on CIFAR-10/100 on SimCLR-CLD
```
# CIFAR-10
python fine_tune.py --cfg configs/cifar10_usl-t_finetune.yaml
# CIFAR-100
python fine_tune.py --cfg configs/cifar100_usl-t_finetune.yaml
```
### ImageNet
#### Download models trained with unsupervised learning methods
```
mkdir -p usl-t_pretraining/pretrained
wget https://people.eecs.berkeley.edu/~longlian/files/pretrained_kNN_imagenet.tgz
tar zxvf pretrained_kNN_imagenet.tgz

wget https://people.eecs.berkeley.edu/~longlian/files/pretrained_kNN_imagenet100.tgz
tar zxvf pretrained_kNN_imagenet100.tgz
```
#### Perform USL-T (training and sample selection)
* Training:
```
cd usl-t_pretraining
# ImageNet100
python usl-t-imagenet-pretrain.py --cfg configs/ImageNet100_usl-t_pretrain.yaml
# ImageNet
python usl-t-imagenet-pretrain.py --cfg configs/ImageNet_usl-t_pretrain.yaml
cd ..
```
* Sample selection:
```
cd ../selective_labeling
# CIFAR-10
python usl-t-imagenet.py --cfg ImageNet100_usl-t_0.3.yaml
# CIFAR-100
python usl-t-imagenet.py --cfg ImageNet_usl-t_0.2.yaml
cd ..
```
#### Evaluate USL-T on ImageNet100/ImageNet with SimCLR
* SimCLR weights `r50_1x_sk0.pth` can be downloaded at [here](https://people.eecs.berkeley.edu/~longlian/files/r50_1x_sk0.pth.tgz) and should be put under pretrained.
```
cd semisup-simclrv2
# ImageNet-100
python fine_tune.py --lr 0.16 --seed 0 --name-to-save "usl-t_0.3p_imagenet100" --trainindex_x "[path to train_0.3p_gen_imagenet100_usl-t_0.3_index.csv]" -j 16 --batch-size 256 --eval-freq 48 --epochs 48 --warmup-epochs 0 --weight-decay 0.0001 --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0 --pretrain pretrained/r50_1x_sk0.pth --roll-data 20 --amp-opt-level O1 --roll-data 40 --num-classes 100 --freeze-backbone [Path to ImageNet 100]

# ImageNet
python fine_tune.py --lr 0.16 --seed 0 --name-to-save "usl-t_0.2p_imagenet" --trainindex_x "[path to train_0.2p_gen_imagenet_usl-t_0.2_index.csv]" -j 16 --batch-size 512 --eval-freq 6 --epochs 12 --warmup-epochs 0 --weight-decay 0 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrain pretrained/r50_1x_sk0.pth --multi-epochs-data-loader --amp-opt-level O1 --no-lr-decay [Path to ImageNet]
```
## Samples selected by USL and USL-T
### CIFAR-10
Note that both USL and USL-T have some randomness due to non-deterministic computations in GPU and could vary between each run or server, despite setting the seed. Therefore, we release samples selected by USL run on our end. The indices are in torchvision CIFAR dataset order (the order commonly used in most PyTorch SSL implementations).

These are the instance indices by the torch CIFAR-10 dataset.
<details>
<summary>Random indices on CIFAR-10</summary>
Seed 1 (class distribution [1, 6, 5, 3, 1, 3, 5, 4, 6, 6]):

```
[26247, 35067, 34590, 16668, 12196,  2600,  9047,  2206, 25607,
11606,  3624, 43027, 15190, 25816, 26370,  1281, 29433, 36256,
34217, 39950,  6756, 26652,  3991, 40312, 24580,  4949, 18783,
39205, 23784, 39508, 19062, 48140, 11314,   766, 39319, 15299,
10298, 25573, 18750, 19298]
```

Seed 2 (class distribution [4, 2, 6, 5, 7, 1, 5, 2, 4, 4]):

```
[23656, 27442, 40162,  8459,  8051, 42404,    89,  1461, 13519,
42536, 20817, 45479,  3121, 36502, 40119, 35971,  8784, 14084,
4063, 18730, 17763, 29366, 43841, 10741,  3986, 40475,  8470,
35621, 30892, 27652, 35359, 24435, 47853,  8835,  6572, 36456,
8750, 21067,  4337, 24908]
```

Seed 5 (class distribution [6, 6, 2, 3, 5, 3, 5, 2, 2, 6]):

```
[24166, 42699, 15927,  7473,  5070, 33926, 21409,  9495, 16235,
35747, 46288, 13560, 29644, 28992, 35350, 43077, 35757, 24106,
26555, 22478,  1951, 29145, 33373, 10043, 21988, 37116, 15760,
48939, 29761,  3702,  3273,  4175, 30998, 31012,  8754, 33953,
22206, 28896, 31669, 19275]
```

Seed 3 and 4 are not selected because seed 3 and seed 4 do not lead to instances of 10 classes for **random selection** and thus the comparison would not bring us much insights. Note that seed 3 and 4 lead to instances of 10 classes for **our selection**.

Note that these can be obtained by `selective_labeling/random-cifar.py`.
</details>

<details>
<summary>USL indices on CIFAR-10</summary>
Seed 1 (class distribution [5, 4, 5, 2, 2, 5, 5, 4, 3, 5]):

```
[ 3301, 37673, 33436, 28722, 10113,  5286, 21957, 13485,   445,
48678, 43647, 27879, 39987, 14374, 32536, 14741, 38215, 22102,
23082, 16734,  7409,   881, 10912, 37632, 39363,  7119,  6203,
28474, 25040, 43960, 24780, 45742, 49642, 25728,  9297, 21092,
4689,  4712, 48444, 30117]
```

Seed 2 (class distribution [4, 4, 4, 3, 3, 5, 4, 5, 3, 5]):

```
[19957, 40843, 45218,   881,  4557,  6203, 11400, 14374, 27595,
21092, 41009, 38215, 35471, 49642, 25728, 28722, 17094, 48678,
43960, 39363, 43647,  3907, 16734, 48023,  3301, 22102, 37632,
21130,  3646, 14741,  7127,  9297, 11961, 39987,  4712, 45568,
39908, 23505, 48421, 33436]
```

Seed 5 (class distribution [4, 5, 4, 3, 3, 4, 4, 4, 4, 5]):

```
[38215, 43213, 39363, 27965,   445, 16734, 14374,   914, 17063,
45918,  3301,  5286, 32457, 19867, 48678, 10455, 43647, 10912,
28722,  4712, 29946,  1221,  3907, 10110, 20670, 13410,  4689,
49642, 10018, 41210, 43755, 46227, 11961, 15682, 45742, 21092,
9692, 48023, 14741,  2703]
```

Seed 3 and 4 are not selected because seed 3 and seed 4 do not lead to instances of 10 classes for **random selection** and thus the comparison would not bring us much insights. Note that seed 3 and 4 lead to instances of 10 classes for **our selection**.

Note that these can be obtained by `selective_labeling/usl-cifar.py`.
</details>

<details>
<summary>USL-T indices on CIFAR-10</summary>
Class distribution [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]:

```
[7998, 45774, 27115, 8389, 28558, 8454, 12390, 42528, 28249, 
12885, 25101, 39912, 19571, 7904, 43637, 3267, 6935, 21794, 
24489, 13999, 24554, 19979, 1573, 36597, 5403, 44836, 29500, 
16935, 9408, 47504, 35673, 20778, 44636, 37123, 49130, 8086, 
39994, 8499, 48597, 7753]
```

</details>

### CIFAR-100
Note that both USL and USL-T have some randomness due to non-deterministic computations in GPU and could vary between each run or server, despite setting the seed. Therefore, we release samples selected by USL run on our end. The indices are in torchvision CIFAR dataset order (the order commonly used in most PyTorch SSL implementations).

<details>
<summary>Random indices on CIFAR-100</summary>
Class distribution [ 2,  4,  4,  3,  3,  4,  8,  7,  6,  1,  2,  3,  4,  3,  4,  4,  2,
        3,  2,  0,  1,  2,  3,  5,  4,  4,  6,  5,  7,  2,  5,  2,  2, 10,
        7,  3,  2,  2,  2,  7,  3,  7,  4,  1,  5,  4,  4,  2,  4,  4, 13,
        5,  4,  4,  4,  3,  2,  3,  7,  4,  4,  1,  4,  7,  2,  9,  3,  5,
        3,  4,  5,  4,  1,  7,  6,  3,  2,  3,  6,  2,  4,  1,  2,  2,  3,
        3,  5,  3,  7,  2,  6,  5,  6,  5,  4,  3,  5,  7,  5,  4]:

```
[24166, 42699, 15927, 7473, 5070, 33926, 21409, 9495, 16235, 35747, 46288, 13560, 29644, 28992, 35350, 43077, 35757, 24106, 26555, 22478, 1951, 29145, 33373, 10043, 21988, 37116, 15760, 48939, 29761, 3702, 3273, 4175, 30998, 31012, 8754, 33953, 22206, 28896, 31669, 19275, 49843, 36020, 22750, 7287, 17858, 2307, 44346, 12850, 23481, 19587, 40928, 40633, 35091, 42855, 27790, 5834, 3866, 19436, 31884, 37267, 23120, 33086, 4440, 27294, 3067, 45307, 7740, 30408, 12046, 7469, 8633, 991, 36400, 33945, 43384, 619, 295, 6399, 399, 34346, 43381, 21838, 3031, 951, 15485, 41403, 18809, 37656, 31150, 42183, 25731, 27322, 26964, 12174, 10713, 21227, 11015, 39811, 40368, 28291, 29288, 16029, 21946, 9705, 21352, 33934, 37044, 24317, 36476, 25635, 7886, 4473, 41868, 26742, 29129, 38795, 1719, 4790, 31217, 14773, 16794, 47168, 27679, 42505, 10825, 31802, 49931, 5575, 28614, 6435, 44541, 5085, 17016, 42117, 38402, 29967, 25431, 25062, 33502, 15008, 10109, 8426, 28882, 9342, 6308, 35548, 12832, 24762, 33338, 42121, 46170, 17984, 42630, 29659, 7446, 28009, 23580, 38684, 43558, 888, 20080, 11419, 46386, 34533, 6104, 17931, 4785, 4495, 47383, 17126, 29605, 37074, 14999, 27624, 20763, 33869, 3823, 36053, 20029, 19489, 40526, 9001, 36766, 5168, 38401, 47320, 24840, 43386, 35017, 47749, 36024, 33553, 45709, 28068, 11612, 26724, 19599, 17505, 30267, 28034, 6525, 20387, 42058, 13573, 20016, 18810, 6579, 37825, 11653, 43824, 4535, 9211, 17387, 32223, 39727, 26713, 27986, 42511, 17841, 42043, 17476, 30838, 22163, 9398, 11837, 17402, 44474, 46779, 34172, 49624, 13008, 814, 14162, 37951, 17370, 47018, 23821, 40787, 4148, 49582, 1733, 14839, 15748, 23940, 2074, 17809, 10011, 28631, 26640, 6567, 1893, 17286, 41825, 4060, 29351, 42622, 34632, 24497, 41630, 8892, 43350, 21663, 36068, 17874, 14887, 6986, 574, 40534, 6410, 19994, 40707, 4914, 25078, 33826, 20188, 40485, 42960, 42533, 1500, 16193, 13313, 10466, 26284, 43916, 47801, 15414, 7746, 44384, 21855, 39092, 29991, 18908, 25548, 19031, 16423, 31446, 24607, 46107, 37059, 48936, 46619, 37714, 34645, 49173, 20050, 23766, 15251, 29339, 37963, 9887, 16646, 8353, 5072, 14002, 16087, 24325, 17034, 24853, 3059, 8437, 10524, 32249, 21469, 12733, 30641, 43425, 402, 20134, 33961, 31995, 14819, 12533, 43068, 49730, 18873, 49475, 49092, 49282, 26051, 9070, 3479, 9355, 8142, 27134, 29154, 44421, 134, 7139, 42677, 45885, 35052, 28516, 9571, 17065, 38029, 45069, 26723, 12765, 1041, 33472, 43872, 5377, 45825, 18063, 12945, 32732, 30400, 2353, 44644, 21439, 14658, 25817, 35673, 36260, 22863, 25272, 3251, 16628, 36201, 49088, 33481, 36333, 39695, 32444, 28474, 12797, 32118, 37912, 22011, 3196, 36365, 12075, 47069, 15728, 534, 44533, 3113, 12785, 43732, 25765]
```
</details>

<details>
<summary>USL indices on CIFAR-100</summary>
Class distribution [5, 4, 3, 4, 4, 4, 5, 4, 4, 2, 3, 8, 4, 2, 8, 2, 5, 3, 2, 2, 2, 2,
       2, 2, 5, 4, 4, 6, 5, 2, 3, 3, 5, 5, 4, 3, 3, 4, 6, 3, 1, 5, 6, 6,
       4, 5, 3, 3, 6, 5, 4, 3, 9, 2, 7, 2, 2, 3, 5, 1, 4, 3, 5, 3, 4, 7,
       2, 6, 4, 3, 4, 4, 4, 3, 5, 3, 4, 4, 4, 5, 3, 6, 3, 2, 4, 2, 4, 5,
       5, 7, 4, 4, 5, 4, 6, 4, 7, 2, 6, 3]:

```
[22206, 15361, 24078, 22803, 10747, 21048, 27300, 1106, 31810, 41199, 8736, 47459, 5900, 23184, 210, 47195, 40234, 7629, 6458, 31732, 41228, 27974, 23980, 30146, 1368, 31891, 46592, 40815, 20847, 8155, 18011, 29160, 22693, 46498, 37427, 2006, 22506, 2302, 11743, 4500, 32338, 31112, 26553, 36430, 47351, 41152, 46279, 46028, 2149, 11125, 48700, 12325, 47166, 47410, 4290, 33275, 11410, 23998, 22278, 13267, 715, 35417, 8517, 20521, 7508, 9462, 27757, 20893, 49175, 29673, 6477, 8255, 36917, 21076, 30512, 7195, 17986, 9877, 31824, 4340, 40584, 20741, 51, 12714, 48609, 37284, 45509, 40413, 47958, 14536, 23990, 1871, 44686, 48318, 5245, 35253, 2519, 20553, 28140, 13514, 13276, 21972, 5423, 47901, 17726, 40743, 7859, 47350, 43054, 31207, 49312, 4780, 9741, 13351, 9022, 38961, 10975, 40218, 16656, 12985, 35080, 21969, 40436, 8160, 20180, 17159, 49497, 4600, 12640, 9306, 27679, 36206, 12959, 32090, 3049, 30026, 25417, 40116, 34101, 21479, 9083, 5768, 19417, 3441, 18566, 33689, 49152, 36668, 22747, 6087, 16483, 20472, 14240, 6790, 32485, 34823, 5591, 26852, 22707, 12774, 45823, 34676, 10430, 10683, 39252, 25088, 6682, 12444, 26152, 3818, 34555, 3232, 47836, 12584, 18753, 28289, 46985, 40834, 22674, 25657, 29516, 29945, 48491, 18271, 36346, 44892, 2599, 16124, 37785, 19595, 4331, 12955, 31965, 48730, 563, 26109, 2580, 27360, 8033, 32350, 29868, 12805, 25414, 32947, 612, 38133, 41872, 14239, 28040, 43421, 49358, 45928, 17874, 30130, 8343, 48325, 34903, 16680, 22661, 5929, 868, 28565, 25180, 27498, 25216, 30783, 17604, 41793, 14387, 36673, 24815, 9966, 44087, 27517, 34323, 19583, 13727, 28632, 13872, 9738, 42044, 12358, 2878, 44860, 30303, 30007, 15769, 2538, 27205, 33147, 5308, 41664, 14280, 8903, 38737, 30401, 7952, 19026, 22712, 14610, 23180, 43114, 1843, 30665, 16331, 19947, 24107, 3009, 13682, 16944, 30998, 32069, 3618, 2679, 13812, 11975, 1846, 34370, 32934, 36766, 47276, 38654, 12141, 22821, 38348, 30562, 6064, 4892, 39669, 35458, 43696, 10557, 5427, 1707, 45805, 7727, 44672, 47255, 47831, 12951, 16570, 36288, 8138, 49012, 14205, 34474, 41050, 10882, 34830, 1627, 3784, 25923, 11883, 18593, 14734, 37586, 8442, 46250, 30750, 41352, 10437, 39812, 8550, 49114, 25148, 6764, 16857, 16947, 17439, 27469, 34486, 5103, 27571, 46567, 433, 5576, 39624, 32810, 38208, 10233, 34721, 35500, 7198, 16577, 36927, 1364, 22608, 48528, 41758, 32909, 44279, 5844, 36946, 3974, 12468, 44529, 5998, 2176, 45598, 49136, 22725, 26274, 19532, 17104, 14411, 2376, 36061, 26912, 19367, 38488, 7345, 23549, 19293, 20523, 28316, 21374, 41267, 44467, 19224, 34958, 29115, 9603, 2237, 44402, 41499, 23403, 38756, 5096, 6143, 13455, 29615, 25857, 15402, 2151, 10046, 790, 21690, 7772, 46735, 5318]
```
</details>

<details>
<summary>USL-T indices on CIFAR-100</summary>
Class distribution [5, 5, 4, 4, 4, 5, 6, 2, 4, 3, 2, 5, 2, 4, 3, 3, 4, 5, 6, 3, 3, 6,
3, 3, 6, 2, 4, 3, 3, 4, 4, 5, 4, 4, 2, 3, 4, 3, 5, 6, 2, 6, 3, 6,
5, 4, 0, 7, 4, 4, 5, 3, 4, 6, 2, 1, 3, 3, 3, 2, 4, 6, 7, 5, 2, 4,
4, 6, 4, 3, 6, 5, 4, 1, 5, 6, 4, 3, 3, 2, 3, 5, 4, 2, 5, 5, 3, 4,
6, 3, 5, 4, 3, 2, 6, 7, 4, 5, 5, 6]:

```
[21762, 16447, 49147, 10389, 18413, 35442, 33474, 46464, 8338, 14525, 34628, 47555, 29499, 33006, 15704, 39226, 22682, 20220, 39482, 28379, 11077, 5201, 46545, 48978, 46444, 23451, 6294, 20971, 42100, 43381, 6904, 9155, 8731, 25248, 11694, 22934, 15677, 41072, 36827, 19802, 4593, 4847, 8897, 21525, 12729, 8578, 28329, 20981, 49514, 2711, 30004, 38806, 1723, 2943, 26878, 7087, 19209, 49254, 6446, 31422, 20969, 26117, 23572, 6410, 26734, 40907, 8296, 31978, 25857, 31126, 15024, 35952, 19102, 34214, 37404, 33551, 8454, 21571, 37177, 37070, 22569, 6519, 47011, 43725, 32006, 12986, 34865, 4324, 9779, 46093, 22246, 41444, 46387, 13288, 24731, 10286, 46384, 7574, 24220, 21826, 6140, 19237, 616, 27663, 46045, 30240, 48781, 35184, 6081, 42246, 1458, 48719, 9891, 14490, 21687, 24246, 13089, 8257, 15239, 1873, 39691, 25481, 24495, 8455, 38323, 13807, 6036, 25472, 14880, 2797, 12212, 48310, 47899, 39238, 21455, 7343, 9462, 508, 38205, 9973, 29254, 1797, 34802, 11474, 44070, 17242, 38568, 4101, 44808, 20624, 19976, 43584, 17020, 10984, 10756, 35500, 26976, 25107, 30896, 26283, 40935, 35799, 41217, 12935, 24193, 43840, 7911, 28369, 35101, 1843, 45917, 14696, 2868, 49554, 28012, 35154, 2355, 24112, 8203, 31274, 6933, 29179, 42669, 31020, 12493, 45961, 1476, 17077, 15287, 41520, 21172, 39689, 33892, 20100, 11246, 33001, 29609, 3614, 29688, 34446, 13096, 7953, 12863, 38488, 23459, 38311, 45879, 12076, 3853, 14705, 25755, 35035, 18426, 42637, 4375, 29598, 24182, 27208, 47842, 46663, 49903, 24443, 1681, 22698, 19446, 16168, 22036, 7260, 45326, 30289, 40054, 25173, 12881, 39212, 16684, 46354, 17429, 30140, 38577, 43055, 6243, 41306, 23593, 16979, 31824, 48917, 2340, 49058, 44091, 25068, 20199, 31109, 6841, 31451, 4944, 18916, 15397, 39035, 16920, 34210, 21804, 13167, 13405, 8312, 5271, 40302, 36489, 18208, 23850, 16257, 5103, 33416, 26902, 19625, 7576, 30412, 27098, 7135, 46662, 46237, 2013, 13923, 39640, 13251, 2812, 16878, 48923, 24540, 30728, 26896, 20537, 37602, 48008, 7862, 30725, 3899, 10048, 29422, 44526, 38679, 13503, 22859, 13350, 10245, 27467, 21403, 4895, 39100, 44376, 15669, 37786, 1323, 807, 28568, 18199, 4054, 31353, 4424, 4367, 21359, 7092, 15131, 41729, 33263, 16688, 23562, 31033, 26816, 24209, 32544, 44294, 5649, 34927, 44546, 32031, 47751, 17975, 30215, 30740, 49699, 28670, 40664, 2570, 27057, 19430, 34764, 4413, 15967, 32042, 41441, 48957, 22367, 18955, 32866, 41457, 7634, 18922, 31547, 19411, 33710, 42915, 11569, 2808, 9905, 32939, 46386, 21291, 45922, 43343, 46521, 742, 36504, 35395, 4119, 22490, 49340, 40765, 15719, 35317, 38737, 30610, 4798, 283, 45332, 41408, 24516, 48768, 32476, 40236, 23567, 13674, 30089, 21409, 40468, 18040, 46783, 29182, 28345, 14026, 34255]
```
</details>

### ImageNet
The random selection and USL selected samples on ImageNet (in `csv` format) could be obtained [here](https://people.eecs.berkeley.edu/~longlian/files/usl_imagenet.html).
**Update**: USL-T selected samples on ImageNet is available at [here](https://people.eecs.berkeley.edu/~longlian/files/usl_t_imagenet_0.2_selected_indices.zip). 

## Model Zoo

### USL
Note that USL does not require training, so the weights are just pretraining weights (e.g., with MoCo/SimCLR/CLD).

| Dataset   | Models for Sample Selection |
|-----------|-----------------------------|
| CIFAR-10  | [Download (CLD)](https://people.eecs.berkeley.edu/~longlian/files/cifar10_ckpt_epoch_200.pth)              |
| CIFAR-100 | [Download (CLD)](https://people.eecs.berkeley.edu/~longlian/files/cifar100_ckpt_epoch_200.pth)             |
| ImageNet  | [Download (MoCov2-EMAN)](https://eman-cvpr.s3.amazonaws.com/models/res50_moco_eman_800ep.pth.tar)            |

These weights can be directly parsed by `selective_labeling/usl-{cifar,imagenet}.py` to get selections. The selection should match with the selections released (besides slight variations due to indeterministic operations in GPUs).

#### FixMatch SSL Trained Model for USL
You can obtain the EMAN-FixMatch trained model for baselines/Stratified/USL-MoCo/USL-CLIP on ImageNet [here](https://people.eecs.berkeley.edu/~longlian/files/usl_imagenet.html).

### USL-T
| Dataset   | Models for Sample Selection | Results of this re-implementation | Results in the work |
|-----------|------------------------------------------------------------------------------------------------------------|-----------|-----------|
| CIFAR-10  | [Download](https://people.eecs.berkeley.edu/~longlian/files/cifar10_usl_t_3_heads.pth)                     | SimCLR-CLD: 76.1 | SimCLR-CLD: 76.1 |
| CIFAR-100 | [Download](https://people.eecs.berkeley.edu/~longlian/files/cifar100_usl_t_3_heads.pth)                    | SimCLR-CLD: 36.1 | SimCLR-CLD: 36.9 |
| ImageNet  | [Download](https://people.eecs.berkeley.edu/~longlian/files/imagenet_usl_t_3_heads.pth)                    | SimCLR: 42.0 | SimCLR: 41.3 |

These weights can be directly parsed by `selective_labeling/usl-t-{cifar,imagenet}.py` to get selections. The selection should mostly match with the selections released (besides slight variations due to indeterministic operations in GPUs).

## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.

```
@inproceedings{wang2022unsupervised,
  title={Unsupervised Selective Labeling for More Effective Semi-Supervised Learning},
  author={Wang, Xudong and Lian, Long and Yu, Stella X},
  booktitle={European Conference on Computer Vision},
  pages={427--445},
  year={2022},
  organization={Springer}
}
```

## How to get support from us?
This is a re-implementation of our original codebase. If you have any questions about the implementation in this repo, feel free to email us at `longlian at berkeley.edu`.

I'm also happy to provide additional checkpoints for baselines and selected labels.

## License
This project is licensed under the MIT license. See [LICENSE](LICENSE) for more details. The parts described below follow their original license.

## Acknowledgements
This project uses code fragments from many projects. See credits in comments for the code fragments and adhere to their own LICENSE. The code that is written by the authors of this project is licensed under the MIT license.

We thank the authors of the following projects that we referenced in our implementation:
1. [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) for the overall configuration framework. 
2. [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification) for augmentations, auxiliary dataloader wrappers, and many utility functions in USL-T.
3. [PAWS](https://github.com/facebookresearch/suncet) for the use of sharpening function.
4. [MoCov2](https://github.com/facebookresearch/moco) for augmentations.
5. [pykeops](https://www.kernel-operations.io/keops/_auto_tutorials/knn/plot_knn_torch.html) for kNN and k-Means.
6. [FixMatch-pytorch](https://github.com/kekmodel/FixMatch-pytorch) for EMA.
7. [SimCLRv2-pytorch](https://github.com/Separius/SimCLRv2-Pytorch) for extracting and using SimCLRv2 weights
8. Other functions listed with their own credits.
