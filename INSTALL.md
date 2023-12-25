## Installation

### Requirements
- Linux or macOS with Python ≥ 3.7
- PyTorch ≥ 1.8 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional but needed by demo and visualization


### Build detectron2 
Step 1: Install Pytroch: following the instruction in https://pytorch.org/ to install the latest version of pytorch.

Step 2: Following the corresponding structure, clone the code, and run:
```
cd ./U2Seg

pip install -e .
```

### Install other dependencies
In order to make the model compatible to your syste, you may need adjust the version of some pachages:

```
pip install pillow==9.5
pip install opencv-python
pip install git+https://github.com/cocodataset/panopticapi.git
pip install scikit-image
```
