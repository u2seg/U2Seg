#!/bin/bash

# Create the conda environment with the specific Python version
# conda create -n stego python=3.6.9 -y

# Activate the environment
# source activate stego

# Install minimal conda packages from specific channels
# conda install -c pytorch \
#   cudatoolkit=11.0 \
#   nvidia-apex=0.1.0 -y

# Install pip packages
pip install torch==1.7.1 \
  torchvision==0.8.2 \
  torchaudio==0.7.2 \
  'matplotlib>=3.3,<3.4' \
  'psutil>=5.8,<5.9' \
  'tqdm>=4.59,<4.60' \
  'pandas>=1.1,<1.2' \
  'scipy>=1.5,<1.6' \
  'numpy>=1.10,<1.20' \
  tensorboard==2.4.0 \
  future==0.17.1 \
  kornia \
  hydra-core \
  wget \
  seaborn \
  easydict \
  torchpq \
  pydensecrf \
  setuptools==59.5.0 \
  pyDeprecate==0.3.1 \
  scikit-image \
  pytorch-lightning


# Additional pip installations using the find-links option
pip install --find-links https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html

# Print completion message
echo "Environment 'stego' created and packages installed successfully."