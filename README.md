# DeblurDiNAT
Pytorch Implementation of "[DeblurDiNAT: A Lightweight and Effective Transformer for Image Deblurring](https://arxiv.org/abs/...)" 

<img src="./Figure/architecture.png" width = "800" height = "343" div align=center />

## Installation
The implementation is modified from "[DeblurGANv2](https://github.com/VITA-Group/DeblurGANv2)"
```
git clone https://github.com/HanzhouLiu/DeblurDiNAT.git
cd DeblurDiNAT
conda create -n DeblurDiNAT python=3.8
source activate DeblurDiNAT
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install opencv-python tqdm pyyaml joblib glog scikit-image tensorboardX albumentations
pip install -U albumentations[imgaug]
pip install albumentations==1.1.0
```
The NATTEN package is required. 
Please follow the NATTEN installation instructions.
Make sure Python, PyTorch, and CUDA versions are compatible with NATTEN.
[NATTEN Homepage](https://shi-labs.com/natten/)

## Citation
```

```
