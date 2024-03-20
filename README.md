# DeblurDiNAT
Pytorch Implementation of "[DeblurDiNAT: A Lightweight and Effective Transformer for Image Deblurring](https://arxiv.org/abs/...)" 

<img src="./Figure/architecture.png" width = "800" height = "160" div align=center />

## Installation
The implementation is modified from "[DeblurGANv2](https://github.com/VITA-Group/DeblurGANv2)"
```
git clone https://github.com/HanzhouLiu/DeblurDiNAT.git
cd DeblurDiNAT
conda create -n DeblurDiNAT python=3.6
source activate DeblurDiNAT
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install opencv-python tqdm pyyaml joblib glog scikit-image tensorboardX albumentations
pip install -U albumentations[imgaug]
pip install albumentations==1.1.0
```

## Citation
```

```
