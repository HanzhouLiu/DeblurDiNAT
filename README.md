# DeblurDiNAT: A Lightweight and Effective Transformer for Image Deblurring
<a href="https://arxiv.org/abs/2403.13163"><img src="https://img.shields.io/badge/arXiv-2403.13163-orange" /></a> </br>

<a href='https://www.linkedin.com/in/hanzhouliu/'>Hanzhou Liu</a>, 
<a href='https://www.linkedin.com/in/binghanli/'>Binghan Li</a>, 
<a href='https://chengkai-liu.github.io/'>Chengkai Liu</a>,
<a href='https://cesg.tamu.edu/faculty/mi-lu/'>Mi Lu</a>


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deblurdinat-a-lightweight-and-effective/deblurring-on-realblur-j-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-j-trained-on-gopro?p=deblurdinat-a-lightweight-and-effective)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deblurdinat-a-lightweight-and-effective/deblurring-on-realblur-r-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-r-trained-on-gopro?p=deblurdinat-a-lightweight-and-effective)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deblurdinat-a-lightweight-and-effective/deblurring-on-hide-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-hide-trained-on-gopro?p=deblurdinat-a-lightweight-and-effective)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deblurdinat-a-lightweight-and-effective/deblurring-on-gopro)](https://paperswithcode.com/sota/deblurring-on-gopro?p=deblurdinat-a-lightweight-and-effective)

This is the Official Pytorch Implementation of DeblurDiNAT.

<img src="./Figure/architecture.png" width = "800" height = "343" div align=center />

## Visual Results
| Blurry | DeblurDiNAT-L | FFTformer | Uformer-B | Stripformer | Restormer |
| --- | --- | --- | --- | --- | --- |
| <img src="Figure/books/blur.png" width="100"> | <img src="Figure/books/nadeblurL.png" width="100"> | <img src="Figure/books/fftformer.png" width="100"> | <img src="Figure/books/uformerb.png" width="100"> | <img src="Figure/books/stripformer.png" width="200"> | <img src="Figure/books/restormer.png" width="100"> |


## Installation
The implementation is modified from "[DeblurGANv2](https://github.com/VITA-Group/DeblurGANv2)".
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
Please follow the NATTEN installation instructions "[NATTEN Homepage](https://shi-labs.com/natten/)".
Make sure Python, PyTorch, and CUDA versions are compatible with NATTEN.

## Training
Download "[GoPro](https://drive.google.com/drive/folders/1BdV2l7A5MRXLWszGonMxR88eV27geb_n?usp=sharing)" dataset into './datasets' </br>
For example: './datasets/GoPro'

**We train our DeblurDiNAT in two stages:** </br>
* We pre-train DeblurDiNAT for 3000 epochs with patch size 256x256 </br> 
* Run the following command 
```
python train_DeblurDiNAT_pretrained.py
```

* After 3000 epochs, we keep training DeblurDiNAT for 1000 epochs with patch size 512x512 </br>
* Run the following command 
```
python train_DeblurDiNAT_gopro.py
```

## Testing
For reproducing our results on GoPro and HIDE datasets, download "[DeblurDiNATL.pth](https://drive.google.com/file/d/1hkZxPMqhAZTP-DS0S6FxM1ZWYkMIEL9b/view?usp=sharing)"

**For testing on GoPro dataset** </br>
* Download "[GoPro](https://drive.google.com/drive/folders/1BdV2l7A5MRXLWszGonMxR88eV27geb_n?usp=sharing)" full dataset or test set into './datasets' (For example: './datasets/GoPro/test') </br>
* Run the following command
```
python predict_GoPro_test_results.py --job_name DeblurDiNATL --weight_name DeblurDiNATL.pth --blur_path ./datasets/GOPRO/test/blur
```
**For testing on HIDE dataset** </br>
* Download "[HIDE](https://drive.google.com/drive/folders/1BdV2l7A5MRXLWszGonMxR88eV27geb_n?usp=sharing)" into './datasets' </br>
* Run the following command
```
python predict_HIDE_results.py --weights_path ./DeblurDiNATL.pth 
```
**For testing on RealBlur test sets** </br>
* Download "[RealBlur_J](https://drive.google.com/drive/folders/1BdV2l7A5MRXLWszGonMxR88eV27geb_n?usp=sharing)" and "[RealBlur_R](https://drive.google.com/drive/folders/1BdV2l7A5MRXLWszGonMxR88eV27geb_n?usp=sharing)" into './datasets' </br>
* Run the following command
```
python predict_RealBlur_J_test_results.py --weights_path ./DeblurDiNATL.pth 
```
```
python predict_RealBlur_R_test_results.py --weights_path ./DeblurDiNATL.pth 
```

## Citation
```
@misc{liu2024deblurdinat,
      title={DeblurDiNAT: A Lightweight and Effective Transformer for Image Deblurring}, 
      author={Hanzhou Liu and Binghan Li and Chengkai Liu and Mi Lu},
      year={2024},
      eprint={2403.13163},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
