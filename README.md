# DeblurDiNAT: A Generalizable Transformer for Perceptual Image Deblurring
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

<img src="Figure/sota.png" width = "800"  div align=center />

<img src="./Figure/architecture.png" width = "800" height = "500" div align=center />

## Update:
* **2024.03.19** Release the initial version of codes for our DeblurDiNAT. 
* **2024.06.21** Improve the PSNR/SSIM scores and release the second version of codes for our DeblurDiNAT. 
* **2024.06.24** The updated preprint paper is available. 
* **2024.07.12** The updated preprint paper is available. 
* **2025.08.02** Extension work [DiNAT-IR](https://arxiv.org/abs/2507.17892), paper is available on arxiv.
* **2025.08.02** Extension work [DiNAT-IR](https://github.com/HanzhouLiu/DiNAT-IR), code has been released.

## Visual Results
| Blurry | DeblurDiNAT-L | FFTformer | Uformer-B | Stripformer | Restormer |
| --- | --- | --- | --- | --- | --- |
| <img src="Figure/books/blur.png" width="110"> | <img src="Figure/books/nadeblurL.png" width="110"> | <img src="Figure/books/fftformer.png" width="110"> | <img src="Figure/books/uformerb.png" width="110"> | <img src="Figure/books/stripformer.png" width="110"> | <img src="Figure/books/restormer.png" width="110"> |
| <img src="Figure/starbucks/blur.png" width="110"> | <img src="Figure/starbucks/nadeblurL.png" width="110"> | <img src="Figure/starbucks/fftformer.png" width="110"> | <img src="Figure/starbucks/uformerb.png" width="110"> | <img src="Figure/starbucks/stripformer.png" width="110"> | <img src="Figure/starbucks/restormer.png" width="110"> |
| <img src="Figure/pinkads/blur.png" width="110"> | <img src="Figure/pinkads/nadeblurL.png" width="110"> | <img src="Figure/pinkads/fftformer.png" width="110"> | <img src="Figure/pinkads/uformerb.png" width="110"> | <img src="Figure/pinkads/stripformer.png" width="110"> | <img src="Figure/pinkads/restormer.png" width="110"> |
| <img src="Figure/realj/blur.png" width="110"> | <img src="Figure/realj/nadeblurL.png" width="110"> | <img src="Figure/realj/fftformer.png" width="110"> | <img src="Figure/realj/uformerb.png" width="110"> | <img src="Figure/realj/stripformer.png" width="110"> | <img src="Figure/realj/restormer.png" width="110"> |

## Quantitative Results
<img src="Figure/table.png" width = "800"  div align=center />

## Installation
The implementation is modified from "[DeblurGANv2](https://github.com/VITA-Group/DeblurGANv2)".
```
git clone https://github.com/HanzhouLiu/DeblurDiNAT.git
cd DeblurDiNAT
conda create -n DeblurDiNAT python=3.8
conda activate DeblurDiNAT
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install cmake lit timm opencv-python tqdm pyyaml joblib glog scikit-image tensorboardX albumentations einops
pip install -U albumentations[imgaug]
pip install albumentations==1.1.0
pip3 install natten==0.14.6+torch200cu118 -f https://shi-labs.com/natten/wheels
pip install "numpy<2"
pip install timm==0.9.2
```
The Older Releases of NATTEN package is required. 
Please follow the NATTEN installation instructions "[NATTEN Homepage](https://shi-labs.com/natten/)".
Make sure Python, PyTorch, and CUDA versions are compatible with NATTEN.
If you installed the latest version, you may meet the unexpected key issue when loading pre-trained weights.

## Training
Download "[GoPro](https://drive.google.com/drive/folders/1BdV2l7A5MRXLWszGonMxR88eV27geb_n?usp=sharing)" dataset into './datasets'
for example: './datasets/GoPro'. Note: we say the blur images is A and the sharp images is B, e.g., ./GOPRO/test/sharp <-> .GOPRO/test/testB. </br>

Download "[VGG19 Pretrained Weights](https://drive.google.com/file/d/1r2_clZ02-ai6xM7EOHW9APqY9IxkPYsS/view?usp=drive_link)" into './models',
which is used to calculate ContrastLoss.  </br>

**We train our DeblurDiNAT in two stages:** </br>
* We pre-train DeblurDiNAT for 4000 epochs with patch size 256x256 </br> 
* Run the following command 
```
python train_DeblurDiNAT_pretrained.py
```

* After 4000 epochs, we keep training DeblurDiNAT for 2000 epochs with patch size 512x512 </br>
* Run the following command 
```
python train_DeblurDiNAT_gopro.py
```

## Testing
For reproducing our results on GoPro and HIDE datasets, download "[DeblurDiNATL.pth](https://drive.google.com/file/d/1VT7dpP550b83YZ0LjfmGA5t0nEA32EEs/view?usp=sharing)"

**For testing on GoPro dataset** </br>
* Download "[GoPro](https://drive.google.com/file/d/1Fp0MuEwFlzT_NKAFjr3SpuQl3Sm0cFYA/view?usp=sharing)" full dataset or test set into './datasets' (For example: './datasets/GoPro/test') </br>
* Run the following command
```
python predict_GoPro_test_results.py --job_name DeblurDiNATL --weight_name DeblurDiNATL.pth --blur_path ./datasets/GOPRO/test/testA
```
**For testing on HIDE dataset** </br>
* Download "[HIDE](https://drive.google.com/drive/folders/1BdV2l7A5MRXLWszGonMxR88eV27geb_n?usp=sharing)" into './datasets' </br>
* Run the following command
```
python predict_HIDE_results.py --job_name DeblurDiNATL --weight_name DeblurDiNATL.pth --blur_path ./datasets/HIDE/test/blur
```
**For testing on RealBlur test sets** </br>
* Download "[RealBlur_J](https://drive.google.com/drive/folders/1BdV2l7A5MRXLWszGonMxR88eV27geb_n?usp=sharing)" and "[RealBlur_R](https://drive.google.com/drive/folders/1BdV2l7A5MRXLWszGonMxR88eV27geb_n?usp=sharing)" into './datasets' </br>
* Run the following command
```
python predict_RealBlur_J_test_results.py --job_name DeblurDiNATL --weight_name DeblurDiNATL.pth --blur_path ./datasets/RealBlur_J/test/blur
```
```
python predict_RealBlur_R_test_results.py --job_name DeblurDiNATL --weight_name DeblurDiNATL.pth --blur_path ./datasets/RealBlur_R/test/blur
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
