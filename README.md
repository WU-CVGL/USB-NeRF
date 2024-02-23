# USB-NeRF: Unrolling Shutter Bundle Adjusted Neural Radiance Fields
<!-- ### [Project Page]() | [Video]() |  -->
### [Paper](https://arxiv.org/abs/2310.02687) | [Data](https://westlakeu-my.sharepoint.com/:f:/g/personal/cvgl_westlake_edu_cn/EtiKvq7Vm-lHhB2lkRPedGMBnn6J2J1IAElp9xjUp5Nkog)<br>
PyTorch implementation of rolling shutter effect correction with NeRF.<br><br>
[USB-NeRF: Unrolling Shutter Bundle Adjusted Neural Radiance Fields](https://arxiv.org/abs/2310.0268)  
 [Moyang Li](https://scholar.google.com/citations?user=Qvu8bNYAAAAJ&hl)\*<sup>1,2</sup>,
 [Peng Wang](https://wangpeng000.github.io/)\*<sup>1,3</sup>,
 [Lingzhe Zhao](https://scholar.google.com/citations?user=mN764NsAAAAJ&hl=en)<sup>1</sup>,
 [Bangyan Liao](https://scholar.google.com/citations?user=0z2qluIAAAAJ&hl)<sup>1,3</sup>,
 [Peidong Liu](https://ethliup.github.io/)†<sup>1,3</sup>,<br>
 <sup>1</sup>Westlake University, <sup>2</sup>ETH Zurich, <sup>3</sup>Zhejiang University  
\* denotes equal contribution  
† denotes corresponding author  
in ICLR 2024

USB-NeRF is able to correct rolling shutter distortions and recover accurate camera motion trajectory simultaneously under the framework of NeRF, by modeling the physical image formation process of a rolling shutter camera.

## Quickstart

### 1. Setup environment

```
git clone https://github.com/WU-CVGL/USB-NeRF
cd USB-NeRF
pip install -r requirements.txt
```

### 2. Download datasets

You can download the data [here](https://westlakeu-my.sharepoint.com/:f:/g/personal/cvgl_westlake_edu_cn/EtiKvq7Vm-lHhB2lkRPedGMBnn6J2J1IAElp9xjUp5Nkog).

After acquiring the data, your folder structure should look like
```
Dataset/
    Unreal-RS/
        Adornment/
            images/
            start/
            mid/
            groundtruth.txt
            poses_bounds.npy
        BlueRoom/
            ...
        LivingRoom/
            ...
        WhiteRoom/
            ...
        intrinsics.txt
    ...
```
`images` folder contains captured rolling shutter images. `start` and `mid` folder contain global shutter images corresponding to the first and middle scanline, respectively. `groundtruth.txt` file saves the groundtruth poses, while `poses_bounds.npy` file saves the estimated camera poses with rolling shutter images via COLMAP. `intrinsics.txt` saves camera intrinsics (fx, fy, cx, cy).

### 3. Configs

Modify parameters of config file (eg: `configs/Unreal-RS/Adornment_CubicSpline.txt`) if needed.


### 4. Training

```
python train_usb_nerf.py --config ./configs/Unreal-RS/Adornment_CubicSpline.txt
```

After training, you can get global shutter images, optimized camera poses and synthesized novel view images.

## Citation

If you find this useful, please consider citing our paper:

```bibtex
@article{li2023usb,
  title={USB-NeRF: Unrolling Shutter Bundle Adjusted Neural Radiance Fields},
  author={Li, Moyang and Wang, Peng and Zhao, Lingzhe and Liao, Bangyan and Liu, Peidong},
  journal={arXiv preprint arXiv:2310.02687},
  year={2023}
}
```

## Acknowledgment

The overall framework, metrics computing and camera transformation are derived from [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch/), [CVR](https://github.com/GitCVfb/CVR) and [BAD-NeRF](https://github.com/WU-CVGL/BAD-NeRF) respectively. We appreciate the effort of the contributors to these repositories.