<div class="title">
<h1>Universal Beta Splatting</h1>
</div>

[![button](https://img.shields.io/badge/Project-Website-blue.svg?style=social&logo=Google-Chrome)](https://rongliu-leo.github.io/universal-beta-splatting/)
[![button](https://img.shields.io/badge/Paper-arXiv-red.svg?style=social&logo=arXiv)](https://arxiv.org/abs/2510.03312)

<div class="authors is-size-5">
<a href="https://rongliu-leo.github.io/">Rong Liu<sup>1,2</sup></a>
<a href="https://sites.google.com/site/gaozhongpai/home">Zhongpai Gao<sup>2</sup>*</a>
<a href="https://planche.me/">Benjamin Planche<sup>2</sup></a>
<a href="https://www.linkedin.com/in/meida-chen-938a265b/">Meida Chen<sup>1</sup></a>
<a href="https://nv-nguyen.github.io/">Van Nguyen Nguyen<sup>2</sup></a>
<a href="https://meng-zheng.com/">Meng Zheng<sup>2</sup></a>
<a href="https://anwesachoudhuri.github.io/">Anwesa Choudhuri<sup>2</sup></a>
<a href="https://scholar.google.com/citations?user=S2BT6ogAAAAJ&hl=en">Terrence Chen<sup>2</sup></a>
<a href="https://yuewang.xyz/">Yue Wang<sup>1</sup></a>
<a href="https://www.linkedin.com/in/andrewfeng-ict/">Andrew Feng<sup>1</sup></a>
<a href="https://wuziyan.com/">Ziyan Wu<sup>2</sup></a>
</div>

<div class="affiliations is-size-5">
*Corresponding author<br>
<a href="https://usc.edu/" style="color:#990000;"><sup>1</sup>University of Southern California</a><br>
<a href="https://usa.united-imaging.com/" style="color:#003366;"><sup>2</sup>United Imaging Intelligence</a>
</div>

<div class="section teaser">
    <img src="teaser.png" alt="Teaser" style="width: 100%; height: auto; display: block; margin: 0 auto;">
    <p>UBS achieves superior angular-spatial rendering of reflective and specular materials in static scenes (left) and maintains high-fidelity temporal-spatial reconstruction in dynamic volumetric scenes (right), avoiding the blurring artifacts of 3DGS and 4DGS.</p>
  </div>

## Abstract
*We introduce Universal Beta Splatting (UBS), a unified framework that generalizes 3D Gaussian Splatting to N-dimensional anisotropic Beta kernels for explicit radiance field rendering. Unlike fixed Gaussian primitives, Beta kernels enable controllable dependency modeling across spatial, angular, and temporal dimensions within a single representation. Our unified approach captures complex light transport effects, handles anisotropic view-dependent appearance, and models scene dynamics without requiring auxiliary networks or specific color encodings. UBS maintains backward compatibility by approximating to Gaussian Splatting as a special case, guaranteeing plug-in usability and lower performance bounds. The learned Beta parameters naturally decompose scene properties without explicit supervision: spatial (surface vs. texture), angular (diffuse vs. specular), and temporal (static vs. dynamic). Our CUDA-accelerated implementation achieves real-time rendering while consistently outperforming existing methods across static, view-dependent, and dynamic benchmarks, establishing Beta kernels as a scalable universal primitive for radiance field rendering.*


## Demo Video
See more in our [project website](https://rongliu-leo.github.io/universal-beta-splatting/).



https://github.com/user-attachments/assets/39c6a1b2-5afb-4dc7-b6bd-0cb84a17abfa

## Quickstart

This project is built on top of the [Original 3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [3DGS-MCMC](https://github.com/ubc-vision/3dgs-mcmc), [gsplat](https://github.com/nerfstudio-project/gsplat), and [Deformable Beta Splatting](https://rongliu-leo.github.io/beta-splatting/) code bases. The authors are grateful to the original authors for their open-source codebase contributions.

### Installation Steps

1. **Clone the Repository:**
   ```sh
   git clone --single-branch --branch main https://github.com/RongLiu-Leo/universal-beta-splatting.git
   cd universal-beta-splatting
   ```
1. **Set Up the Conda Environment:**
    ```sh
    conda create -y -n ubs python=3.8
    conda activate ubs
    ```
1. **Install [Pytorch](https://pytorch.org/get-started/locally/) (Based on Your CUDA Version)**
    ```sh
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```
1. **Install Dependencies and Submodules:**
    ```sh
    pip install -e . --no-build-isolation
    ```

### Train a Beta Model
```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```
<details>
<summary><span style="font-weight: bold;">Important Command Line Arguments for train.py</span></summary>

  #### --input_dim
  default 6 for static scenes (xyz + view direction); set to 7 for dynamic scenes (with time as the 7th dimension).
  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --cap_max
  Number of primitives that the final model produces.
  #### --model_path / -m 
  Path where the trained model should be stored.
  #### --resolution / -r
  Image resolution downsample factor.
  #### --white_background / -w
  Whether use white background.
  #### --eval
  Whether use evaluation mode.
  #### --data_loader
  Whether use iterative data loader.

</details>
<br>


### Visualize a Trained Beta Model
```shell
python view.py --ply <path to a trained Beta Model ply file>
```
<details>
<summary><span style="font-weight: bold;">Important Command Line Arguments for view.py</span></summary>

  #### --input_dim
  default 6 for static scenes (xyz + view direction); set to 7 for dynamic scenes (with time as the 7th dimension).
  #### --ply
  Path to a trained Beta Model.
  #### --port
  Port to connect to the viewer.

</details>
<br>

### Evaluate a Trained Beta Model
```shell
python eval.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to trained model directory> 
```
<details>
<summary><span style="font-weight: bold;">Important Command Line Arguments for eval.py</span></summary>

  #### --input_dim
  default 6 for static scenes (xyz + view direction); set to 7 for dynamic scenes (with time as the 7th dimension).
  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path to the trained model directory where the trained model should be stored (```output/<random>``` by default).
  #### --iteration
  Loading trained iteration for rendering. "Best" by default.

</details>
<br>

### Produce benchmark results
```shell
python benchmark.py -<dataset> <path to dataset>
#For example, python benchmark.py -m360 <path to Mip-NeRF 360 dataset>
```
<details>
<summary><span style="font-weight: bold;">Important Command Line Arguments for benchmark.py</span></summary>

  #### --output_path
  Path to output directory. "eval" by default.
  #### --mipnerf360 / -m360
  Path to Mip-NeRF360 dataset
  #### --dnerf 
  Path to D-NeRF dataset
  #### --nerfsynthetic / -ns
  Path to NeRF Synthetic dataset

</details>
<br>


### Processing your own Scenes

The COLMAP loaders expect the following dataset structure in the source path location:

```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

For rasterization, the camera models must be either a SIMPLE_PINHOLE or PINHOLE camera. We provide a converter script ```convert.py```, to extract undistorted images and SfM information from input images. Optionally, you can use ImageMagick to resize the undistorted images. This rescaling is similar to MipNeRF360, i.e., it creates images with 1/2, 1/4 and 1/8 the original resolution in corresponding folders. To use them, please first install a recent version of COLMAP (ideally CUDA-powered) and ImageMagick. Put the images you want to use in a directory ```<location>/input```.
```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```
 If you have COLMAP and ImageMagick on your system path, you can simply run 
```shell
python convert.py -s <location> [--resize] #If not resizing, ImageMagick is not needed
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for convert.py</span></summary>

  #### --no_gpu
  Flag to avoid using GPU in COLMAP.
  #### --skip_matching
  Flag to indicate that COLMAP info is available for images.
  #### --source_path / -s
  Location of the inputs.
  #### --camera 
  Which camera model to use for the early matching steps, ```OPENCV``` by default.
  #### --resize
  Flag for creating resized versions of input images.
  #### --colmap_executable
  Path to the COLMAP executable (```.bat``` on Windows).
  #### --magick_executable
  Path to the ImageMagick executable.
</details>
<br>

## Citation
If you find our code or paper helps, please consider giving us a star or citing:
```bibtex
@misc{liu2025universalbetasplatting,
    title={Universal Beta Splatting}, 
    author={Rong Liu and Zhongpai Gao and Benjamin Planche and Meida Chen and Van Nguyen Nguyen and Meng Zheng and Anwesa Choudhuri and Terrence Chen and Yue Wang and Andrew Feng and Ziyan Wu},
    year={2025},
    eprint={2510.03312},
    archivePrefix={arXiv},
    primaryClass={cs.GR},
    url={https://arxiv.org/abs/2510.03312}, 
}
```
