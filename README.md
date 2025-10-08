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
    <img src="static/images/teaser.png" alt="Teaser" style="width: 100%; height: auto; display: block; margin: 0 auto;">
    <p>UBS achieves superior angular-spatial rendering of reflective and specular materials in static scenes (left) and maintains high-fidelity temporal-spatial reconstruction in dynamic volumetric scenes (right), avoiding the blurring artifacts of 3DGS and 4DGS.</p>
  </div>

## Abstract
*We introduce Universal Beta Splatting (UBS), a unified framework that generalizes 3D Gaussian Splatting to N-dimensional anisotropic Beta kernels for explicit radiance field rendering. Unlike fixed Gaussian primitives, Beta kernels enable controllable dependency modeling across spatial, angular, and temporal dimensions within a single representation. Our unified approach captures complex light transport effects, handles anisotropic view-dependent appearance, and models scene dynamics without requiring auxiliary networks or specific color encodings. UBS maintains backward compatibility by approximating to Gaussian Splatting as a special case, guaranteeing plug-in usability and lower performance bounds. The learned Beta parameters naturally decompose scene properties without explicit supervision: spatial (surface vs. texture), angular (diffuse vs. specular), and temporal (static vs. dynamic). Our CUDA-accelerated implementation achieves real-time rendering while consistently outperforming existing methods across static, view-dependent, and dynamic benchmarks, establishing Beta kernels as a scalable universal primitive for radiance field rendering.*

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