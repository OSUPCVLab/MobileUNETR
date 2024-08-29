# MobileUNETR
## A Lightweight End-To-End Hybrid Vision Transformer For Efficient Medical Image Segmentation

### Overview:
    Segmentation approaches broadly fall into 2 categories. 
        1. End to End CNN Based Segmentation Methods
        2. Transformer Based Encoder with a CNN Based Decoder. 
    Many Transformer based segmentation approaches rely primarily on CNN based decoders overlooking the benefits of the Transformer architecture within the decoder. We address the need for an efficient/ lightweight End to End Transformer based segmentation architecture by introducing MobileUNETR, which aims to overcome the performance constraints associated with both CNNs and Transformers while minimizing model size, presenting a promising stride towards efficient image segmentation. MobileUNETR has 3 main features. 1) MobileUNETR comprises of a lightweight hybrid CNN-Transformer encoder to help balance local and global contextual feature extraction in an efficient manner; 2) A novel hybrid decoder that simultaneously utilizes low-level and global features at different resolutions within the decoding stage for accurate mask generation; 3) surpassing large and complex architectures, MobileUNETR achieves superior performance with 3 million parameters and a computational complexity of 1.3 GFLOPs.

## :rocket: News
* Repository Construction in Progress ... 

### Architecture Overview
<p align="center">
  <div style="position: relative; display: inline-block;">
    <img src="./resources/muvit_architecture.png" alt="Wide Image" width="900" style="display: block;">
  </div>
</p>

### Parameter Distribution and Computational Complexity
<p align="center">
  <div style="position: relative; display: inline-block;">
    <img src="./resources/params.png" alt="Wide Image" width="600" style="display: block;">
  </div>
</p>

### ISIC 2016 Performance
<p align="center">
  <div style="position: relative; display: inline-block;">
    <img src="./resources/isic_2016.png" alt="Wide Image" width="600" style="display: block;">
  </div>
</p>

### ISIC 2017 Performance
<p align="center">
  <div style="position: relative; display: inline-block;">
    <img src="./resources/isic_2017.png" alt="Wide Image" width="600" style="display: block;">
  </div>
</p>

### ISIC 2018 Performance
<p align="center">
  <div style="position: relative; display: inline-block;">
    <img src="./resources/isic_2018.png" alt="Wide Image" width="600" style="display: block;">
  </div>
</p>

### ISIC PH2 Performance
<p align="center">
  <div style="position: relative; display: inline-block;">
    <img src="./resources/ph2.png" alt="Wide Image" width="600" style="display: block;">
  </div>
</p>

### Advanced Architectures and Training Methods
<p align="center">
  <div style="position: relative; display: inline-block;">
    <img src="./resources/adv_arch.png" alt="Wide Image" width="600" style="display: block;">
  </div>
</p>
