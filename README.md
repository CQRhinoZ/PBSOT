# Parallel Branches-based Second-Order Transformer for Robust Group Re-identification with Layout-Guided Occlusion Mitigation

This is a pytorch implementation of 《Parallel Branches-based Second-Order Transformer for Robust Group Re-identification with Layout-Guided Occlusion Mitigation》(IEEE TIM 2024, under review). 


## Abstract

Group Re-identification (GReID) seeks to accurately associate group images with the same members across different cameras. However, existing methods mainly focus on the challenges of layout and membership variations while neglecting the occlusion problem caused by layout variations. To address this issue, we propose a novel Parallel Branches-based Second-Order Transformer (PB-SOT) framework, which consists of a Parallel Learning Branch (PLB) module and a Layout-guided Local Feature Sampling (LLFS) module. Specifically, the PLB module designs two group feature transformers with distinct weights to extract global and local features at the group level, respectively. The LLFS module leverages inter-member 2D layout to generate spatial density relationships and performs targeted sampling on local features with the highest density, mitigating the impact of occlusion with layout variations. Experimental evaluations conducted on three public available datasets, including RoadGroup, CSG, and SYSUGroup, demonstrate the effectiveness and superiority of our proposed method.

## Performance


## Architecture

```
<TBD>
```

## Installation

- Install Pytorch 1.8.1 (Note that the results reported in the paper are obtained by running the code on this Pytorch version. As raised by the issue, using higher version of Pytorch may seem to have a performance decrease on optic cup segmentation.)
- Clone this repo

```
git clone https://github.com/CQRhinoZ/GUGEN
```

## Project Structure



## Dependency

After installing the dependency:

    pip install -r requirements.txt

## Train



## Citation

```

```

Feel free to contact us:

Xu ZHANG, Ph.D, Professor

Chongqing University of Posts and Telecommunications

Email: zhangx@cqupt.edu.cn

Website: https://faculty.cqupt.edu.cn/zhangx/zh_CN/index.htm
