# Parallel Branches-based Second-Order Transformer for Robust Group Re-identification with Layout-Guided Occlusion Mitigation

This is a pytorch implementation of 《Parallel Branches-based Second-Order Transformer for Robust Group Re-identification with Layout-Guided Occlusion Mitigation》(IEEE TIM 2024, under review). 


## Abstract

Group Re-identification (GReID) seeks to accurately associate group images with the same members across different cameras. However, existing methods mainly focus on the challenges of layout and membership variations while neglecting the occlusion problem caused by layout variations. To address this issue, we propose a novel Parallel Branches-based Second-Order Transformer (PB-SOT) framework, which consists of a Parallel Learning Branch (PLB) module and a Layout-guided Local Feature Sampling (LLFS) module. Specifically, the PLB module designs two group feature transformers with distinct weights to extract global and local features at the group level, respectively. The LLFS module leverages inter-member 2D layout to generate spatial density relationships and performs targeted sampling on local features with the highest density, mitigating the impact of occlusion with layout variations. Experimental evaluations conducted on three public available datasets, including RoadGroup, CSG, and SYSUGroup, demonstrate the effectiveness and superiority of our proposed method.

## Performance
## Performance
TABLE I: Comparison of the proposed method with the state-of-the-art approaches on the CSG and RoadGroup datasets
|        Method       | Publication | CSG   | CSG   |  CSG   |  CSG   | RoadGroup    | RoadGroup   | RoadGroup    | RoadGroup    |
|  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |
|                       |       | mAP   | rank1 | rank5 | rank10 | mAP   | rank1 | rank5 | rank10                         |
| CRRRO-BRO | BMVC 2009                    | -     | 10.40  | 25.80  | 37.50   | -     | 17.80  | 34.60 | 48.10  |
| Covariance | ICPR 2010                    | -     | 16.50  | 34.10  | 47.90   | -     | 38.00    | 61.00    | 73.10 |
| BSC+CM | ICIP 2016                    | -     | 24.60  | 38.50  | 55.10   | -     | 58.60  | 80.60  | 87.40 |
| PREF | ICCV 2017                    | -     | 19.20  | 36.40  | 55.10   | -     | 43.00    | 68.70  | 77.90  |
| LIMI | ACM MM 2018                  | -     | -     | -     | -      | -     | 72.30  | 90.60  | 94.10  |
| DOTGNN | ACM MM 2019                  | -     | -     | -     | -      | -     | 74.10  | 90.10  | 92.60  |
| GCGNN | IEEE TMM 2020                | -     | -     | -     | -      | -     | 81.70  | 94.30  | 96.50 |
| MCG | TCYB 2021                    | -     | 57.80  | 71.60  | 76.50   | -     | 80.20  | 93.80  | 96.30 |
| DotSCN | TCSVT 2021                   | -     |       | -     | -      | -     | 84.00    | 95.10  | 96.30 |
| BGFNet-S | ICPR 22                      | 87.90  | 89.20  | 95.10  | 96.50   | 92.70  | 90.10  | 96.30  | 97.50 |
| SOT | AAAI 2022                    | 90.70  | 91.70  | 96.50  | 97.60   | 91.30  | 86.40  | 96.30  | 98.80  |
| TSN | ICARCV 2022                  | 94.60  | 96.30  | 97.70  | 98.00   | -     | -     | -     | -    |
| 3DT | CVPR 2022                    | 92.10  | 92.90  | 97.30  | 98.10   | 94.30  | 91.40  | 97.50 | 98.80  |
| MACG | TPAMI 2023                   | -     | 63.20  | 75.40  | 79.70   | -     | 84.50  | 95.00    | 96.90  |
| CPM* | CVPR 2023                    | 51.45 | 88.57 | -     | -      | -     | -     | -     | -   |
| UMSOT | IJCV 2024                    | 92.60  | 93.60  | 97.30  | 98.30   | 91.70  | 88.90  | 95.10  | 98.80  |
| TSN+ | IEEE TIM 2024                    | 96.57 | 96.30  | 98.13 | 99.00   | 96.70  | 95.53 | 98.34 | 99.80  |
| SSRG | TPAMI 2024                   | -     | 90.80  | 96.20  | 97.40   | -     | -     | -     | -  |
| Ours | -                            | 95.12 | **96.35** | 98.20  | 98.61  | 96.97 | 95.06 | 98.77 |  99.80  | 


## Architecture

```
<TBD>
```

## Installation

- Install Pytorch 1.8.1 (Note that the results reported in the paper are obtained by running the code on this Pytorch version. As raised by the issue, using higher version of Pytorch may seem to have a performance decrease on optic cup segmentation.)
- Clone this repo

```
git clone https://github.com/CQRhinoZ/PBSOT
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
