# IANet: Instance-Aware Feature Learning for Efficient Few-Shot Hyperspectral Image Classification

The source code for our new work on few-shot hyperspectral image classification tasks. The details will be reported after the acceptance.

## Abstract
Few-shot hyperspectral image classification poses a significant challenge due to the high dimensionality of data and the scarcity of labeled samples. Existing deep learning-based methods often employ fixed-architecture network models, applying a monolithic computation path to all input samples regardless of their intrinsic complexity. This can lead to over-processing of simple samples and under-processing of complex ones, exacerbating overfitting in few-shot scenarios. To address this issue, we propose the Instance-wise Adaptive Network (IAN), a novel framework that performs dynamic, instance-aware feature learning with an adaptive computation path. The core of IAN features a synergy between a Global Path Selector (GPS) and a Hierarchical Feature Refiner (HFR). First, the HFR is constructed by cascading a series of Spatial-spectral Adaptive Granularity Blocks (SAGBs), each with distinct receptive fields to automatically capture spatial-spectral features at varying granularities. Then, a lightweight GPS assesses the complexity of the input sample to generate a binary gating vector. Guided by this vector, the HFR tailors a customized computation path by dynamically activating or bypassing the corresponding SAGBs, thereby adjusting its effective network depth on an instance-by-instance basis. This adaptive mechanism not only enhances the model expressiveness but also regularizes the network by preventing unnecessary computations, thus effectively mitigating overfitting. Extensive experiments on several benchmark datasets demonstrate that our proposed IAN significantly outperforms state-of-the-art methods, while also offering a more compact and lightweight architecture.
## Datasets

```
├── Patch9_TRIAN_META_DATA.pickle
└── test_datasets
    ├── PaviaU_data.mat
    ├── PaviaU_gt.mat
    ├── Indian_pines_corrected.mat
    ├── Indian_pines_gt.mat
    ├── Salinas_corrected.mat
    ├── Salinas_gt.mat
    ├── WHU_Hi_HanChuan_gt.mat
    └── WHU_Hi_HanChuan.mat

```
1) Run "trainMetaDataProcess.py" to generate the meta-training data "Patch9_TRIAN_META_DATA.pickle". And you can choose to download the meta-training data through Baidu Netdisk (link: https://pan.baidu.com/s/1i6SV57db3k4ErZs0UKyUeA?pwd=iykv)
2) Run "python IANet.py".
