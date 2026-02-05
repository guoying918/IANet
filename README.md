# IANet: Instance-Aware Feature Learning for Efficient Few-Shot Hyperspectral Image Classification

The source code for our new work on few-shot hyperspectral image classification tasks. The details will be reported after the acceptance.

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
