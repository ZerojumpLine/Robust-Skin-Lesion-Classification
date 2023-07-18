## Introduction

Data processing for datasets from different sources.

Here is the details of multi-domain skin leision datasets.

<br/> <div align=center><img src="../figs/datatable.png" width="500px"/></div>


## Data downloaded

HAM, BCN, VIE, MSK and UDA can be downloaded from [the link](https://challenge.isic-archive.com/data/). We utilized the [tool](https://github.com/GalAvineri/ISIC-Archive-Downloader) to download the whole dataset. We saved the raw data in the folder `/ISIC-Archive-Downloader`.

D7P can be downloaded from [the link](https://challenge.isic-archive.com/data/). We saved the raw data in the folder `/D7Pdownload`.

PH2 can be downloaded from [the link](https://challenge.isic-archive.com/data/). We saved the raw data in the folder `/PH2download`.

The raw data should be like:

```
ISIC-Archive-Downloader/
├── Data/
  ├── Descriptions/
    ├── ISIC_0000000
    ├── ...
    └── ISIC_9999806
  ├── Images/
    ├── ISIC_0000000.jpeg
    ├── ...
    └── ISIC_9999806.jpeg
├── download_archive.py
└── ...
D7Pdownload/
├── release_v0/
  ├── images
  └── meta
PH2download/
├── PH2Dataset/
  ├── PH2 Dataset images
  ├── PH2_dataset.txt
  └── PH2_dataset.xlsx
```

## Data preprocessing

If you want to reproduce all the results of this paper, the reader can run the processing code sequentially. Otherwise, choose what you want:

`SplitSixDatasetsFromISIC.ipynb`, which picks the datasets from ISIC arichive.

`SplitHAMtrainval.ipynb`, which would split the dataset of HAM (which we take as the source dataset).

`MakeD7Pdataset.ipynb`, which would split the dataset of HAM (which we take as the source dataset).

`MakePH2dataset.ipynb`, which would split the dataset of HAM (which we take as the source dataset).

`Generate_HAMtestc.ipynb`, which would generated sythesized test data with domain shifts (adpated from [cifar-10-c](https://github.com/hendrycks/robustness)).

After running the notebook, the processed datasets would be saved in `/SkinLesionDatasets` and `/SkinLesionDatasets_C`.

```
SkinLesionDatasets/
├── HAMtrain/
├── HAMtest/
├── VIE/
└── ...
SkinLesionDatasets_C/
├── brightness/
├── contrast/
└── ...
```