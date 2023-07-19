## Introduction

Here we provide the data preprocessing code for multiple skin lesion datasets (now **7**). We also provide the code to generated sythesized datasets with domain shifts (similar to [cifar-10-c](https://github.com/hendrycks/robustness)). This can be useful for real-world medical images domain shfits studies.

In addition, we provide code to train a neural network to classify skin lesion with dataset HAM, as we utilized in our [paper](https://arxiv.org/abs/2207.09957). 

## Installation

For Conda users, you can create a new Conda environment using

```
conda create -n skinclassifier python=3.9
```

after activating the environment with 
```
source activate skinclassifier
```
try to install all the dependencies with

```
pip install -r requirements.txt
```
also install the conda environment for the jupyter notebook kernel.

```
python -m ipykernel install --user --name=skinclassifier
```

## Prepare datasets

Please refer to the instructions in `/skinlesiondatasets` to prepare the datasets.


## Training Neural Networks

```console
Usage: python skin_train.py [options]...

Training a ResNet in our experiments: python skin_train.py --gpu 0 --loss_type CE --train_rule None --b 96 --epochs 2000 --cosine


  --dataset             Dataset for training. Default: HAM
  --loss_type           The type of training loss. Support: (CE, LDAM, Focal) Default: CE
  --train_rule          The rule of sampling training samples. Support: (None, Resample, Reweight, DRW) Default: None
  --exp_str             Experiments saving name indicator. Default: 0
  --workers             Number of data loading workers Default: 4
  --epochs              Number of training epoches. Default: 200
  --start-epoch         Starting training epoch, useful for resuming. Default: 0
  --batch-size          Number of batch size. Default: 32
  --learning-rate       Initial learning rate. Default: 0.1
  --weight-decay        Weight decay of training process. Default: 1e-4
  --print-freq          Printing the trainign status. Default: 10
  --resume              Resuming checkpoint path. Default: None
  --evaluate            Only evaluate the model. Default: False
  --pretrained          Use the pretrained model. Default: False
  --seed                Seed to control the reproduciblity. Default: None
  --gpu                 Gpu id. Default: 0
  --root_log            Log name. Default: log
  --root_model          Save name. Default: checkpoint
  --crop                Crop the image to 224. Default: False
  --cosine              Decrease the learning rate with cosine Default: False

```


## Test Neural Networks

To generate the predictions, please refer to `/HAMtest.ipynb`. The results will be saved in `/skinresults`.

We provided a pretrained model that trained on HAM training splits. 

<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>accuracy on HAM test</th>
    <th colspan="6">download</th>
  </tr>
  <tr>
    <td>ResNet</td>
    <td>62M</td>
    <td>86.0%</td>
    <td><a href="https://drive.google.com/file/d/1yc0aJ-6TfP3_ysvx_FUrfzFCxqx8JwoF/view?usp=drive_link">ckpt</a></td>
    <td><a href="https://drive.google.com/file/d/1sCtYfPS_mJeKGfBk7PiMuzfNMNJRXih_/view?usp=sharing">args</a></td>
    <td><a href="https://drive.google.com/file/d/17j2JhxZm5saORRiT78VYjQKj6nkTG9MN/view?usp=sharing">log_train</a></td>
    <td><a href="https://drive.google.com/file/d/1dxiqMMQiviT3vPPvEA05T6GNk7V5pI9o/view?usp=sharing">log_eval</a></td>
  </tr>
</table>


## Acknowledgement

We brought code from [CICL](https://github.com/YMarrakchi/CICL) and [LDAM](https://github.com/kaidic/LDAM-DRW).

## Citation
If you find this code useful, please consider citing our work:

```
@article{li2022estimating,
  title={Estimating Model Performance under Domain Shifts with Class-Specific Confidence Scores},
  author={Li, Zeju and Kamnitsas, Konstantinos and Islam, Mobarakol and Chen, Chen and Glocker, Ben},
  journal={arXiv preprint arXiv:2207.09957},
  year={2022}
}
```
