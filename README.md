# A Deep Generative Distance-Based Classifier for Out-of-Domain Detection with Mahalanobis Space 

This repository is the official implementation of [A Deep Generative Distance-Based Classifier for Out-of-Domain Detection with Mahalanobis Space](https://www.aclweb.org/anthology/2020.coling-main.125/) (**GOLING2020**) by [Hong Xu](https://www.aclweb.org/anthology/people/h/hong-xu/), [Keqing He](https://www.aclweb.org/anthology/people/k/keqing-he/), [Yuanmeng Yan](https://www.aclweb.org/anthology/people/y/yuanmeng-yan/), [Sihong Liu](https://www.aclweb.org/anthology/people/s/sihong-liu/), [Zijun Liu](https://www.aclweb.org/anthology/people/z/zijun-liu/), [Weiran Xu](https://www.aclweb.org/anthology/people/w/weiran-xu/). 

## Introduction
A deep generative distance-based model with Mahalanobis distance  to detect OOD samples. 

The architecture of the proposed model:

![](https://github.com/pris-nlp/Generative_distance-based_OOD/blob/main/img/model.jpg)

## Dependencies

We use anaconda to create python environment:

```
conda create --name python=3.6
```
Install all required libraries:
```
pip install -r requirements.txt
```
## How to run
#### 1. Train (only):
  ```
  python3 experiment.py --dataset SNIPS --proportion 50 --mode train
  ```
  &ensp;&ensp; When only training model, there is no need to provide `setting` parameters.

#### 2. Predict (only):
- **LOF**
```
python3 experiment.py --dataset <dataset> --proportion <proportion> --mode test --setting lof --model_dir <model_dir>
```
- **GDA**
```
python3 experiment.py --dataset <dataset> --proportion <proportion> --mode test --setting gda_lsqr_800 --model_dir <model_dir>
```
- **MSP**
```
python3 experiment.py --dataset <dataset> --proportion <proportion> --mode test --setting msp --model_dir <model_dir>
```

&ensp;&ensp;When only predicting the model(no training), the parameter `model_dir` is required to represent the folder where the model resides (which contains `model.h5` model files).

#### 3. Train model,  and use the trained model to predict:
```
python3 experiment.py --dataset <dataset> --proportion <proportion> --mode both --setting <setting>
```
&ensp;&ensp; `Setting` parameter is required to specify using which algorithm to predict, but `model_dir` parameter is not required.

#### 4. Specify the visible category:
```
python3 experiment.py --dataset <dataset> --proportion <proportion> --mode both --setting <setting> --seen_classes SearchCreativeWork RateBook
```
```
python3 experiment.py --dataset SNIPS --proportion 50 --mode test --setting msp_0.5 msp_0.6 msp_0.7 msp_0.8 msp_0.9 --model_dir ./outputs/SNIPS-50-06112350 --seen_classes AddToPlaylist BookRestaurant PlayMusic RateBook
```
## Parameters
**The parameters that must be specified:**
- `dataset`, required, The dataset to use, `ATIS` or `SNIPS` or `CLINC`.
- `proportion`, required, The proportion of seen classes, range from `0` to `100`.
- `seen_classes`, optional, The random seed to randomly choose seen classes.(e.g.`--seen_classes SearchCreativeWork RateBook`)
- `mode`, optional,Specify running mode, only`train`,only`test` or `both`
- `setting`, required, The settings to detect ood samples, e.g.
    - `lof`：using LOF for predicting.
    - `gda_lsqr_800`：using GDA for predicting, using `lsqr` for `solver`, and the threshold is 800 (Mahalanobis distance).
    - `msp`: using MSP for predicting.
- `model_dir`, The directory contains model file (.h5), requried when test only.

**The parameters that have default values (In general, it can stay fixed):**
- `gpu_device`, default=1
- `output_dir`, default="./outputs"
- `embedding_file`,default="glove.6B.300d.txt"
- `embedding_dim`, default=300
- `max_seq_len`, default=None
- `max_num_words`, default=10000
- `max_epoches`, default=200
- `patience`, default=20
- `batch_size`, default=256
## Results

1. Macro f1-score of unknown intents with different proportions (25%, 50% and 75%) of classes are treated as known intents on SNIPS and ATIS datasets.
<table>
    <tr  align="center">
    <td></td>
        <td colspan="3"><b>Snips</b></td>
        <td colspan="3"><b>ATIS</b></td>
        <td colspan="3"><b>CLINC-Full</b></td>
        <td colspan="3"><b>CLINC-Imbal</b></td>
    </tr>
    <tr>
         <td rawspan="2"> %of known intents</td>
        <td>25%</td>
        <td>50%</td>
        <td>75%</td>
        <td>25%</td>
        <td>50%</td>
        <td>75%</td>
        <td>25%</td>
        <td>50%</td>
        <td>75%</td>
        <td>25%</td>
        <td>50%</td>
        <td>75%</td>
    <tr>
				 <td>MSP</td>
				  <td>0.0</td>
			<td>6.2 </td>
			<td> 8.3</td>
			<td> 8.1</td>
			<td> 15.3</td>
			<td>17.2</td>
			<td>0.0</td>
			<td>21.3</td>
			<td>40.4</td>
			<td>0.0</td>
			<td>27.8</td>
			<td>40.4</td>
    </tr>
    <tr>
				<td>DOC </td>
				<td> 72.5</td>
				<td>67.9</td>
				<td>63.9</td>
				<td>61.6 </td>
				<td>62.8 </td>
				<td>37.7 </td>
				<td>-</td>
				<td> -</td>
				<td> -</td>
				<td>- </td>
				<td>-</td>
				<td> -</td>
    </tr>
<tr>
				<td>DOC(softmax) </td>
				<td>72.8</td>
				<td>65.7</td>
				<td>61.8</td>
				<td>63.6 </td>
				<td>63.3 </td>
				<td>38.7 </td>
				<td>-</td>
				<td> -</td>
				<td> -</td>
				<td>- </td>
				<td>-</td>
				<td> -</td>
</tr>
       <tr>
<td> LOF(softmax) </td>
<td> 76.0</td>
<td> 69.4 </td>
<td>65.8</td>
<td> 67.3 </td>
<td>61.8</td>
<td> 38.9</td>
<td> 91.1 </td>
<td>83.1</td>
<td> 63.5</td>
<td> 88.4</td>
<td> 77.6</td>
<td> 57.5</td>
   </tr>
<tr>
	<td> LOF(LMCL) </td>
	<td> 79.2</td>
	<td> 84.1 </td>
	<td>78.8</td>
	<td> 68.6 </td>
	<td>63.4</td>
	<td> 39.6</td>
	<td> 91.3 </td>
	<td>83.3</td>
	<td> 62.8</td>
	<td> 88.7</td>
	<td> 78.9</td>
	<td> 56.7</td>
   </tr>
<td>GDA+Euclidean distance </td>
<td>85.6</td>
<td> 85.6 </td>
<td>82.9</td>
<td> 77.9</td>
<td> <b>75.4*</b></td>
<td> <b>43.7*</b></td>
<td> 91.1 </td>
<td>84.2 </td>
<td>64.5</td>
<td> 91.1 </td>
<td>81.2</td>
<td> 60.8 </td>
   </tr>
   <tr>
<td>GDA+Mahalanobis distance</td>
<td>  <b>89.2*</b></td>
<td> <b>87.4*</b></td>
<td> <b>83.2</b></td>
<td><b>78.5* </b></td>
<td>72.8</td>
<td> 42.1 </td>
<td><b>91.4</b> </td>
<td><b>84.4</b> </td>
<td><b>65.1*</b></td>
<td><b>91.5</b> </td>
<td><b>81.5</b></td>
<td> <b>61.3*</b></td>
   </tr>
</table>


2. Comparison between our unsupervised OOD detection method and supervised N+1 classification.
<table>
<tr >
	<td>% of known intents </td>
	<td colspan="3"  align="center">50</td>
	<td colspan="3"  align="center"> 75</td>
</tr>
<tr >
<td>Macro f1-score </td>
<td>overall </td>
<td>seen</td>
<td> unseen </td>
<td>overall </td>
<td>seen </td>
<td>unseen </td>
</tr>
<tr>
<td>GDA+Mahalanobis distance </td>
<td>80.2 </td>
<td>80.1</td>
<td> 84 </td>
<td>79.4 </td>
<td>79.6</td>
<td> 65.7 </td>
</tr>
<tr>
<td>N+1 classification(2000) </td>
<td>64.6 </td>
<td>64.6 </td>
<td>67.7</td>
<td> 65.7 </td>
<td>65.7 </td>
<td>66.6</td>
</tr>
<tr>
<td> N+1 classification(4000) </td>
<td>45.3</td>
<td> 44.9 </td>
<td>77.7 </td>
<td>46.3 </td>
<td>46.1 </td>
<td>78.9</td>
</tr>
</table>

## Citation
```
@inproceedings{xu2020deep,
  title={A Deep Generative Distance-Based Classifier for Out-of-Domain Detection with Mahalanobis Space},
  author={Xu, Hong and He, Keqing and Yan, Yuanmeng and Liu, Sihong and Liu, Zijun and Xu, Weiran},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={1452--1460},
  year={2020}
}
```
