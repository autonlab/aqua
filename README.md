<h1 align="center">AQuA: A Benchmarking Tool for Label Quality Assessment</h1>
<h3 align="center">A benchmarking environment to rigorously evaluate methods that enable machine learning in the presence of label noise</h3>

Machine learning (ML) models are only as good as the data they are trained on. But recent studies have found datasets widely used to train and evaluate ML models, e.g. _ImageNet_, to have pervasive labeling errors. Erroneous labels on the train set hurt ML models' ability to generalize, and they impact evaluation and model selection using the test set. Consequently, learning in the presence of labeling errors is an active area of research, yet this field lacks a comprehensive benchmark to evaluate these methods. Most of these methods are evaluated on a few computer vision datasets with significant variance in the experimental protocols. With such a large pool of methods and inconsistent evaluation, it is also unclear how ML practitioners can choose the right models to assess label quality in their data. To this end, we propose a benchmarking environment `AQuA` to rigorously evaluate methods that enable machine learning in the presence of label noise. We also introduce a design space to delineate concrete design choices of label error detection models. We hope that our proposed design space and benchmark enable practitioners to choose the right tools to improve their label quality and that our benchmark enables objective and rigorous evaluation of machine learning tools facing mislabeled data.

<p align="center">
<img height ="300px" src="assets/overview.png">
</p>

Figure 1: _Overview of the `AQuA` benchmark framework_. `AQuA` comprises of datasets from **4** modalities, **4** single-label and **3** multi-annotator label noise injection methods, **4** state-of-the-art label error detection models, classification models, and several evaluation metrics beyond metrics of predictive accuracy. We are in the process of integrating several fairness, generalization, and robustness metrics into `AQuA`. The red and blue arrows show two example experimental pipelines for image data and time-series data, respectively. 

----

## Contents

1. [Design Space](#design_space)
2. [Datasets](#datasets)
3. [Classification Models](#models)
4. [Cleaning Methods](#cleaning_methods)
5. [Synthetic Noise](#syn_noise)
6. [Compatibility and Installation](#installation)
7. [Results](#results)
8. [Citation](#citation)
9. [License](#license)

<a id="design_space"></a>
### A Design Space of Label Error Detection Methods


<p align="center">
<img height ="300px" src="assets/design_space.png">
</p>

Figure 2: Design space of labeling error detection models to delineate concrete design choices. For more details, check out Section 3 in our [paper](https://arxiv.org/pdf/2306.09467.pdf). 

<a id="datasets"></a>
### List of supported datasets
* Text
    * IMDB Sentiment Analysis (name:`imdb`)
    * TweetEval (name:`tweeteval`)
* Time-Series
    * MIT-BIH Arrhythmia Detection Dataset (name: `mitbih`)
    * PenDigits: Differentiate between time-series pen tracings of different handwritten digits (name: `pendigits`)
    * ElectricDevices: Track usage of electric devices (name: `electricdevices`)
    * Crop: Differentiate between different crops based on plant characteristics (name: `crop`)
    * WhaleCalls: Classify audio signals as a right whale up-call or not (name: `whalecalls`)
* Vision
    * CIFAR-10 (name: `cifar10`)
    * Noisy CXR : Classify pneumonia based on chest x-rays (name: `cxr`)
    * Clothing100K : Classify clothing images (name: `clothing`)
* Tabular
    * Credit Card Fraud Detection: Classify credit card transaction features into fraud or not (name: `credit_fraud`)
    * Mushrooms: Classify edibility of mushrooms (name: `mushrooms`)
    * Adult: Classify income of adults (name: `adult`)
    * Dry Bean: Classify species of beans (name: `dry_bean`)
    * COMPAS: Correctional Offender Management Profiling for Alternative Sanctions (name: `compas`)
    * Car Evaluation: Classify condition of cars (name: `car_evaluation`)

<a id="models"></a>
### List of Supported Models:
* Text:
    * MiniLM-L6 (name: ` all-MiniLM-L6-v2`)
    * DistilRoBERTa (name: `all-distilroberta-v1`)
* Vision:
    * ResNet-18 (name: `resnet18`)
    * MobileNet-v2 (name: `mobilenet_v2`) 
* Time-Series:
    * ResNet-1D (name: `resnet1d`)
    * LSTM-FCN (name: `fcn`)
* Tabular:
    * MLP (name: `mlp`)

<a id="cleaning_methods"></a>
### List of Supported Cleaning Methods

* AUM (name: `aum`)
* CINCER (name: `cincer`)
* SimiFeat (name: `simifeat`)
* Cleanlab (name: `cleanlab`)

All names indicated inside parentheses can be used to fill `aqua/configs/main_config.json`. `main_config.json` is the main entry point for setting up experiments. Each  dataset, model and cleaning method have their own config files under `aqua/configs/datasets`, `aqua/configs/models/base` and `aqua/configs/models/cleaning` respectively.

<a id="syn_noise"></a>
### Synthetic Noise
* For single-label datasets:
    * Uniform Noise
    * Class-dependent Noise
    * Asymmetric Label Noise
    * Instance-dependent Label Noise
* For datasets with labels from multiple annotators:
    * Dissenting Label
    * Dissenting Worker
    * Crowd Majority

<a id="installation"></a>
### Compatibility and Installation

`aqua` requires `python` 3.7+ to install. 

For a full list of dependencies required to run `aqua`, please refer to `requirements.txt`. 

To install `aqua`, run the following:

```console
foo@bar:~$ git clone https://github.com/autonlab/aqua.git
foo@bar:~$ python setup.py install
```

<a id="results"></a>
### Results

|     Datasets     | No Noise Injected |       |      |      |       | Assymmetric |      |      |      |      | Class-dependent |       |      |      |       | Instance-dependent |      |      |      |      | Uniform |      |      |      |      |
|:----------------:|:-----------------:|:-----:|:----:|:----:|:-----:|:-----------:|:----:|:----:|:----:|:----:|:---------------:|:-----:|:----:|:----:|:-----:|:------------------:|:----:|:----:|:----:|:----:|:-------:|:----:|:----:|:----:|:----:|
|                  |        **NON**       | **AUM**  | **CIN** | **CON** | **SIM**  |        **NON**       | **AUM**  | **CIN** | **CON** | **SIM**  |        **NON**       | **AUM**  | **CIN** | **CON** | **SIM**  |        **NON**       | **AUM**  | **CIN** | **CON** | **SIM**  |        **NON**       | **AUM**  | **CIN** | **CON** | **SIM**  |
|       Crop       |        58.4       |  57.8 | 53.1 | 12.4 |  56.5 |     47.4    | 49.6 | 46.8 | 13.7 | 46.3 |       47.5      |  47.1 | 42.3 | 13.5 |  43.9 |        42.9        | 35.3 | 37.9 |  8.5 | 48.5 |   48.1  | 54.9 | 54.7 | 13.7 | 54.1 |
| Electric Devices |        63.2       |  67.2 | 67.3 | 39.9 |  65.3 |     56.4    | 55.0 | 54.3 | 37.3 | 56.2 |       34.2      |  31.0 | 34.6 | 24.9 |  33.7 |        50.7        | 46.8 | 50.9 | 28.9 | 53.2 |   57.9  | 57.9 | 60.7 | 35.3 | 55.1 |
|      MIT-BIH     |        76.0       |  64.8 | 86.9 | 72.5 |  76.4 |     75.3    | 78.2 | 72.1 | 52.3 | 71.6 |       82.4      |  80.5 | 83.2 | 76.2 |  82.2 |        67.0        | 71.9 | 75.1 | 66.3 | 77.4 |   84.5  | 77.1 | 87.3 | 74.3 | 81.8 |
|     PenDigits    |        96.2       |  96.7 | 95.4 | 58.3 |  95.8 |     81.8    | 84.4 | 80.7 | 34.5 | 83.9 |       49.8      |  55.2 | 54.8 | 19.6 |  53.0 |        84.4        | 77.4 | 81.7 | 22.4 | 85.2 |   93.8  | 96.3 | 95.4 | 39.6 | 94.8 |
|    WhaleCalls    |        85.6       |  34.7 | 59.6 | 62.4 |  62.9 |     52.5    | 63.7 | 52.2 | 48.9 | 54.3 |       39.1      |  41.1 | 40.3 | 41.9 |  41.9 |        51.4        | 58.9 | 68.5 | 47.8 | 50.8 |   53.5  | 42.8 | 60.5 | 44.5 | 62.2 |
|       Adult      |        84.4       |  84.4 | 84.2 | 77.7 |  84.3 |     83.6    | 83.2 | 83.2 | 76.9 | 83.1 |       82.3      |  82.5 | 83.4 | 83.5 |  82.6 |        82.4        | 82.6 | 83.1 | 69.3 | 82.2 |   83.5  | 83.5 | 83.4 | 68.0 | 83.7 |
|  Car Evaluation  |        92.4       |  89.9 | 79.7 | 57.6 |  89.3 |     82.9    | 81.4 | 64.5 | 59.0 | 75.8 |       90.3      |  86.6 | 83.3 | 57.6 |  85.0 |        77.3        | 75.9 | 68.7 | 58.4 | 73.7 |   81.1  | 76.3 | 73.1 | 57.6 | 74.3 |
|      COMPAS      |        66.9       |  66.8 | 65.9 | 65.5 |  66.4 |     65.2    | 65.6 | 65.6 | 33.8 | 66.0 |       38.2      |  66.6 | 64.8 | 36.7 |  67.1 |        64.4        | 65.3 | 58.3 | 49.4 | 64.5 |   65.0  | 58.0 | 64.5 | 62.4 | 65.1 |
|   Credit Fraud   |       100.0       | 100.0 | 99.9 | 99.9 | 100.0 |     99.9    | 99.9 | 99.9 | 99.8 | 99.9 |       99.9      |  99.9 | 99.9 | 99.8 | 100.0 |        99.9        | 99.7 | 99.8 | 99.9 | 99.7 |   99.9  | 99.9 | 99.9 | 75.0 | 99.9 |
|     Dry Bean     |        92.0       |  91.2 | 91.0 | 67.3 |  90.8 |     82.5    | 84.6 | 88.8 | 51.7 | 84.6 |       91.2      |  89.8 | 89.3 | 19.8 |  88.2 |        84.8        | 81.7 | 83.0 | 40.5 | 87.3 |   86.0  | 90.6 | 90.7 | 62.1 | 62.1 |
|     Mushrooms    |        99.5       | 100.0 | 99.3 | 99.7 |  99.8 |     98.1    | 98.3 | 98.3 | 81.6 | 98.9 |       99.3      | 100.0 | 98.6 | 98.4 |  99.8 |        96.2        | 96.8 | 96.4 | 75.7 | 95.9 |   98.9  | 98.0 | 98.2 | 89.3 | 98.5 |
|     CIFAR-10     |        80.7       |  80.5 | 80.3 | 38.3 |  79.9 |     53.5    | 65.1 | 64.1 | 28.5 | 65.3 |       77.7      |  78.6 | 78.5 | 42.0 |  71.9 |        57.2        | 62.8 | 65.4 | 24.9 | 63.8 |   66.0  | 64.3 | 69.2 | 25.1 | 66.3 |
|   Chest X-rays   |        64.4       |  65.2 | 65.0 | 15.0 |  63.9 |     51.4    | 50.3 | 54.2 |  8.1 | 50.2 |       63.3      |  62.9 | 65.4 |  7.9 |  63.5 |        48.4        | 48.4 | 52.4 | 10.6 | 48.9 |   52.7  | 52.4 | 59.5 |  9.6 | 51.5 |
|   Clothing-100K  |        91.0       |  90.7 | 90.7 | 90.9 |  90.7 |     80.9    | 77.4 | 72.7 | 74.7 | 77.6 |       85.1      |  80.4 | 80.3 | 90.6 |  87.1 |        74.6        | 61.1 | 70.7 | 77.3 | 74.6 |   77.2  | 74.1 | 77.2 | 84.6 | 76.0 |
|       IMDb       |        84.9       |  87.5 | 89.2 | 69.6 |  90.3 |     70.1    | 57.7 | 73.3 | 60.3 | 76.4 |       87.1      |  84.9 | 89.1 | 85.5 |  87.1 |        58.7        | 57.6 | 59.4 | 55.0 | 55.5 |   59.4  | 56.0 | 61.0 | 58.5 | 60.9 |
|     TweetEval    |        73.6       |  73.6 | 77.1 | 65.1 |  76.8 |     65.9    | 65.5 | 68.7 | 55.2 | 69.6 |       77.0      |  80.1 | 78.7 | 51.4 |  77.9 |        66.1        | 67.6 | 68.2 | 67.5 | 55.8 |   71.2  | 68.2 | 73.8 | 45.4 | 70.1 |

Table 1: Impact of label noise on weighted F1 score of a downstream model for each modality on the test set, averaged across noise rates and downstream models.


|     Datasets     | Asymmetric |       |       |       | Class-dependent |       |       |       | Instance-dependent |       |       |       | Uniform |       |       |       |
|------------------|:----------:|:-----:|:-----:|:-----:|:---------------:|:-----:|:-----:|:-----:|:------------------:|:-----:|:-----:|:-----:|:-------:|:-----:|:-----:|:-----:|
|                  |  **AUM**   |**CIN**|**CON**|**SIM**|      **AUM**    |**CIN**|**CON**|**SIM**|       **AUM**      |**CIN**|**CON**|**SIM**| **AUM** |**CIN**|**CON**|**SIM**|
|       Crop       |    65.5    | 65.6  | 21.3  | 70.3  |       40.8      |  61.1 | 32.9  | 60.7  |        59.4        | 57.7  | 25.1  | 68.9  |   65.8  | 76.4  | 21.7  | 77.3  |
| Electric Devices |    65.2    | 74.3  | 38.8  | 74.7  |       41.3      |  67.7 | 51.1  | 64.9  |        60.1        | 64.8  | 31.7  | 71.9  |   65.5  | 82.8  | 33.2  | 81.1  |
|      MIT-BIH     |    65.3    | 78.1  | 48.6  | 70.6  |       55.5      |  67.3 | 45.6  | 75.3  |        59.7        | 70.6  | 47.6  | 71.1  |   65.1  | 86.6  | 50.0  | 81.9  |
|     PenDigits    |    65.3    | 81.5  | 26.6  | 73.6  |       51.9      |  51.1 | 49.7  | 78.9  |        59.6        | 78.0  | 23.6  | 74.5  |   65.6  | 94.7  | 26.1  | 75.1  |
|    WhaleCalls    |    65.3    | 66.3  | 57.9  | 69.3  |       34.4      |  38.7 | 50.8  | 39.0  |        59.2        | 60.3  | 54.6  | 62.1  |   65.3  | 65.7  | 57.6  | 70.3  |
|       Adult      |    65.3    | 65.8  | 59.2  | 69.1  |       62.5      |  63.4 | 58.1  | 63.5  |        59.3        | 60.1  | 62.9  | 60.8  |   65.3  | 65.9  | 62.3  | 66.5  |
|  Car Evaluation  |    65.2    | 73.7  | 77.2  | 78.6  |       85.6      |  91.8 | 84.1  | 88.2  |        59.5        | 71.1  | 74.4  | 68.5  |   64.2  | 81.0  | 78.3  | 83.7  |
|      COMPAS      |    65.3    | 65.4  | 59.7  | 66.3  |       55.6      |  55.4 | 53.3  | 55.3  |        58.9        | 59.4  | 54.6  | 65.3  |   65.3  | 65.2  | 57.8  | 64.6  |
|   Credit Fraud   |    65.3    | 65.2  | 69.5  | 66.9  |       77.9      |  78.0 | 91.1  | 93.7  |        59.4        | 59.0  | 67.5  | 65.0  |   65.4  | 65.2  | 58.9  | 69.4  |
|     Dry Bean     |    65.3    | 80.1  | 37.2  | 73.0  |       86.6      |  94.5 | 34.6  | 90.3  |        59.5        | 76.9  | 32.4  | 70.0  |   65.1  | 88.3  | 40.1  | 74.0  |
|     Mushrooms    |    65.3    | 73.7  | 57.0  | 75.3  |       99.2      | 100.0 | 70.1  | 99.8  |        59.0        | 65.0  | 53.6  | 62.6  |   65.7  | 74.5  | 55.2  | 78.1  |
|     CIFAR-10     |    65.4    | 70.9  | 25.2  | 68.6  |       94.7      |  86.1 | 17.9  | 94.4  |        59.4        | 70.6  | 26.9  | 64.8  |   65.4  | 75.4  | 25.1  | 72.3  |
|   Chest X-rays   |    65.3    | 67.9  | 22.3  | 65.2  |       95.2      |  83.7 | 15.2  | 95.8  |        59.4        | 68.7  | 24.8  | 59.5  |   65.4  | 74.3  | 22.5  | 65.2  |
|   Clothing-100K  |    65.2    | 60.6  | 70.9  | 65.1  |       88.7      |  84.7 | 82.2  | 88.9  |        59.1        | 52.4  | 73.7  | 58.8  |   65.3  | 59.3  | 78.2  | 65.1  |
|       IMDb       |    65.3    | 65.8  | 56.8  | 69.7  |       90.7      |  89.4 | 62.1  | 94.5  |        59.6        | 62.1  | 53.8  | 64.6  |   65.2  | 65.3  | 55.1  | 69.4  |
|     TweetEval    |    65.4    | 65.6  | 55.2  | 68.8  |       71.6      |  72.3 | 55.0  | 73.4  |        59.4        | 59.2  | 56.8  | 64.8  |   64.9  | 65.0  | 51.9  | 69.2  |

Table 2: Performance of cleaning methods across different types of synthetic noise added to the train set in terms of weighted F1, averaged across noise rates and downstream models.

<a id="citation"></a>
## Citation

If you use AQuA in any scientific publication, please consider citing our work in addition to any model and data-specific references that are relevant for your work:
```bib
@article{goswami2023aqua,
title={AQuA: A Benchmarking Tool for Label Quality Assessment},
author={Goswami, Mononito and Sanil, Vedant and Choudhry, Arjun and Srinivasan, Arvind and Udompanyawit, Chalisa and Dubrawski, Artur},
journal={arXiv preprint arXiv:2306.09467},
year={2023}
```

<a id="contribution"></a>
## Contributions
We encourage researchers to contribute their methods and datasets to AQuA. We are actively working on contributing guidlines. Stay tuned for updates!

<a id="license"></a>
## License

MIT License

Copyright (c) 2023 Carnegie Mellon University, Auton Lab.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See [MIT LICENSE](https://github.com/mononitogoswami/labelerrors/blob/main/LICENSE) for details.

<img align="right" height ="120px" src="assets/cmu_logo.png">
<img align="right" height ="110px" src="assets/autonlab_logo.png"> 
