# AQuA: Annotation Quality Assessment 
## Identifying Labeling Errors

### List of supported datasets
* Text
  * IMDB Sentiment Analysis (name:`imdb`)
  * TweetEval (name:`tweeteval`)
* Time-Series
    * MIT-BIH Arrhythmia Detection Dataset (name: `mitbih`)
    * DuckDuckGeese: Differentiate between sounds of ducks and geese (name: `duckduckgeese`) 
    * EigenWorms: Differentiate species of worms based on their motion (name: `eigenworms`)
    * ElectricDevices: Track usage of electric devices (name: `electricdevices`)
    * Crop: Differentiate between different crops based on plant characteristics (name: `crop`)
    * InsectWingbeatSound: Differentiate between species of insects based on the sound of their wingbeats (name: `insectwingbeat`)
* Vision
    * CIFAR-10 (name: `cifar10`)
    * Noisy CXR : Classify pneumonia based on chest x-rays (name: `cxr`)
* Tabular
    * Mushrooms: Classify edibility of mushrooms (name: `mushrooms`)
    * Adult: Classify income of adults (name: `adult`)
    * Dry Bean: Classify species of beans (name: `dry_bean`)
    * COMPAS: Correctional Offender Management Profiling for Alternative Sanctions (name: `compas`)
    * Car Evaluation: Classify condition of cars (name: `car_evaluation`)

### List of Supported Models:
* Text:
    * XLNet (name: `xlnet-base-cased`)
    * RoBERTa (name: `roberta-base`)
* Vision:
    * ResNet-18 (name: `resnet18`)
    * ResNet-34 (name: `resnet34`)
    * MobileNet-v2 (name: `mobilenet_v2`) 
* Time-Series:
    * ResNet-1D (name: `resnet1d`)
    * LSTM-FCN (name: `fcn`)
* Tabular:
    * MLP (name: `mlp`)

### List of Supported Cleaning Methods
* AUM (name: `aum`)
* CINCER (name: `cincer`)
* SimiFeat (name: `simifeat`)
* Cleanlab (name: `cleanlab`)


All names indicated inside parentheses can be used to fill `aqua/configs/main_config.json`. `main_config.json` is the main entry point for setting up experiments. Each  dataset, model and cleaning method have their own config files under `aqua/configs/datasets`, `aqua/configs/models/base` and `aqua/configs/models/cleaning` respectively.

<a id="license"></a>
## License

MIT License

Copyright (c) 2022 Carnegie Mellon University, Auton Lab.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See [MIT LICENSE](https://github.com/mononitogoswami/labelerrors/blob/main/LICENSE) for details.

<img align="right" height ="120px" src="assets/cmu_logo.png">
<img align="right" height ="110px" src="assets/autonlab_logo.png"> 