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