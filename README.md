# FastConceptMapping
Code and data corresponding to work on fast concept mapping (FCM) based on sparse and meaningful representations learned self-supervised through curiosity and embodiment.

### Pre-Print: 
https://arxiv.org/abs/2102.02153

### Data:
Find data in the [./data](./data) folder and the bigger trainings and test data sets here: http://dx.doi.org/10.17632/zdh4d5ws2z.2

### Project Structure:
The jupyter notebooks in the [./scripts](./scripts) folder are numbered. The two scripts indexed with 0 are for model training of the autoencoder and the classifier. To train the agent I use ml-agents ppo (https://github.com/vkakerbeck/ml-agents-dev). The trained models can also be found in the data folder as well as the corresponding hidden layer activations on the test set of images. These activations were produced with code from [1_GetNetworkActivations.ipynb](./scripts/1_GetNetworkActivations.ipynb). From this you can then calculate the performance of fast concept mapping (FCM) on the repersentations using [2_CalculateFCMStatistics.ipynb](./scripts/2_CalculateFCMStatistics.ipynb). To visualize the statistics and further insights into the encodings use [3_AnalyzeResults.ipynb](./scripts/3_AnalyzeResults.ipynb).

### Setup

The project still used an old version of tensorflow (originally written in tf 1.8 but lets try 1.13). If you want to run the scripts without updating them to tf>2.0 it is best to setup an environment with python 3.7.

`conda create --name FCMPaper python==3.7`

`python3.7 -m pip install tensorflow==1.13.1` (maybe need 1.8.0)

`python3.7 -m pip install keras==2.1.6`

`python3.7 -m pip install pillow`

`python3.7 -m pip install 'h5py==2.10.0' --force-reinstall` to avoid keras model load error

BYOL is written on much more recent dependencies. Because of lack of time this is just a quick fix here to compare BYOL which relies heavily on the original implementation. Therefore you need to use a different environment to run 0_BYOL-Extract.ipynb

If you have any questions or requests feel free to contact me: vkakerbeck@uos.de
