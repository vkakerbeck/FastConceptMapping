# FastConceptMapping
Code and data corresponding to work on fast concept mapping (FCM) based on sparse and meaningful representations learned self-supervised through curiosity and embodiment.

Find data in the [./data](./data) folder and the bigger trainings and test data sets here: http://dx.doi.org/10.17632/zdh4d5ws2z.2

The jupyter notebooks in the [./scripts](./scripts) folder are numbered. The two scripts indexed with 0 are for model training of the autoencoder and the classifier. To train the agent I use ml-agents ppo (https://github.com/vkakerbeck/ml-agents-dev). The trained models can also be found in the data folder as well as the corresponding hidden layer activations on the test set of images. These activations were produced with code from [1_GetNetworkActivations.ipynb](./scripts/1_GetNetworkActivations.ipynb). From this you can then calculate the performance of fast concept mapping (FCM) on the repersentations using [2_CalculateFCMStatistics.ipynb](./scripts/2_CalculateFCMStatistics.ipynb). To visualize the statistics and further insights into the encodings use [3_AnalyzeResults.ipynb](./scripts/3_AnalyzeResults.ipynb).

If you have any questions or requests feel free to contact me: vkakerbeck@uos.de
