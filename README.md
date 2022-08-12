# XDNNBioimaging

Tensorflow implementation for the pre-print [What do Deep Neural Networks Learn in Medical Images?](https://arxiv.org/abs/2208.00953)

## Description

This repo is the implementation accompanying the above mentioned paper. The directory structure is as follows:
-Data contains a subdirectory (brainTumorDataPublic) for the dataset where the .mat files are found as well as the README.txt and cvind.mat files. Description of the dataset is found in the README.txt.
-Figures contains all plots from running the code. They are in png and svg formats.
-Models contains the trained and evaluated models, each in its subdirectory

## Getting Started
To get start, close this repo using the *git clone https://github.com/yusufbrima/XDNNBioimaging.git* command.

### Dependencies
* For now, all dependencies are in the requirements.txt file and can be installed with a *pip install requirements.txt* command. However, the code was written in Python 3.9.12, which shouldn't be much of a hassle installing.
* ex. Windows 10

### Installing
* From your terminal, cd into the repo
* Run "pip install requirements.txt" to install all dependencies
* That's it, :heart_eyes:

### Executing program
* From the terminal execute the following command passing the desired switch: "Download" -> To download the dataset, "Process" -> To Preprocess the dataset, "Train" -> To train the models and save them to the Models directory, "Saliency" -> To perform saliency analysis using the best performing models
* For examples *python cli.py --name Download* to download the dataset, typing lowercase download also works fine :heart_eyes:
* Do the same for the other actions e.g., *python cli.py --name train* to train the models
* By the way, all evaluation results are written as a .csv file in the ./Data directory for further analysis
* That's all there is, sit back and analyse your results from the plots.


## Authors

Contributors names and contact info

[Yusuf Brima ](https://yusufbrima.github.io/)

## Version History


* 0.1
    * Initial Release

## License

This project is licensed under the [MIT License] License - see the LICENSE.md file for details

## Acknowledgments
* Special thanks to the [Pair Team](https://pair-code.github.io/saliency/) for their well written saliency codebase
