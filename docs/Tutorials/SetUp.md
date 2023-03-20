# 🐊 Setting up Gator 
## Kindly note that **Gator is not a plug-and-play solution**, rather it's a framework that requires significant upfront investment of time from potential users for training and validating deep learning models, which can then be utilized in a plug-and-play manner for processing large volumes of similar multiplexed imaging data.

**There are two ways to set it up based on how you would like to run the program**
- Using an interactive environment like Jupyter Notebooks
- Using Command Line Interface

Before we set up Gator, we highly recommend using a environment manager like Conda. Using an environment manager like Conda allows you to create and manage isolated environments with specific package versions and dependencies. 

**Download and Install the right [conda](https://docs.conda.io/en/latest/miniconda.html) based on the opertating system that you are using**

<hr>

## Let's create a new conda environment and install Gator

```
# use the terminal (mac/linux) and anaconda promt (windows) to run the following command
conda create --name gator -y python=3.9
```

**Install `gatorpy` within the conda environment.**

```
conda activate gator
pip install gatorpy
```

<hr>

## Download the Exemplar Dataset
To help you get used to the program we have provided some dummy data.   
**Download link to the [exemplar dataset provided here.](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QDZ6XO)**  
All of the following files are mandatory for running Gator, but `phenotype_workflow.csv` is optional and can be skipped if single cell phenotyping is not required.
```
gatorExampleData
├── image
│   └── exampleImage.tif
├── markers.csv
├── phenotype_workflow.csv
├── quantification
│   └── exampleSpatialTable.csv
└── segmentation
    └── exampleSegmentationMask.tif
```

<hr>

## Method 1: Set up Jupyter Notebook (If you would like to run Gator in an interactive setting)
Install jupyter notebook within the conda environment
```
conda activate gator
pip install notebook
```
After installation, open Jupyter Notebook by typing the following command in the terminal, ensuring that the Gator environment is activated and you are within the environment before executing the jupyter notebook command.
```
jupyter notebook
```
We will talk about how to run Gator in the next tutorial.

## Method 2: Set up Command Line Interface (If you like to run Gator in the CLI, HPC, etc)

Activate the conda environment that you created earlier
```
conda activate gator
```

Download the gator program from github
```
wget https://github.com/nirmalLab/gatorpy/archive/main.zip
unzip main.zip 
cd gatorpy-main/gatorpy 
```
We will talk about how to run Gator in the next tutorial.

