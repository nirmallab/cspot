

[![Downloads](https://static.pepy.tech/badge/cspot)](https://pepy.tech/project/cspot)
[![docs](https://github.com/nirmallab/cspot/actions/workflows/docs.yml/badge.svg)](https://github.com/nirmallab/cspot/actions/workflows/docs.yml)


# 🐊 Getting Started with CSPOT 
Kindly note that **CSPOT is not a plug-and-play solution**. It's a framework that requires significant upfront investment of time from potential users for training and validating deep learning models, which can then be utilized in a plug-and-play manner for processing large volumes of similar multiplexed imaging data.
  
**There are two ways to set it up based on how you would like to run the program**  
- Using an interactive environment like Jupyter Notebooks  
- Using Command Line Interface  
  
Before we set up CSPOT, we highly recommend using a environment manager like Conda. Using an environment manager like Conda allows you to create and manage isolated environments with specific package versions and dependencies. 
  
**Download and Install the right [conda](https://docs.conda.io/en/latest/miniconda.html) based on the opertating system that you are using**

## **Create a new conda environment**

```
# use the terminal (mac/linux) and anaconda promt (windows) to run the following command
conda create --name cspot -y python=3.9
conda activate cspot
```

**Install `cspot` within the conda environment.**

```
pip install cspot
```

## **Interactive Mode**
Using IDE or Jupyter notebooks

```python
pip install notebook

# open the notebook and import CSPOT
import cspot as cs
# Go to the tutorial section to follow along
```

## **Command Line Interface**
```
wget https://github.com/nirmalLab/cspot/archive/main.zip
unzip main.zip 
cd cspot-main/cspot 
# Go to the tutorial section to follow along

```

## **Docker Container**
```
docker pull nirmallab/cspot:cspot
# Go to the tutorial section to follow along
```