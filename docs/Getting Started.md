---
hide:
  - toc        # Hide table of contents
  - navigation
---

# 🐊 Getting Started with Gator 
Kindly note that **Gator is not a plug-and-play solution**, rather it's a framework that requires significant upfront investment of time from potential users for training and validating deep learning models, which can then be utilized in a plug-and-play manner for processing large volumes of similar multiplexed imaging data.
  
**There are two ways to set it up based on how you would like to run the program**
- Using an interactive environment like Jupyter Notebooks
- Using Command Line Interface
  
Before we set up Gator, we highly recommend using a environment manager like Conda. Using an environment manager like Conda allows you to create and manage isolated environments with specific package versions and dependencies. 
  
**Download and Install the right [conda](https://docs.conda.io/en/latest/miniconda.html) based on the opertating system that you are using**

## Create a new conda environment

```
# use the terminal (mac/linux) and anaconda promt (windows) to run the following command
conda create --name gator -y python=3.9
conda activate gator
```

**Install `gatorpy` within the conda environment.**

```
pip install gatorpy
```

## Interactive Mode
Using IDE or Jupyter notebooks

```python
pip install notebook

# open the notebook and import Gator
import gatorpy as ga
# follow the tutorials for how to run specific function
```

## Command Line Interface
```
wget https://github.com/nirmalLab/gatorpy/archive/main.zip
unzip main.zip 
cd gatorpy-main/gatorpy 
# follow the tutorials for how to run specific function

```

## Docker Container
```
docker pull nirmallab/gatorpy:gatorpy
# follow the tutorials for how to run specific function
```