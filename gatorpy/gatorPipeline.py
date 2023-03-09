#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 21:51:38 2023
@author: Ajit Johnson Nirmal
Run Gator Pipeline
"""

# libs
import inspect



# Libs
import os, argparse
import pandas as pd
import pathlib

# tools libs
from skimage import io as skio
import tensorflow.compat.v1 as tf
import tifffile
import numpy as np
from skimage.transform import resize

# from other .py scripts
if __name__ == '__main__':
    from toolbox.imtools import im2double
    from UNet import *
else:
    from .toolbox.imtools import im2double
    from .UNet import UNet2D
    


if __name__ == '__main__':
    from gatorPredict import gatorPredict
else:
    from gatorPredict import gatorPredict




# Function

def gatorPipeline (**kwargs):    
    # start
    function1_args = inspect.signature(gatorPredict).parameters.keys()
    print(function1_args)

# execute the pipeline
gatorPipeline()


# =============================================================================
# def my_wrapper(*args, **kwargs):
#     # Do something before calling the wrapped functions
#     
#     # Get the list of arguments that my_function1 expects
#     function1_args = inspect.signature(cloneFolder).parameters.keys()
#     # Extract only the arguments that my_function1 expects from the keyword arguments
#     function1_kwargs = {k: kwargs[k] for k in kwargs if k in function1_args}
#     # Call my_function1 with the extracted arguments
#     result = my_function1(*args, **function1_kwargs)
#     # Do something with the result
#     
#     # Get the list of arguments that my_function2 expects
#     function2_args = inspect.signature(my_function2).parameters.keys()
#     # Extract only the arguments that my_function2 expects from the keyword arguments
#     function2_kwargs = {k: kwargs[k] for k in kwargs if k in function2_args}
#     # Call my_function2 with the extracted arguments
#     result = my_function2(*args, **function2_kwargs)
#     # Do something with the result
#     
#     # ...
# =============================================================================
