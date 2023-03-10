#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 21:51:38 2023
@author: Ajit Johnson Nirmal
Run Gator Pipeline
"""

# libs
import inspect
import argparse


if __name__ == '__main__':
    from gatorPredict import gatorPredict
else:
    from .gatorPredict import gatorPredict
    from .generateGatorScore import generateGatorScore
    from .gatorObject import gatorObject
    from .gator import gator




# Function

def gatorPipeline (**kwargs):    

    
    ##########################################################################
    # STEP: 1 :- PREDICT
    ##########################################################################
    function1_args = inspect.signature(gatorPredict).parameters.keys()
    # Extract only the arguments that gatorPredict expects from the keyword arguments
    function1_kwargs = {k: kwargs[k] for k in kwargs if k in function1_args}
    # Call gatorPredict with the extracted arguments
    # run the predictions
    gatorPredict (**function1_kwargs)
    
    ##########################################################################
    # STEP: 2 :- generateGatorScore
    ##########################################################################
    function2_args = inspect.signature(generateGatorScore).parameters.keys()
    function2_kwargs = {k: kwargs[k] for k in kwargs if k in function2_args}
    generateGatorScore (**function2_kwargs)
    
    
    ##########################################################################
    # STEP: 3 :- Generate gatorObject
    ##########################################################################
    function3_args = inspect.signature(gatorObject).parameters.keys()
    function3_kwargs = {k: kwargs[k] for k in kwargs if k in function3_args}
    gatorObject (**function3_kwargs)
    
    
    ##########################################################################
    # STEP: 4 :- Run Gator Algorithm
    ##########################################################################
    function4_args = inspect.signature(gator).parameters.keys()
    function4_kwargs = {k: kwargs[k] for k in kwargs if k in function4_args}
    gator (**function4_kwargs)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run gatorPipeline function')
    args = parser.parse_args()

