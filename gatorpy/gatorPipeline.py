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
import pathlib
import os

# Gator Functions
if __name__ == '__main__':
    from gatorPredict import gatorPredict
    from generateGatorScore import generateGatorScore
    from gatorObject import gatorObject
    from gator import gator
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
    gatorPredict (**function1_kwargs)
    
    
    ##########################################################################
    # STEP: 2 :- generateGatorScore
    ##########################################################################
    
    # derive the probability mask path
    probPath = pathlib.Path(kwargs['projectDir'] + '/GATOR/gatorPredict/')
    fileName = os.path.basename(kwargs['imagePath'])
    fileNamePrefix = fileName.split(os.extsep, 1)
    probabilityMaskPath = str(probPath / (fileNamePrefix[0] + '_gatorPredict.ome.tif'))
    
    # extract key words for generateGatorScore
    function2_args = inspect.signature(generateGatorScore).parameters.keys()
    function2_kwargs = {k: kwargs[k] for k in kwargs if k in function2_args}
    generateGatorScore (probabilityMaskPath=probabilityMaskPath, **function2_kwargs)


    ##########################################################################
    # STEP: 3 :- Generate gatorObject
    ##########################################################################
     
    # derive the path to Gator scores
    gPath = pathlib.Path(kwargs['projectDir'] + '/GATOR/gatorScore/')
    file_name = pathlib.Path(probabilityMaskPath).stem + '.csv'
    gatorScorePath = str(gPath / file_name)
    
    # extract key words for gatorObject
    function3_args = inspect.signature(gatorObject).parameters.keys()
    function3_kwargs = {k: kwargs[k] for k in kwargs if k in function3_args}
    gatorObject (gatorScorePath=gatorScorePath, **function3_kwargs)
    
    ##########################################################################
    # STEP: 4 :- Run Gator Algorithm
    ##########################################################################
    
    # derive the path to Gator object
    oPath = pathlib.Path(kwargs['projectDir'] + '/GATOR/gatorObject/')
    file_name = pathlib.Path(gatorScorePath).stem + '.h5ad'
    gatorObjectPath = str(oPath / file_name)
    
    # extract key words for running gator
    function4_args = inspect.signature(gator).parameters.keys()
    function4_kwargs = {k: kwargs[k] for k in kwargs if k in function4_args}
    gator (gatorObject=gatorObjectPath, **function4_kwargs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run gatorPipeline function')
    args = parser.parse_args()
    gatorPipeline(**vars(args))

