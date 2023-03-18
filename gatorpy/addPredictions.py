#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Sat Feb  4 12:06:32 2023
#@author: Ajit Johnson Nirmal
#Helper function to add the gatorPredicted postive and negative cells to vizualize 
#on the image using scimap. 

"""
!!! abstract "Short Description"
    The `addPredictions` function serves as a link between `gatorpy` and `scimap` package. 
    It's useful for evaluating model performance. The function transforms results 
    stored in `anndata.uns` to `anndata.obs` so they can be visualized using 
    the `scimap` package's `sm.pl.image viewer` function. This displays `positive` 
    and `negative` cells overlaid on the raw image.
    
    The `addPredictions` function can take in two methods. 
    `gatorOutput` displays the result of running the `gator` function, 
    while `gatorScore` shows the raw output produced by the `gatorScore` 
    function, which returns a probability score. The `midpoint` parameter, 
    with a default value of 0.5, can be adjusted to define what is 
    considered a `positive` result, when method is set to `gatorScore`.
    
## Function
"""

# libs
import pandas as pd
import anndata as ad
import pathlib
import os
import argparse


# Function
def addPredictions (gatorObject, 
                    method='gatorOutput',
                    gatorOutput='gatorOutput',
                    gatorScore='gatorScore', 
                    midpoint=0.5,
                    outputDir=None):
    """
Parameters:

    gatorObject (anndata):  
        Single or combined Gator object.
        
    method (str, optional):  
        There are two options: `gatorOutput` and `gatorScore`. 
        `gatorOutput` displays the result of running the `gator` function, 
        while `gatorScore` shows the raw output produced by the `gatorScore` 
        function, which returns a probability score. The `midpoint` parameter, 
        with a default value of 0.5, can be adjusted to define what is 
        considered a `positive` result, when method is set to `gatorScore`.
        
    gatorOutput (str, optional):  
        The name under which the `gatorOutput` is stored.
        
    gatorScore (str, optional):  
        The name under which the `gatorScore` is stored.
        
    midpoint (float, optional):  
        The threshold for determining positive cells, in conjunction with 'gatorScore'.

    outputDir (string, optional):  
        Provide the path to the output directory. Kindly take note that this particular 
        output will not be automatically saved in a predetermined directory, 
        unlike the other outputs. The file will be saved in the directory 
        specified by the `outputDir` parameter. If `None`, the `gatorObject` will 
        be returned to memory.

Returns:

    gatorObject (anndata):  
        If output directory is provided the `gatorObject` will 
        be stored else it will be returned to memory. The results are stored in 
        `anndata.obs` with a `p_` appended to the markers names. So if you would 
        like to vizulaize `CD3`, the column that you are looking for is `p_CD3`.
        
Example:

    	```python    
        
        # set the working directory & set paths to the example data
        cwd = '/Users/aj/Desktop/gatorExampleData'
        
        # Module specific paths
        gatorObject = cwd + '/GATOR/gatorOutput/exampleImage_gatorPredict.ome.h5ad'
        
        adata = ga.addPredictions (gatorObject, 
                        method='gatorOutput',
                        gatorOutput='gatorOutput',
                        gatorScore='gatorScore', 
                        midpoint=0.5,
                        outputDir=None)
        
        # Same function if the user wants to run it via Command Line Interface
        python addPredictions.py --gatorObject Users/aj/Desktop/gatorExampleData/GATOR/gatorOutput/exampleImage_gatorPredict.ome.h5ad    	

    """
    # Load the adata
    if isinstance(gatorObject, str):
        adata = ad.read(gatorObject)
    else: 
        adata = gatorObject
        
    # function to convert the prob scores to binary pos or neg
    def assign_labels(df, midpoint):
        df = df.applymap(lambda x: 'neg' if x < midpoint else 'pos')
        return df
    
    # intialize the data    
    if method == 'gatorOutput':
        attach_df = adata.uns[gatorOutput].copy()
    elif method == 'gatorScore':
        df = adata.uns[gatorScore].copy()
        attach_df = assign_labels (df, midpoint=midpoint)
        
        
    obs = adata.obs.copy()
    columns_to_drop = [col for col in obs.columns if col.startswith('p_')]
    obs.drop(columns_to_drop, axis=1, inplace=True)
    
    new_col_names = ['p_{}'.format(idx) for idx in attach_df.columns]
    attach_df.columns = new_col_names
    # add to obs
    final_obs = pd.concat([obs, attach_df], axis=1)
    adata.obs = final_obs
    
    
    # Return to adata
    # Save data if requested
    if outputDir is not None:    
        finalPath = pathlib.Path(outputDir)     
        if not os.path.exists(finalPath):
            os.makedirs(finalPath)
        # determine file name
        if isinstance (gatorObject, str):
            imid = pathlib.Path(gatorObject).stem
        else:
            imid = 'addPredictions'    
        adata.write(finalPath / f'{imid}.h5ad')
    else:
        # Return data
        return adata
    
    

# Make the Function CLI compatable
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add predictions to an anndata object.')
    parser.add_argument('--gatorObject', type=str, help='Single or combined Gator object.')
    parser.add_argument('--method', type=str, default='gatorOutput', help='Method for determining positive cells (gatorOutput or gatorScore).')
    parser.add_argument('--gatorOutput', type=str, default='gatorOutput', help='Name under which gatorOutput is stored.')
    parser.add_argument('--gatorScore', type=str, default='gatorScore', help='Name under which gatorScore is stored.')
    parser.add_argument('--midpoint', type=float, default=0.5, help='Threshold for determining positive cells, in conjunction with gatorScore.')
    parser.add_argument('--outputDir', type=str, default=None, help='Path to the output directory. If None, gatorObject will be returned to memory.')
    args = parser.parse_args()
    addPredictions(gatorObject=args.gatorObject, 
                   method=args.method, 
                   gatorOutput=args.gatorOutput, 
                   gatorScore=args.gatorScore, 
                   midpoint=args.midpoint, 
                   outputDir=args.outputDir)
