#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Sat Feb  4 12:06:32 2023
#@author: Ajit Johnson Nirmal
#Helper function to add the cspotPredicted postive and negative cells to vizualize 
#on the image using scimap. 

"""
!!! abstract "Short Description"
    The `addPredictions` function serves as a link between `cspot` and `scimap` package. 
    It's useful for evaluating model performance. The function transforms results 
    stored in `anndata.uns` to `anndata.obs` so they can be visualized using 
    the `scimap` package's `sm.pl.image viewer` function. This displays `positive` 
    and `negative` cells overlaid on the raw image.
    
    The `addPredictions` function can take in two methods. 
    `cspotOutput` displays the result of running the `cspot` function, 
    while `csScore` shows the raw output produced by the `csScore` 
    function, which returns a probability score. The `midpoint` parameter, 
    with a default value of 0.5, can be adjusted to define what is 
    considered a `positive` result, when method is set to `csScore`.
    
## Function
"""

# libs
import pandas as pd
import anndata as ad
import pathlib
import os
import argparse


# Function
def addPredictions (csObject, 
                    method='cspotOutput',
                    cspotOutput='cspotOutput',
                    csScore='csScore', 
                    midpoint=0.5,
                    outputDir=None):
    """
Parameters:
    csObject (anndata):  
        Single or combined CSPOT object.
        
    method (str, optional):  
        There are two options: `cspotOutput` and `csScore`. 
        `cspotOutput` displays the result of running the `CSPOT` function, 
        while `csScore` shows the raw output produced by the `csScore` 
        function, which returns a probability score. The `midpoint` parameter, 
        with a default value of 0.5, can be adjusted to define what is 
        considered a `positive` result, when method is set to `csScore`.
        
    cspotOutput (str, optional):  
        The name under which the `cspotOutput` is stored.
        
    csScore (str, optional):  
        The name under which the `csScore` is stored.
        
    midpoint (float, optional):  
        The threshold for determining positive cells, in conjunction with 'csScore'.

    outputDir (string, optional):  
        Provide the path to the output directory. Kindly take note that this particular 
        output will not be automatically saved in a predetermined directory, 
        unlike the other outputs. The file will be saved in the directory 
        specified by the `outputDir` parameter. If `None`, the `csObject` will 
        be returned to memory.

Returns:
    csObject (anndata):  
        If output directory is provided the `csObject` will 
        be stored else it will be returned to memory. The results are stored in 
        `anndata.obs` with a `p_` appended to the markers names. So if you would 
        like to vizulaize `CD3`, the column that you are looking for is `p_CD3`.
        
Example:
    	```python    
        
        # Path to projectDir
        projectDir = '/Users/aj/Documents/cspotExampleData'
        
        # Path to csObject
        csObject = projectDir + '/CSPOT/cspotOutput/exampleImage_cspotPredict.ome.h5ad'
        
        adata = cs.addPredictions (csObject, 
                            method='cspotOutput',
                            cspotOutput='cspotOutput',
                            csScore='csScore', 
                            midpoint=0.5)
        
        # Same function if the user wants to run it via Command Line Interface
        python addPredictions.py \
            --csObject Users/aj/Documents/cspotExampleData/CSPOT/cspotOutput/exampleImage_cspotPredict.ome.h5ad    	
        ```

    """
    # Load the adata
    if isinstance(csObject, str):
        adata = ad.read(csObject)
    else: 
        adata = csObject
        
    # function to convert the prob scores to binary pos or neg
    def assign_labels(df, midpoint):
        df = df.applymap(lambda x: 'neg' if x < midpoint else 'pos')
        return df
    
    # intialize the data    
    if method == 'cspotOutput':
        attach_df = adata.uns[cspotOutput].copy()
    elif method == 'csScore':
        df = adata.uns[csScore].copy()
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
        if isinstance (csObject, str):
            imid = pathlib.Path(csObject).stem
        else:
            imid = 'addPredictions'    
        adata.write(finalPath / f'{imid}.h5ad')
    else:
        # Return data
        return adata
    
    

# Make the Function CLI compatable
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add predictions to an anndata object.')
    parser.add_argument('--csObject', type=str, help='Single or combined CSPOT object.')
    parser.add_argument('--method', type=str, default='cspotOutput', help='Method for determining positive cells (cspotOutput or csScore).')
    parser.add_argument('--cspotOutput', type=str, default='cspotOutput', help='Name under which cspotOutput is stored.')
    parser.add_argument('--csScore', type=str, default='csScore', help='Name under which csScore is stored.')
    parser.add_argument('--midpoint', type=float, default=0.5, help='Threshold for determining positive cells, in conjunction with csScore.')
    parser.add_argument('--outputDir', type=str, default=None, help='Path to the output directory. If None, csObject will be returned to memory.')
    args = parser.parse_args()
    addPredictions(csObject=args.csObject, 
                   method=args.method, 
                   cspotOutput=args.cspotOutput, 
                   csScore=args.csScore, 
                   midpoint=args.midpoint, 
                   outputDir=args.outputDir)
