# -*- coding: utf-8 -*-
# Created on Mon Nov  9 21:00:57 2020
# @author: Ajit Johnson Nirmal
"""
!!! abstract "Short Description"
    Users can utilize the `gatorExport` function to store the contents of the gatorObject to a `.CSV` file. 
      
    Keep in mind that the presence of multiple intermediate files in the object will result in the production of several CSV files.

## Function
"""

# Import
import pandas as pd
import argparse
import pathlib
import anndata as ad
import os

# Function
def gatorExport (gatorObject, 
                 projectDir, 
                 fileName=None, 
                 raw=False, 
                 CellID='CellID',
                 verbose=True):
    """
Parameters:
    gatorObject (anndata):
        Pass the `gatorObject` loaded into memory or a path to the `gatorObject` 
        file (.h5ad).
        
    projectDir (str, optional):
        Provide the path to the output directory. The result will be located at
        `projectDir/GATOR/gatorExport/`. 

    fileName (str, optional):
        Specify the name of the CSV output file. If you don't provide a file name, t
        he default name `gatorExport.csv` will be assigned.
        
    raw (bool, optional):
        If `True` raw data will be returned instead of the gator scaled data.
        
    CellId (str, optional):
        Specify the column name that holds the cell ID (a unique name given to each cell).

    verbose (bool, optional):
        If True, print detailed information about the process to the console.  

Returns:
    CSV files (.csv):
        The `.csv` files can be found under `projectDir/GATOR/gatorExport/`

Example:

        ```python
        # path to files
        projectDir = '/Users/aj/Desktop/gatorExampleData'
        gatorObject = projectDir + '/GATOR/gatorOutput/exampleImage_gatorPredict.ome.h5ad'
        ga.gatorExport (gatorObject, 
                         projectDir, 
                         fileName=None, 
                         raw=False, 
                         CellID='CellID',
                         verbose=True)
        
        # Same function if the user wants to run it via Command Line Interface
        python gatorExport.py --gatorObject /Users/aj/Desktop/gatorExampleData/GATOR/gatorOutput/exampleImage_gatorPredict.ome.h5ad --projectDir /Users/aj/Desktop/gatorExampleData
        
        ```

    """
    
    
    # projectDir = '/Users/aj/Desktop/gatorExampleData'
    # gatorObject='/Users/aj/Desktop/gatorExampleData/GATOR/gatorOutput/exampleImage_gatorPredict.ome.h5ad' 
    # data_type='raw'; output_dir=None; fileName=None; CellID='CellID'
    
    # Load the andata object    
    if isinstance(gatorObject, str):
        if fileName is None:
            imid = pathlib.Path(gatorObject).stem
        else: 
            imid = str(fileName)
        gatorObject = ad.read(gatorObject)
    else:
        if fileName is None:
            imid = "gatorExport.csv"
        else: 
            imid = str(fileName)
        gatorObject = gatorObject
    
    # Expression matrix & obs data
    if raw is True:
        data = pd.DataFrame(gatorObject.raw.X, index=gatorObject.obs.index, columns=gatorObject.var.index)
    else:
        data = pd.DataFrame(gatorObject.X, index=gatorObject.obs.index, columns=gatorObject.var.index)
    meta = pd.DataFrame(gatorObject.obs)
    # Merge the two dataframes
    merged = pd.concat([data, meta], axis=1, sort=False)
    

    # Add a column to save cell-id
    # make cellID the first column
    if CellID in merged.columns:
        first_column = merged.pop(CellID)
        merged.insert(0, CellID, first_column)
    else:
        merged['CellID'] = merged.index
        first_column = merged.pop(CellID)
        merged.insert(0, CellID, first_column)
    
    # reset index
    merged = merged.reset_index(drop=True)
    
    # create a folder to hold the results
    folderPath = pathlib.Path(projectDir + '/GATOR/gatorExport/')
    folderPath.mkdir(exist_ok=True, parents=True)
    
    
    # extract some of the data stored in .uns and save
    if hasattr(gatorObject, 'uns') and 'gatorOutput' in gatorObject.uns:
        gatorOutput = gatorObject.uns['gatorOutput']
        gatorOutput.index = merged['CellID']
        gatorOutput.to_csv(folderPath / 'gatorOutput.csv')
    if hasattr(gatorObject, 'uns') and 'gatorScore' in gatorObject.uns:    
        gatorScore = gatorObject.uns['gatorScore']
        gatorScore.index = merged['CellID']
        gatorScore.to_csv(folderPath / 'gatorScore.csv')
    
    # scaled data
    merged.to_csv(folderPath / f'{imid}.csv', index=False)
    
    # Finish Job
    if verbose is True:
        print('Contents of the gatorObject have been exported to "' + str(projectDir) + '/GATOR/gatorExport"')

    
if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Export Gator results to CSV.')
    parser.add_argument('--gatorObject', type=str, help='Path to the gatorObject file (.h5ad)', required=True)
    parser.add_argument('--projectDir', type=str, help='Path to the output directory', required=True)
    parser.add_argument('--fileName', type=str, help='Name of the CSV output file', default='gatorExport.csv')
    parser.add_argument('--raw', type=bool, default=False, help='Return raw data instead of gator scaled data')
    parser.add_argument('--CellID', type=str, help='Column name that holds the cell ID', default='CellID')
    parser.add_argument('--verbose', type=bool, default=True, help='Print detailed information about the process')
    args = parser.parse_args()
    
    gatorExport(gatorObject=args.gatorObject, 
                projectDir=args.projectDir, 
                fileName=args.fileName, 
                raw=args.raw, 
                CellID=args.CellID, 
                verbose=args.verbose)



