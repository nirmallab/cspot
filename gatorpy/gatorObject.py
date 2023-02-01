# -*- coding: utf-8 -*-
#Created on Thu Nov 10 09:53:28 2022
#@author: Ajit Johnson Nirmal
#Function to incorporate the feature matrix and probability matrix into an adata object

"""
!!! abstract "Short Description"
    The `gatorObject` function creates a gator object using the anndata 
    framework by inputting DLScores and a pre-calculated single-cell spatial table. 
    This centralizes all information into one file, streamlining the data analysis 
    process and reducing the risk of losing data.

## Function
"""

# libs
import anndata as ad
import pandas as pd
import pathlib
import numpy as np
import os
import argparse

# function
def gatorObject (spatialTablePath,
                 DLScorePath,
                 CellId='CellID',
                 uniqueCellId=True,
                 split='X_centroid',
                 removeDNA=True,
                 remove_string_from_name=None,
                 log=True,
                 dropMarkers=None,
                 outputDir=None):
    """
Parameters:
    spatialTablePath (list):
        Provide a list of paths to the single-cell spatial feature tables, ensuring each image has a unique path specified.

    DLScorePath (list):
        Supply a list of paths to the DL score tables created using generateDLScore,
        ensuring they correspond to the image paths specified.

    CellId (str, optional):
        Specify the column name that holds the cell ID (a unique name given to each cell).

    uniqueCellId (bool, optional):
        The function generates a unique name for each cell by combining the CellId and imageid.
        If you don't want this, pass False. In such case the function will default to using just the CellId.
        However, make sure CellId is unique especially when loading multiple images together.

    split (string, optional):
        The spatial feature table generally includes single cell expression data
        and meta data such as X, Y coordinates, and cell shape size. The Gator
        object separates them. Ensure that the expression data columns come first,
        followed by meta data columns. Provide the column name that marks the split,
        i.e the column name immediately following the expression data.

    removeDNA (bool, optional):
        Exclude DNA channels from the final output. The function searches for
        column names containing the string `dna` or `dapi`. 

    remove_string_from_name (string, optional):
        Cleans up channel names by removing user specified string from all marker
        names. 

    log (bool, optional):
        Apply log1p transformation to log the data. 

    dropMarkers (list, optional):
        Specify a list of markers to be removed from the analysis, for
        example: ["background_channel", "CD20"]. 

    outputDir (string, optional):
        Provide the path to the output directory. The result will be located at
        `outputDir/GATOR/gatorObject/`.

Returns:
    gatorObject (anndata):
        If outputDir is provided the Gator Object will be saved as a
        `.h5ad` file in the provided directory.

Example:

        ```python
        
        # set the working directory & set paths to the example data
        cwd = '/Users/aj/Desktop/gatorExampleData'
        
        # Module specific paths
        spatialTablePath = cwd + '/quantification/exampleSpatialTable.csv'
        DLScorePath = cwd + '/GATOR/DLScore/exampleProbabiltyMap.ome.csv'
        
        # please note that there are a number of defaults in the below function that assumes certain structure within the spatialTable.
        # Please confirm it is similar with user data or modifiy the parameters accordingly
        # check out the documentation for further details
        adata = ga.gatorObject (spatialTablePath=spatialTablePath,
                        DLScorePath=DLScorePath,
                        CellId='CellID',
                        uniqueCellId=True,
                        split='X_centroid',
                        removeDNA=True,
                        remove_string_from_name=None,
                        log=True,
                        dropMarkers=None,
                        outputDir=cwd)
        
        # Same function if the user wants to run it via Command Line Interface
        python gatorObject.py --spatialTablePath /Users/aj/Desktop/gatorExampleData/quantification/exampleSpatialTable.csv --DLScorePath /Users/aj/Desktop/gatorExampleData/GATOR/DLScore/exampleProbabiltyMap.ome.csv --outputDir /Users/aj/Desktop/gatorExampleData
        
        ```

    """
#spatialTablePath = r"C:\Users\ajn16\Dropbox (Partners HealthCare)\Data\gator\data\Exemplar\modified_dearray\quantification\unmicst-113_cellMask.csv"
#probTablePath = r"C:\Users\ajn16\Dropbox (Partners HealthCare)\Data\gator\data\ajn_training_data\GATOR\probQuant\113_GatorOutput.csv"
#dropMarkers = ['bg2b', 'bg3b', 'bg4b', 'ECAD_2']

#outputDir = 'C:/Users/ajn16/Dropbox (Partners HealthCare)/Data/gator/data/ajn_training_data'
#gatorObject (spatialTablePath=spatialTablePath, probTablePath=probTablePath, outputDir=outputDir, dropMarkers=dropMarkers)

    # spatialTablePath list or string
    if isinstance(spatialTablePath, str):
        spatialTablePath = [spatialTablePath]
    spatialTablePath = [pathlib.Path(p) for p in spatialTablePath]
    # DLScorePath list or string
    if isinstance(DLScorePath, str):
        DLScorePath = [DLScorePath]
    DLScorePath = [pathlib.Path(p) for p in DLScorePath]

    # Import spatialTablePath
    def load_process_data (image):
        # Print the data that is being processed
        print(f"Loading {image.name}")
        d = pd.read_csv(image)
        # If the data does not have a unique image ID column, add one.
        if 'imageid' not in d.columns:
            imid = image.stem
            d['imageid'] = imid
        # Unique name for the data
        if uniqueCellId is True:
            d.index = d['imageid'].astype(str)+'_'+d[CellId].astype(str)
        else:
            d.index = d[CellId]

        # move image id and cellID column to end
        cellid_col = [col for col in d.columns if col != CellId] + [CellId]; d = d[cellid_col]
        imageid_col = [col for col in d.columns if col != 'imageid'] + ['imageid']; d = d[imageid_col]
        # If there is INF replace with zero
        d = d.replace([np.inf, -np.inf], 0)
        # Return data
        return d

    # Import DLScorePath
    def load_process_probTable (image):
        d = pd.read_csv(image, index_col=0)
        # Return data
        return d

    # Apply function to all spatialTablePath and create a master dataframe
    r_load_process_data = lambda x: load_process_data(image=x) # Create lamda function
    all_spatialTable = list(map(r_load_process_data, list(spatialTablePath))) # Apply function
    # Merge all the spatialTablePath into a single large dataframe
    for i in range(len(all_spatialTable)):
        all_spatialTable[i].columns = all_spatialTable[0].columns
    entire_spatialTable = pd.concat(all_spatialTable, axis=0, sort=False)

    # Apply function to all DLScorePath and create a master dataframe
    r_load_process_probTable = lambda x: load_process_probTable(image=x) # Create lamda function
    all_probTable = list(map(r_load_process_probTable, list(DLScorePath))) # Apply function
    # Merge all the DLScorePath into a single large dataframe
    for i in range(len(all_probTable)):
        all_probTable[i].columns = all_probTable[0].columns
    entire_probTable = pd.concat(all_probTable, axis=0, sort=False)
    # make the index of entire_probTable same as all_probTable
    ## NOTE THIS IS A HARD COPY WITHOUT ANY CHECKS! ASSUMES BOTH ARE IN SAME ORDER
    entire_probTable.index = entire_spatialTable.index


    # Split the data into expression data and meta data
    # Step-1 (Find the index of the column with name X_centroid)
    split_idx = entire_spatialTable.columns.get_loc(split)
    meta = entire_spatialTable.iloc [:,split_idx:]
    # Step-2 (select only the expression values)
    entire_spatialTable = entire_spatialTable.iloc [:,:split_idx]

    # Rename the columns of the data
    if remove_string_from_name is not None:
        entire_spatialTable.columns = entire_spatialTable.columns.str.replace(remove_string_from_name, '')

    # Save a copy of the column names in the uns space of ANNDATA
    markers = list(entire_spatialTable.columns)

    # Remove DNA channels
    if removeDNA is True:
        entire_spatialTable = entire_spatialTable.loc[:,~entire_spatialTable.columns.str.contains('dna', case=False)]
        entire_spatialTable = entire_spatialTable.loc[:,~entire_spatialTable.columns.str.contains('dapi', case=False)]

    # Drop unnecessary markers
    if dropMarkers is not None:
        if isinstance(dropMarkers, str):
            dropMarkers = [dropMarkers]
        dropMarkers = list(set(dropMarkers).intersection(entire_spatialTable.columns))
        entire_spatialTable = entire_spatialTable.drop(columns=dropMarkers)

    # Create an anndata object
    adata = ad.AnnData(entire_spatialTable, dtype=np.float64)
    adata.obs = meta
    adata.uns['all_markers'] = markers
    adata.uns['DLScore'] = entire_probTable

    # Add log data
    if log is True:
        adata.raw = adata
        adata.X = np.log1p(adata.X)

    # Save data if requested
    if outputDir is not None:
        finalPath = pathlib.Path(outputDir + '/GATOR/gatorObject')
        if not os.path.exists(finalPath):
            os.makedirs(finalPath)
        if len(spatialTablePath) > 1:
            imid = 'gatorObject'
        else:
            imid = DLScorePath[0].stem
        adata.write(finalPath / f'{imid}.h5ad')
    else:
        # Return data
        return adata

# Make the Function CLI compatable
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a GatorObject from single-cell spatial feature tables.')
    parser.add_argument('--spatialTablePath', type=str, nargs='+', help='Provide a list of paths to the single-cell spatial feature tables.')
    parser.add_argument('--DLScorePath', type=str, nargs='+', help='Supply a list of paths to the DL score tables created using generateDLScore.')
    parser.add_argument('--CellId', type=str, default='CellID', help='Specify the column name that holds the cell ID.')
    parser.add_argument('--uniqueCellId', type=bool, default=True, help='The function generates a unique name for each cell by combining the CellId and imageid.')
    parser.add_argument('--split', type=str, default='X_centroid', help='Provide the column name that marks the split between expression data and meta data.')
    parser.add_argument('--removeDNA', type=bool, default=True, help='Exclude DNA channels from the final output.')
    parser.add_argument('--remove_string_from_name', type=str, default=None, help='Cleans up channel names by removing user specified string from all marker names.')
    parser.add_argument('--log', type=bool, default=True, help='Apply log1p transformation to log the data.')
    parser.add_argument('--dropMarkers', type=str, nargs='+', default=None, help='Specify a list of markers to be removed from the analysis.')
    parser.add_argument('--outputDir', type=str, default=None, help='Provide the path to the output directory.')
    args = parser.parse_args()
    gatorObject(spatialTablePath=args.spatialTablePath,
                DLScorePath=args.DLScorePath,
                CellId=args.CellId,
                uniqueCellId=args.uniqueCellId,
                split=args.split,
                removeDNA=args.removeDNA,
                remove_string_from_name=args.remove_string_from_name,
                log=args.log,
                dropMarkers=args.dropMarkers,
                outputDir=args.outputDir)
