#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Wed Dec 28 17:08:09 2022
#@author: Ajit Johnson Nirmal
#Concatenate multiple anndata objects

"""
!!! abstract "Short Description"
    Use `mergecsObject` to combine multiple csObjects into a dataset for 
    analysis when multiple images need to be analyzed.
    
    Note that merging `csObjects` requires merging multiple sections, not 
    simple concatenation. Use parameters to specify which parts of the 
    `csObjects` to merge.
    
## Function
"""

# libs
import pathlib
import pandas as pd
import anndata as ad
import os
import argparse
import numpy as np

# function
def mergecsObject (csObjects,
                      fileName='mergedCSObject',
                      layers=['preProcessed'],
                      uns= ['cspotOutput','csScore','failedMarkers'],
                      verbose=True,
                      projectDir=None):
    """
Parameters:
    csObjects (list):
       A collection of CSPOT Objects to combine into one object, which can
       include both CSPOT Objects stored in memory and those accessed
       via file path.

    fileName (str, optional):
        Designate a Name for the resulting combined CSPOT object.

    layers (list, optional):
        The `.layers` section within the CSPOT Objects to be merged together.

    uns (list, optional):
        The `.uns` section within the CSPOT Objects to be merged together.

    verbose (bool, optional):
        If True, print detailed information about the process to the console. 

    projectDir (str, optional):
        Provide the path to the output directory. The result will be located at
        `projectDir/CSPOT/mergedCSObject/`. 

Returns:
    csObject (anndata):
        If `projectDir` is provided the merged CSPOT Object will saved within the
        provided projectDir.

Example:
        ```python.
        
        # set the working directory & set paths to the example data
        projectDir = '/Users/aj/Documents/cspotExampleData'
        
        csObjects = [projectDir + '/CSPOT/csOutput/exampleImage_cspotPredict.ome.h5ad',
                     projectDir + '/CSPOT/csOutput/exampleImage_cspotPredict.ome.h5ad']
        
        # For this tutorial, supply the same csObject twice for merging, but multiple csObjects can be merged in ideal conditions.
        adata = cs.mergecsObject ( csObjects=csObjects,
                              fileName='mergedcspotObject',
                              layers=['preProcessed'],
                              uns= ['cspotOutput','csScore'],
                              projectDir=projectDir)
        
        # Same function if the user wants to run it via Command Line Interface
        python mergecsObject.py \
            --csObjects /Users/aj/Documents/cspotExampleData/CSPOT/cspotOutput/exampleImage_cspotPredict.ome.h5ad /Users/aj/Documents/cspotExampleData/CSPOT/cspotOutput/exampleImage_cspotPredict.ome.h5ad \
            --projectDir /Users/aj/Documents/cspotExampleData
        
        
        ```
    """

    # Convert to list of anndata objects
    if isinstance (csObjects, list):
        csObjects = csObjects
    else:
        csObjects = [csObjects]

    # converting other parameters to list
    if isinstance (layers, str):
        layers = [layers]
    if isinstance (uns, str):
        uns = [uns]

    # Things to process
    process_layers = ['rawData', 'scaledData', 'obs']
    if layers is not None:
        process_layers.extend(layers)
    if uns is not None:
        process_layers.extend(uns)


    # for expression, uns and layers
    def processX (csObject, process_layers):
        if isinstance(csObject, str):
            # start with raw data
            adata = ad.read(csObject)
        else:
            adata = csObject.copy()

        # print
        if verbose is True:
            print ("Extracting data from: " + str( adata.obs['imageid'].unique()[0]) )

        # process the data
        rawData = pd.DataFrame(adata.raw.X, index=adata.obs.index, columns=adata.var.index)

        # scaled data
        scaledData = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index)

        # obs
        obs = adata.obs.copy()

        # Process layers
        if layers is not None:
            for i in layers:
                exec(f"{i} = pd.DataFrame(adata.layers[i],index=adata.obs.index, columns=adata.var.index)")

        # Process uns
        if uns is not None:
            for j in uns:
                exec(f"{j} = adata.uns[j]")

        # return the results
        objects = []
        for name in process_layers:
            exec(f"objects.append({name})")

        return objects

    # Run the function
    # Run the function:
    if verbose is True:
        print("Extracting data")
    r_processX = lambda x: processX (csObject=x, process_layers=process_layers)
    processX_result = list(map(r_processX, csObjects)) # Apply function

    # combine all the data
    # create a dictinoary between index and data type
    mapping = {i: element for i, element in enumerate(process_layers)}


    # create an empty dictionary to store the final dataframes
    final_data = {}
    # get the number of lists in the data list
    num_lists = len(processX_result)
    # iterate over the mapping dictionary
    for key, value in mapping.items():
        # create an empty list to store the dataframes
        df_list = []
        # iterate over the data lists
        for i in range(num_lists):
            # retrieve the dataframe from the current list
            df = processX_result[i][key]
            # resolve dict independently
            if isinstance(df, dict):
                df = pd.DataFrame.from_dict(df, orient='index', columns=df[list(df.keys())[0]]).applymap(lambda x: 1)   
            # add the dataframe to the df_list
            df_list.append(df)
        # concatenate the dataframes in the df_list
        df = pd.concat(df_list)
        # add the resulting dataframe to the final_data dictionary
        final_data[value] = df


    # create the combined anndata object
    bdata = ad.AnnData(final_data.get("rawData"), dtype=np.float64)
    bdata.obs = final_data.get("obs")
    bdata.raw = bdata
    bdata.X = final_data.get("scaledData")
    # add layers
    if layers is not None:
        for i in layers:
            tmp = final_data.get(i)
            bdata.layers[i] = tmp
    # add uns
    if uns is not None:
        for i in uns:
            tmp = final_data.get(i)
            bdata.uns[i] = tmp

    # last resolve all_markers if it exisits
    if isinstance(csObjects[0], str):
        # start with raw data
        adata = ad.read(csObjects[0])
    else:
        adata = csObjects[0].copy()
    if hasattr(adata, 'uns') and 'all_markers' in adata.uns:
        # the call to adata.uns['all_markers'] is valid
        bdata.uns['all_markers'] = adata.uns['all_markers']

    # write the output
    if projectDir is not None:
        finalPath = pathlib.Path(projectDir + '/CSPOT/mergedcsObject')
        if not os.path.exists(finalPath):
            os.makedirs(finalPath)
        bdata.write(finalPath / f'{fileName}.h5ad')
        # Print
        if verbose is True:
            print('Given csObjects have been merged, head over to "' + str(projectDir) + '/CSPOT/mergedcsObject" to view results')


    # return data
    return bdata

# Make the Function CLI compatable
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge CSPOT Objects')
    parser.add_argument('--csObjects', type=str, nargs='+', help='A collection of CSPOT Objects to combine into one object')
    parser.add_argument('--fileName', type=str, default='mergedcsObject', help='Designate a Name for the resulting combined CSPOT object')
    parser.add_argument('--layers', type=str, nargs='+', default=['preProcessed'], help='The layers section within the CSPOT Objects to be merged together')
    parser.add_argument('--uns', type=str, nargs='+', default=['cspotOutput','csScore'], help='The uns section within the CSPOT Objects to be merged together')
    parser.add_argument("--verbose", type=bool, default=True, help="If True, print detailed information about the process to the console.")       
    parser.add_argument('--projectDir', type=str, default=None, help='Provide the path to the output directory')
    args = parser.parse_args()
    mergecsObject(csObjects=args.csObjects,
                     fileName=args.fileName,
                     layers=args.layers, 
                     uns=args.uns,
                     verbose=args.verbose,
                     projectDir=args.projectDir)
