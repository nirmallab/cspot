#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Wed Mar  8 21:51:38 2023
#@author: Ajit Johnson Nirmal
#Run CSPOT Pipeline

"""
!!! abstract "Short Description"
    The csPipeline function is simply a wrapper for the following functions:  
    - csPredict  
    - generateCSScore  
    - csObject  
    - cspot  
      
    Typically, in production settings, `csPipeline` would be utilized, whereas 
    step-by-step analysis would be employed for troubleshooting, model validation, 
    and similar tasks that necessitate greater granularity or control.
      
    Please refer to the individual function documentation for parameter tuning.

## Function
"""

# libs
import inspect
import argparse
import pathlib
import os

# CSPOT Functions
if __name__ == '__main__':
    from csPredict import csPredict
    from generateCSScore import generateCSScore
    from csObject import csObject
    from cspot import cspot
else:
    from .csPredict import csPredict
    from .generateCSScore import generateCSScore
    from .csObject import csObject
    from .cspot import cspot



# Function
def csPipeline (**kwargs):   
    """
Parameters:
    imagePath (str):  
        The path to the .tif file that needs to be processed. 
     
    csModelPath (str):  
        The path to the `cspotModel` folder. 

    markerChannelMapPath (str, optional):  
        The path to the marker panel list, which contains information about the markers used in the image. This argument is required.

    segmentationMaskPath (str):
        Supply the path of the pre-computed segmentation mask.

    spatialTablePath (list):
        Provide a list of paths to the single-cell spatial feature tables, ensuring each image has a unique path specified.
         
    projectDir (str):  
        The path to the output directory where the processed images (`probabilityMasks`) will be saved.

    verbose (bool, optional):
        If True, print detailed information about the process to the console.  
        
    markerColumnName (str, optional):  
        The name of the column in the marker panel list that contains the marker names. The default value is 'marker'.
     
    channelColumnName (str, optional):  
        The name of the column in the marker panel list that contains the channel names. The default value is 'channel'.
     
    modelColumnName (str, optional):  
        The name of the column in the marker panel list that contains the model names. The default value is 'cspotmodel'.

    GPU (int, optional):  
        An optional argument to explicitly select the GPU to use. The default value is -1, meaning that the GPU will be selected automatically.

    feature (str, optional):
        Calculates the `mean` or `median` CSPOT Score for each cell.

    markerNames (list, optional):
        The program searches for marker names in the meta data (description section)
        of the tiff files created by `csPredict` by default. If the meta data
        is lost due to user modifications, provide the marker names for each
        channel/layer in the `probabilityMaskPath` here.

    CellId (str, optional):
        Specify the column name that holds the cell ID (a unique name given to each cell).

    uniqueCellId (bool, optional):
        The function generates a unique name for each cell by combining the CellId and imageid.
        If you don't want this, pass False. In such case the function will default to using just the CellId.
        However, make sure CellId is unique especially when loading multiple images together.

    split (string, optional):
        The spatial feature table generally includes single cell expression data
        and meta data such as X, Y coordinates, and cell shape size. The CSPOT
        object separates them. Ensure that the expression data columns come first,
        followed by meta data columns. Provide the column name that marks the split,
        i.e the column name immediately following the expression data.

    removeDNA (bool, optional):
        Exclude DNA channels from the final output. The function searches for
        column names containing the string `dna` or `dapi`. 

    remove_string_from_name (string, optional):
        Cleans up channel names by removing user specified string from all marker
        names.

    csScore (str, optional):
        Include the label used for saving the `csScore` within the CSPOT object.

    minAbundance (float, optional):
        Specify the minimum percentage of cells that should express a specific
        marker in order to determine if the marker is considered a failure.
        A good approach is to consider the lowest percentage of rare cells
        expected within the dataset.

    percentiles (list, optional):
        Specify the interval of percentile levels of the expression utilized to intialize
        the GMM. The cells falling within these percentiles are utilized to distinguish
        between negative cells (first two values) and positive cells (last two values).

    dropMarkers (list, optional):
        Specify a list of markers to be removed from the analysis, for
        example: `["background_channel1", "background_channel2"]`. 

    RobustScale (bool, optional):
        When set to True, the data will be subject to Robust Scaling before the
        Gradient Boosting Classifier is trained. 

    log (bool, optional):
        Apply `log1p` transformation on the data, unless it has already been log
        transformed in which case set it to `False`. 

    stringentThreshold (bool, optional):
        The Gaussian Mixture Model (GMM) is utilized to distinguish positive and 
        negative cells by utilizing csScores. The stringentThreshold can be utilized 
        to further refine the classification of positive and negative cells. 
        By setting it to True, cells with csScore below the mean of the negative 
        distribution and above the mean of the positive distribution will be 
        labeled as true negative and positive, respectively.
        
    x_coordinate (str, optional):
        The column name in `single-cell spatial table` that records the
        X coordinates for each cell. 

    y_coordinate (str, optional):
        The column name in `single-cell spatial table` that records the
        Y coordinates for each cell.

    imageid (str, optional):
        The name of the column that holds the unique image ID. 

    random_state (int, optional):
        Seed used by the random number generator. 

    rescaleMethod (string, optional):
        Choose between `sigmoid` and `minmax`.

    label (str, optional):
        Assign a label for the object within `adata.uns` where the predictions
        from CSPOT will be stored. 
    
    
Returns:
    csObject (anndata):
        Returns a csObject with predictions of all positve and negative cells. 

Example:
        ```python
        
        # Path to all the files that are necessary files for running the 
        CSPOT Prediction Algorithm (broken down based on sub functions)
        projectDir = '/Users/aj/Documents/cspotExampleData'

        # gatorPredict related paths
        imagePath = projectDir + '/image/exampleImage.tif'
        markerChannelMapPath = projectDir + '/markers.csv'
        csModelPath = projectDir + '/manuscriptModels/'
        
        # Generate generateGatorScore related paths
        segmentationPath = projectDir + '/segmentation/exampleSegmentationMask.tif'
        
        # gatorObject related paths
        spatialTablePath = projectDir + '/quantification/exampleSpatialTable.csv'
        
        # Run the pipeline
        cs.csPipeline(   
                    # parameters for gatorPredict function
                    imagePath=imagePath,
                    csModelPath=csModelPath,
                    markerChannelMapPath=markerChannelMapPath,

                    # parameters for generateGatorScore function
                    segmentationMaskPath=segmentationPath,

                    # parameters for gatorObject function
                    spatialTablePath=spatialTablePath,

                    # parameters to run gator function
                    # ..

                    # common parameters
                    verbose=False,
                    projectDir=projectDir)
        
        # Same function if the user wants to run it via Command Line Interface
        python csPipeline.py \
                --imagePath /Users/aj/Documents/cspotExampleData/image/exampleImage.tif \
                --csModelPath /Users/aj/Documents/cspotExampleData/CSPOT/cspotModel/ \
                --markerChannelMapPath /Users/aj/Documents/cspotExampleData/markers.csv \
                --segmentationMaskPath /Users/aj/Documents/cspotExampleData/segmentation/exampleSegmentationMask.tif \
                --spatialTablePath /Users/aj/Documents/cspotExampleData/quantification/exampleSpatialTable.csv \
                --projectDir /Users/aj/Documents/cspotExampleData
        ```
                
         
    """
    
    ##########################################################################
    # STEP: 1 :- PREDICT
    ##########################################################################
    function1_args = inspect.signature(csPredict).parameters.keys()
    # Extract only the arguments that csPredict expects from the keyword arguments
    function1_kwargs = {k: kwargs[k] for k in kwargs if k in function1_args}
    # Call csPredict with the extracted arguments
    csPredict (**function1_kwargs)
    
    
    ##########################################################################
    # STEP: 2 :- generateCSScore
    ##########################################################################
    
    # derive the probability mask path
    probPath = pathlib.Path(kwargs['projectDir'] + '/CSPOT/csPredict/')
    fileName = os.path.basename(kwargs['imagePath'])
    fileNamePrefix = fileName.split(os.extsep, 1)
    probabilityMaskPath = str(probPath / (fileNamePrefix[0] + '_cspotPredict.ome.tif'))
    
    # extract key words for generateCSScore
    function2_args = inspect.signature(generateCSScore).parameters.keys()
    function2_kwargs = {k: kwargs[k] for k in kwargs if k in function2_args}
    generateCSScore (probabilityMaskPath=probabilityMaskPath, **function2_kwargs)


    ##########################################################################
    # STEP: 3 :- Generate csObject
    ##########################################################################
     
    # derive the path to CSPOT scores
    gPath = pathlib.Path(kwargs['projectDir'] + '/CSPOT/csScore/')
    file_name = pathlib.Path(probabilityMaskPath).stem + '.csv'
    csScorePath = str(gPath / file_name)
    
    # extract key words for csObject
    function3_args = inspect.signature(csObject).parameters.keys()
    function3_kwargs = {k: kwargs[k] for k in kwargs if k in function3_args}
    csObject (csScorePath=csScorePath, **function3_kwargs)
    
    ##########################################################################
    # STEP: 4 :- Run CSPOT Algorithm
    ##########################################################################
    
    # derive the path to CSPOT object
    oPath = pathlib.Path(kwargs['projectDir'] + '/CSPOT/csObject/')
    file_name = pathlib.Path(csScorePath).stem + '.h5ad'
    csObjectPath = str(oPath / file_name)
    
    # extract key words for running CSPOT
    function4_args = inspect.signature(cspot).parameters.keys()
    function4_kwargs = {k: kwargs[k] for k in kwargs if k in function4_args}    
    cspot (csObject=csObjectPath, **function4_kwargs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run csPipeline function')
    parser.add_argument('--imagePath', type=str, help='path to the .tif file that needs to be processed')
    parser.add_argument('--csModelPath', type=str, help='path to the `cspotModel` folder')
    parser.add_argument('--markerChannelMapPath', type=str, help='path to the marker panel list')
    parser.add_argument('--segmentationMaskPath', type=str, help='path to the pre-computed segmentation mask')
    parser.add_argument('--spatialTablePath', type=str, nargs='+', help='list of paths to the single-cell spatial feature tables')
    parser.add_argument('--projectDir', type=str, help='path to the output directory where the processed images (`probabilityMasks`) will be saved')
    parser.add_argument('--verbose', type=bool, default=True, help='print detailed information about the process to the console')
    
    parser.add_argument('--markerColumnName', type=str, default='marker', help='name of the column in the marker panel list that contains the marker names')
    parser.add_argument('--channelColumnName', type=str, default='channel', help='name of the column in the marker panel list that contains the channel names')
    parser.add_argument('--modelColumnName', type=str, default='cspotmodel', help='name of the column in the marker panel list that contains the model names')
    parser.add_argument('--GPU', type=int, default=-1, help='explicitly select the GPU to use')
    
    parser.add_argument('--feature', type=str, choices=['mean', 'median'], help='calculates the `mean` or `median` CSPOT Score for each cell')
    parser.add_argument('--markerNames', type=str, nargs='+', help='provide the marker names for each channel/layer in the `probabilityMaskPath`')
    
    parser.add_argument('--CellId', type=str, default='CellID', help='Specify the column name that holds the cell ID.')
    parser.add_argument('--uniqueCellId', type=bool, default=False, help='the function generates a unique name for each cell by combining the CellId and imageid')
    parser.add_argument('--split', type=str, default='X_centroid', help='Provide the column name that marks the split between expression data and meta data.')
    parser.add_argument('--removeDNA', type=bool, default=True, help='exclude DNA channels from the final output')
    parser.add_argument('--remove_string_from_name', type=str, help='cleans up channel names by removing user specified string from all marker names')
    parser.add_argument('--dropMarkers', type=list, default=None, help='Specify a list of markers to be removed from the analysis')
    parser.add_argument('--csScore', type=str, default='csScore', help='Include the label used for saving the `csScore` within the CSPOT object')
    parser.add_argument('--minAbundance', type=float, default=0.002, help='Specify the minimum percentage of cells that should express a specific marker in order to determine if the marker is considered a failure')
    parser.add_argument('--percentiles', type=list, default=[1, 20, 80, 99], help='Specify the interval of percentile levels of the expression utilized to intialize the GMM')
    parser.add_argument('--RobustScale', type=bool, default=False, help='When set to True, the data will be subject to Robust Scaling before the Gradient Boosting Classifier is trained')
    parser.add_argument('--log', type=bool, default=True, help='Apply `log1p` transformation on the data, unless it has already been log transformed in which case set it to `False`')
    parser.add_argument('--stringentThreshold', type=bool, default=False, help='Threshold to refine the classification of positive and negative cells')
    parser.add_argument('--x_coordinate', type=str, default='X_centroid', help='The column name in `single-cell spatial table` that records the X coordinates for each cell')
    parser.add_argument('--y_coordinate', type=str, default='Y_centroid', help='The column name in `single-cell spatial table` that records the Y coordinates for each cell')
    parser.add_argument('--imageid', type=str, default='imageid', help='The name of the column that holds the unique image ID')
    parser.add_argument('--random_state', type=int, default=0, help='Seed used by the random number generator')
    parser.add_argument('--rescaleMethod', type=str, default='minmax', help='Choose between `sigmoid` and `minmax`')
    parser.add_argument('--label', type=str, default='cspotOutput', help='Assign a label for the object within `adata.uns` where the predictions from CSPOT will be stored')

    
    args = parser.parse_args()
    csPipeline(imagePath=args.imagePath, 
                  csModelPath=args.csModelPath, 
                  markerChannelMapPath=args.markerChannelMapPath, 
                  segmentationMaskPath=args.segmentationMaskPath, 
                  spatialTablePath=args.spatialTablePath, 
                  projectDir=args.projectDir, 
                  verbose=args.verbose,
                  markerColumnName=args.markerColumnName, 
                  channelColumnName=args.channelColumnName, 
                  modelColumnName=args.modelColumnName, 
                  GPU=args.GPU, 
                  feature=args.feature, 
                  markerNames=args.markerNames, 
                  CellId=args.CellId, 
                  uniqueCellId=args.uniqueCellId, 
                  split=args.split, 
                  removeDNA=args.removeDNA, 
                  remove_string_from_name=args.remove_string_from_name, 
                  dropMarkers=args.dropMarkers,
                  csScore=args.csScore, 
                  minAbundance=args.minAbundance, 
                  percentiles=args.percentiles,
                  RobustScale=args.RobustScale, 
                  log=args.log, 
                  stringentThreshold=args.stringentThreshold, 
                  x_coordinate=args.x_coordinate, 
                  y_coordinate=args.y_coordinate, 
                  imageid=args.imageid, 
                  random_state=args.random_state, 
                  rescaleMethod=args.rescaleMethod, 
                  label=args.label)


