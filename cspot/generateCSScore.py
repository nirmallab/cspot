# -*- coding: utf-8 -*-
#Created on Thu Sep  1 16:54:39 2022
#@author: Ajit Johnson Nirmal
#Function to calculate the mean/median probability

"""
!!! abstract "Short Description"
    The `generateCSScore` function calculates `CSPOT Score` for each cell by using 
    both the generated probability masks and pre-computed segmentation masks as inputs


## Function
"""

# Libs
import tifffile
import pandas as pd
from skimage import measure
#import ast
import xml.etree.ElementTree as ET
import numpy as np
import pathlib
import os
import argparse

# Function
def generateCSScore (probabilityMaskPath,
                         segmentationMaskPath,
                         feature='median',
                         verbose=True,
                         markerNames=None,
                         projectDir=None):

    """
Parameters:
    probabilityMaskPath (str):
        Supply the path of the probability map image produced by `dlModelPredict`.

    segmentationMaskPath (str):
        Supply the path of the pre-computed segmentation mask.

    feature (str, optional):
        Calculates the `mean` or `median` CSPOT Score for each cell.

    verbose (bool, optional):
        If True, print detailed information about the process to the console.  

    markerNames (list, optional):
        The program searches for marker names in the meta data (description section)
        of the tiff files created by `csPredict` by default. If the meta data
        is lost due to user modifications, provide the marker names for each
        channel/layer in the `probabilityMaskPath` here.

    projectDir (str, optional):
        Provide the path to the output directory. The result will be located at
        `projectDir/CSPOT/csScore/`.

Returns:
    CSV (dataframe):
        The `.csv` file containing the `csScore` is stored in the provided projectDir.

Example:
        ```python
        
        # global path
        projectDir = '/Users/aj/Documents/cspotExampleData'
        
        # Path to all the files that are necessary files for running generateCSScore
        segmentationPath = projectDir + '/segmentation/exampleSegmentationMask.tif'
        probabilityMaskPath = projectDir + '/CSPOT/csPredict/exampleImage_cspotPredict.ome.tif'
        
        cs.generateCSScore(probabilityMaskPath=probabilityMaskPath,
                      segmentationMaskPath=segmentationPath,
                      feature='median',
                      projectDir=projectDir)
        
        # Same function if the user wants to run it via Command Line Interface
        python generateCSScore.py \
            --probabilityMaskPath /Users/aj/Documents/cspotExampleData/CSPOT/csPredict/exampleImage_cspotPredict.ome.tif \
            --segmentationMaskPath /Users/aj/Documents/cspotExampleData/segmentation/exampleSegmentationMask.tif \
            --projectDir /Users/aj/Documents/cspotExampleData
        
        ```

    """
    
    # ERROS before parsing the entire image
    
    # Ensure feature is valid
    if feature not in ['median', 'mean']:
        raise ValueError("Error: Invalid feature selection. Please choose feature='median' or 'mean'.")
    
    # check for marker names
    # read the channel names from the tiffile
    tiff = tifffile.TiffFile(probabilityMaskPath)
    try:
        root = ET.fromstring(tiff.pages[0].description)
        # parse the ome XML
        namespace = None
        for elem in root.iter():
            if "Channel" in elem.tag:
                namespace = {"ome": elem.tag.split("}")[0][1:]}
                break
        channel_names = [channel.get("Name") for channel in root.findall(".//ome:Channel", namespace)]
        #omexml_string = ast.literal_eval(tiff.pages[0].description)
        #channel_names = omexml_string['Channel']['Name']
    except:
        pass
    
    # check if channel_names has been defined
    if markerNames is not None:
        channel_names = markerNames
    else:
        #quantTable.columns = channel_names
        channel_names = list(channel_names)
        # Check if channel_names is empty or contains NaN/None values
        if not channel_names or any(x is None or (isinstance(x, float) and np.isnan(x)) for x in channel_names):
            raise ValueError(
                "Error: Unable to identify marker names from the image. "
                "Please manually pass markerNames.")


    # NOW PARSE ENTIRE IMAGE

    # read the seg mask
    segM = tifffile.imread(pathlib.Path(segmentationMaskPath))
    probM = tifffile.imread(pathlib.Path(probabilityMaskPath))

    #probs = []
    #for i in range(len(probM)):
    #    pospix = len(probM[i][(probM[i] / 255) > 0.5]) / (probM[i].shape[0] * probM[i].shape[1])
    #    probs.append(pospix)

    if len(probM.shape) > 2:
        probM = np.moveaxis(probM, 0, -1)

    def median_intensity(mask, img):
        return np.median(img[mask])
    

    # quantify
    if verbose is True:
        print("Quantifying the probability masks")
    quantTable = pd.DataFrame(measure.regionprops_table(segM, intensity_image=probM,
                                                        properties=['label','mean_intensity'],
                                                        extra_properties=[median_intensity])).set_index('label')

    # keep only median
    if feature == 'median':
        quantTable = quantTable.filter(regex='median')
    if feature == 'mean':
        quantTable = quantTable.filter(regex='mean')


    # assign channel names
    quantTable.columns = channel_names

    # build a division vector
    # this is to make sure the low probs are not amplified; chosing 154 as it is 0.6P
    #div_val = []
    #for i in quantTable.columns:
    #    div_val.append(255 if quantTable[i].max() < 154 else  quantTable[i].max())

    # conver to prob
    #quantTable = quantTable / div_val
    quantTable = quantTable / 255

    # identify markers that failed
    #sns.distplot(quantTable['ECAD'])


    # if projectDir is given
    if projectDir is None:
        projectDir = os.getcwd()

    # final path to save results
    finalPath = pathlib.Path(projectDir + '/CSPOT/csScore/')
    if not os.path.exists(finalPath):
        os.makedirs(finalPath)

    # file name
    file_name = pathlib.Path(probabilityMaskPath).stem + '.csv'
    quantTable.to_csv(finalPath / file_name)
    
    # Finish Job
    if verbose is True:
        print('csScore is ready, head over to' + str(projectDir) + '/CSPOT/csScore" to view results')


# Make the Function CLI compatable
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate csScore for a probability map and segmentation mask.')
    parser.add_argument('--probabilityMaskPath', type=str, help='Path of the probability map image produced by dlModelPredict.')
    parser.add_argument('--segmentationMaskPath', type=str, help='Path of the pre-computed segmentation mask.')
    parser.add_argument('--feature', type=str, default='median', help='Calculates the mean or median csScore for each cell.')
    parser.add_argument("--verbose", type=bool, default=True, help="If True, print detailed information about the process to the console.")       
    parser.add_argument('--markerNames', nargs='+', help='List of marker names for each channel/layer in the probabilityMaskPath.')
    parser.add_argument('--projectDir', type=str, help='Path to the output directory.')
    args = parser.parse_args()
    generateCSScore(probabilityMaskPath=args.probabilityMaskPath,
                    segmentationMaskPath=args.segmentationMaskPath,
                    feature=args.feature,
                    verbose=args.verbose,
                    markerNames=args.markerNames,
                    projectDir=args.projectDir)
