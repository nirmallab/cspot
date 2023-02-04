# -*- coding: utf-8 -*-
#Created on Thu Sep  1 16:54:39 2022
#@author: Ajit Johnson Nirmal
#Function to calculate the mean/median probability

"""
!!! abstract "Short Description"
    The `generateGatorScore` function calculates `Gator Score` for each cell by using 
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
def generateGatorScore (probabilityMaskPath,
                     segmentationMaskPath,
                     feature='median',
                     markerNames=None,
                     outputDir=None):

    """
Parameters:
    probabilityMaskPath (str):
        Supply the path of the probability map image produced by `dlModelPredict`.

    segmentationMaskPath (str):
        Supply the path of the pre-computed segmentation mask.

    feature (str, optional):
        Calculates the `mean` or `median` Gator Score for each cell.

    markerNames (list, optional):
        The program searches for marker names in the meta data (description section)
        of the tiff files created by `dlModelPredict` by default. If the meta data
        is lost due to user modifications, provide the marker names for each
        channel/layer in the `probabilityMaskPath` here.

    outputDir (str, optional):
        Provide the path to the output directory. The result will be located at
        `outputDir/GATOR/gatorScore/`.

Returns:
    CSV (dataframe):
        The `.csv` file containing the `gatorScore` is stored in the provided outputDir.

Example:

        ```python
        
        # global path
        cwd = '/Users/aj/Desktop/gatorExampleData'
        
        # function specific paths
        probabilityMaskPath = cwd + '/GATOR/gatorPredict/exampleImage_gatorPredict.ome.tif'
        segmentationPath = cwd + '/segmentation/exampleSegmentationMask.tif'
        
        ga.generateGatorScore (probabilityMaskPath=probabilityMaskPath,
                     segmentationMaskPath=segmentationPath,
                     feature='median',
                     outputDir=cwd)
        
        # Same function if the user wants to run it via Command Line Interface
        python generateGatorScore.py --probabilityMaskPath /Users/aj/Desktop/gatorExampleData/dlPredict/exampleProbabiltyMap.ome.tif --segmentationMaskPath /Users/aj/Desktop/gatorExampleData/segmentation/exampleSegmentationMask.tif --markerNames ECAD CD45 CD4 CD3D CD8A CD45R Ki67 --outputDir /Users/aj/Desktop/gatorExampleData/
        
        ```

    """


    #probabilityMask = '/Users/aj/Dropbox (Partners HealthCare)/Data/gator/data/ajn_training_data/GATOR/dlPredict/6_GatorOutput.ome.tif'
    #segmentationMaskPath = '/Users/aj/Desktop/gatorExampleData/segmentation/exampleSegmentationMask.tif'
    #outputDir = '/Users/aj/Desktop/gatorExampleData'
    #markerNames = ['ECAD', 'CD45', 'CD4', 'CD3D', 'CD8A', 'CD45_2', 'KI67']
    #probQuant (probabilityMask, segmentationMaskPath,  feature='median', markerNames=markerNames, outputDir=outputDir)

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
    print("Quantifying the probability masks")
    quantTable = pd.DataFrame(measure.regionprops_table(segM, intensity_image=probM,
                                                        properties=['label','mean_intensity'],
                                                        extra_properties=[median_intensity])).set_index('label')

    # keep only median
    if feature == 'median':
        quantTable = quantTable.filter(regex='median')
    if feature == 'mean':
        quantTable = quantTable.filter(regex='mean')

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
        quantTable.columns = channel_names
        #omexml_string = ast.literal_eval(tiff.pages[0].description)
        #channel_names = omexml_string['Channel']['Name']
    except:
        pass
    if markerNames is not None:
        channel_names = markerNames
    else:
        channel_names = list(quantTable.columns)

    # assign channel names
    quantTable.columns = channel_names

    # build a division vector
    # this is to make sure the low probs are not amplified; chosing 154 as it is 0.6P
    div_val = []
    for i in quantTable.columns:
        div_val.append(255 if quantTable[i].max() < 154 else  quantTable[i].max())

    # conver to prob
    quantTable = quantTable / div_val

    # identify markers that failed
    #sns.distplot(quantTable['ECAD'])


    # if outputDir is given
    if outputDir is None:
        outputDir = os.getcwd()

    # final path to save results
    finalPath = pathlib.Path(outputDir + '/GATOR/gatorScore/')
    if not os.path.exists(finalPath):
        os.makedirs(finalPath)

    # file name
    file_name = pathlib.Path(probabilityMaskPath).stem + '.csv'
    quantTable.to_csv(finalPath / file_name)


# Make the Function CLI compatable
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate gatorScore for a probability map and segmentation mask.')
    parser.add_argument('--probabilityMaskPath', type=str, help='Path of the probability map image produced by dlModelPredict.')
    parser.add_argument('--segmentationMaskPath', type=str, help='Path of the pre-computed segmentation mask.')
    parser.add_argument('--feature', type=str, default='median', help='Calculates the mean or median gatorScore for each cell.')
    parser.add_argument('--markerNames', nargs='+', help='List of marker names for each channel/layer in the probabilityMaskPath.')
    parser.add_argument('--outputDir', type=str, help='Path to the output directory.')
    args = parser.parse_args()
    generateGatorScore(probabilityMaskPath=args.probabilityMaskPath,
                    segmentationMaskPath=args.segmentationMaskPath,
                    feature=args.feature,
                    markerNames=args.markerNames,
                    outputDir=args.outputDir)
