#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Tue Aug 30 08:28:48 2022
#@author: Ajit Johnson Nirmal
#Generation of Thumbnails/ training data for a given marker

"""
!!! abstract "Short Description"
    The `generateThumbnails` function generates Thumbnails of positive and 
    negative cells for a specified marker. The Thumbnails will be used to train a deep learning model. Make sure to have 
    the raw image, computed single-cell spatial table, and markers.csv file 
    ready for input.


## Function
"""


# import packages
import numpy as np
import pandas as pd
import random
import tifffile
import os
import pathlib
import dask.array as da
import argparse
from sklearn.mixture import GaussianMixture
from collections import defaultdict
from sklearn.neighbors import BallTree
#import zarr

# Function
def generateThumbnails (spatialTablePath, 
                        imagePath, 
                        markerChannelMapPath,
                        markers, 
                        markerColumnName='marker',
                        channelColumnName='channel',
                        transformation=True, 
                        maxThumbnails=2000, 
                        random_state=0,
                        localNorm=True, 
                        globalNorm=False,
                        x_coordinate='X_centroid', 
                        y_coordinate='Y_centroid',
                        percentiles=[2, 12, 88, 98], 
                        windowSize=64,
                        restrictDensity=True,
                        restrictDensityNumber=None,
                        verbose=True,
                        projectDir=None):
    """
Parameters:

    spatialTablePath (str):
        Path to the single-cell spatial feature matrix.

    imagePath (str):
        Path to the image file. Recognizes `.ome.tif` image file.
    
    markerChannelMapPath (str):
        Path to a `markers.csv` file that maps the channel number with the marker information. 
        Create a .csv file with at least two columns named 'channel' and 'marker' that 
        map the channel numbers to their corresponding markers. The channel number 
        should use 1-based indexing.
    
    markers (list):
        Markers for which `Thumbnails` need to be generated. The function looks for
        these listed names in the `single-cell spatial Table`.

    markerColumnName (str):
        The name of the column in the `markers.csv` file that holds the marker information. 
    
    channelColumnName (str):
        The name of the column in the `markers.csv` file that holds the channel information.  

    transformation (bool, optional):
        Performs `arcsinh` transformation on the data. If the `single-cell spatial table`
        is already transformed (like log transformation), set this to `False`.

    maxThumbnails (int, optional):
        Maximum number of Thumbnails to generate. 

    random_state (int, optional):
        Seed used by the random number generator.

    localNorm (bool, optional):
        It creates a duplicate folder of the Thumbnails, with local normalization
        performed on the images. Local normalization is the process of dividing
        each pixel in a thumbnail by the maximum value across the entire thumbnail.
        This is helpful for visual supervised sorting of the Thumbnails.

    globalNorm (bool, optional):
        It creates a duplicate folder of the Thumbnails, with global normalization
        performed on the images. Global normalization is the process of dividing
        each pixel in a thumbnail by the maximum value of the given marker across
        the entire image.

    x_coordinate (str, optional):
        The column name in `single-cell spatial table` that records the
        X coordinates for each cell.

    y_coordinate (str, optional):
        The column name in `single-cell spatial table` that records the
        Y coordinates for each cell.

    percentiles (list, optional):
        Specify the interval of percentile levels of the expression utilized to intialize
        the GMM. The cells falling within these percentiles are utilized to distinguish
        between negative cells (first two values) and positive cells (last two values).

    windowSize (int, optional):
        Size of the Thumbnails.

    restrictDensity (bool, optional):
        This parameter is utilized to regulate the number of positive cells 
        observed in a given field of view. In the case of markers that do not 
        exhibit a distinct spatial pattern, such as immune cells, it is 
        recommended to train the model using sparse cells in the field of view.

    restrictDensityNumber (int, optional):
        This parameter is employed in conjunction with `restrictDensity`. 
        By default, the program attempts to automatically identify less dense 
        regions when restrictDensity is set to `True` using a GMM approach. 
        However, `restrictDensityNumber` can be utilized to exert greater 
        control over the process, allowing the user to limit the number of 
        positive cells they wish to observe within the field of view. 
        This parameter requires integers.

    verbose (bool, optional):
        If True, print detailed information about the process to the console.  
        
    projectDir (string, optional):
        Path to output directory. The result will be located at
        `projectDir/GATOR/Thumbnails/`.

Returns:
    Thumbnails (image):
        Saves Thumbnails of auto identified postive and negative cells the
        designated output directory.

Example:
        
        ```python
        
        # set the working directory & set paths to the example data
        cwd = '/Users/aj/Desktop/gatorExampleData'
        imagePath = cwd + '/image/exampleImage.tif'
        spatialTablePath = cwd + '/quantification/exampleSpatialTable.csv'
        markerChannelMapPath = cwd + '/markers.csv'
        
        # Run the function
        ga.generateThumbnails ( spatialTablePath=spatialTablePath, 
                        imagePath=imagePath, 
                        markerChannelMapPath=markerChannelMapPath,
                        markers=["ECAD", "CD3D"], 
                        markerColumnName='marker',
                        channelColumnName='channel',
                        transformation=True, 
                        maxThumbnails=100, 
                        random_state=0,
                        localNorm=True, 
                        globalNorm=False,
                        x_coordinate='X_centroid', 
                        y_coordinate='Y_centroid',
                        percentiles=[2, 12, 88, 98], 
                        windowSize=64,
                        restrictDensity=True,
                        restrictDensityNumber=None,
                        verbose=True,
                        projectDir=cwd)
        
        # Same function if the user wants to run it via Command Line Interface
        python generateThumbnails.py --spatialTablePath /Users/aj/Desktop/gatorExampleData/quantification/exampleSpatialTable.csv --imagePath /Users/aj/Desktop/gatorExampleData/image/exampleImage.tif --markerChannelMapPath /Users/aj/Desktop/gatorExampleData/markers.csv --markers ECAD CD3D --maxThumbnails 100 --projectDir /Users/aj/Desktop/gatorExampleData/
        
        ```
    """

    # transformation=True; maxThumbnails=100; x_coordinate='X_centroid'; y_coordinate='Y_centroid'; percentiles=[2, 12, 88, 98]; windowSize=64; random_state=0; localNorm=True; globalNorm=False; markerColumnName='marker'; channelColumnName='channel';projectDir=cwd; restrictDensity=True;restrictDensityNumber=None;verbose=True;
    # markers = ['CD3D', 'CD8A']
    # marker = 'CD3D';
    # imagePath = '/Users/aj/Desktop/gatorExampleData/image/exampleImage.tif'
    # spatialTablePath = '/Users/aj/Desktop/gatorExampleData/quantification/exampleSpatialTable.csv'
    # markerChannelMapPath = '/Users/aj/Desktop/gatorExampleData/markers.csv'
    # projectDir = '/Users/aj/Desktop/gatorExampleData/'
    
    # read the markers.csv
    maper = pd.read_csv(pathlib.Path(markerChannelMapPath))
    columnnames =  [word.lower() for word in maper.columns]
    maper.columns = columnnames
    
    # identify the marker column name (doing this to make it easier for people who confuse between marker and markers)
    if markerColumnName not in columnnames:
        if markerColumnName != 'marker':
            raise ValueError('markerColumnName not found in markerChannelMap, please check')
        if 'markers' in columnnames:
            markerCol = 'markers'
        else:
            raise ValueError('markerColumnName not found in markerChannelMap, please check')
    else: 
        markerCol = markerColumnName
    
    # identify the channel column name (doing this to make it easier for people who confuse between channel and channels)
    if channelColumnName not in columnnames:
        if channelColumnName != 'channel':
            raise ValueError('channelColumnName not found in markerChannelMap, please check')
        if 'channels' in columnnames:
            channelCol = 'channels'
        else:
            raise ValueError('channelColumnName not found in markerChannelMap, please check')
    else: 
        channelCol = channelColumnName
    
    # map the marker and channels
    chmamap = dict(zip(maper[markerCol], maper[channelCol]))
    
    # load the CSV to identify potential thumbnails
    data = pd.read_csv(pathlib.Path(spatialTablePath))
    #data.index = data.index.astype(str)

    # subset the markers of interest
    if isinstance (markers, str):
        markers = [markers]
    
    # find the corresponding channel names
    markerChannels = [chmamap[key] for key in markers if key in chmamap]
    # convert markerChannels to zero indexing
    markerChannels = [x-1 for x in markerChannels]

    # creat a dict of marker and corresponding marker channel
    marker_map = dict(zip(markers,markerChannels))

    # create folders if it does not exist
    if projectDir is None:
        projectDir = os.getcwd()

    # TruePos folders
    for i in markers:
        pos_path = pathlib.Path(projectDir + '/GATOR/Thumbnails/' + str(i) + '/TruePos')
        neg_path = pathlib.Path(projectDir + '/GATOR/Thumbnails/' + str(i) + '/TrueNeg')
        pos2neg_path = pathlib.Path(projectDir + '/GATOR/Thumbnails/' + str(i) + '/PosToNeg')
        neg2pos_path = pathlib.Path(projectDir + '/GATOR/Thumbnails/' + str(i) + '/NegToPos')
        if not os.path.exists(pos_path):
            os.makedirs(pos_path)
        if not os.path.exists(neg_path):
            os.makedirs(neg_path)
        if not os.path.exists(pos2neg_path):
            os.makedirs(pos2neg_path)
        if not os.path.exists(neg2pos_path):
            os.makedirs(neg2pos_path)
        if localNorm is True:
            local_pos_path = pathlib.Path(projectDir + '/GATOR/Thumbnails/localNorm/' + str(i) + '/TruePos')
            local_neg_path = pathlib.Path(projectDir + '/GATOR/Thumbnails/localNorm/' + str(i) + '/TrueNeg')
            local_pos2neg_path = pathlib.Path(projectDir + '/GATOR/Thumbnails/localNorm/' + str(i) + '/PosToNeg')
            local_neg2pos_path = pathlib.Path(projectDir + '/GATOR/Thumbnails/localNorm/' + str(i) + '/NegToPos')
            if not os.path.exists(local_pos_path):
                os.makedirs(local_pos_path)
            if not os.path.exists(local_neg_path):
                os.makedirs(local_neg_path)
            if not os.path.exists(local_pos2neg_path):
                os.makedirs(local_pos2neg_path)
            if not os.path.exists(local_neg2pos_path):
                os.makedirs(local_neg2pos_path)

    marker_data = data[markers]
    location = data[[x_coordinate,y_coordinate]]

    # clip the data to drop outliers
    def clipping (x):
        clip = x.clip(lower =np.percentile(x,0.01), upper=np.percentile(x,99.99)).tolist()
        return clip
    
    # return the mean between two percentiles
    def meanPercentile (values, lowPercentile, highPercentile):
        # Calculate the 1st percentile value
        p1 = np.percentile(values, lowPercentile)
        # Calculate the 20th percentile value
        p20 = np.percentile(values, highPercentile)
        # Select the values between the 1st and 20th percentile using numpy.where()
        filtered_values = np.where((values >= p1) & (values <= p20))
        # Calculate the mean of the filtered values
        meanVal = np.mean(values[filtered_values])
        return meanVal

    # Function for GMM
    def simpleGMM (data, n_components, means_init, random_state):
        gmm = GaussianMixture(n_components=n_components, means_init=means_init,  random_state=random_state)
        gmm.fit(data)
        # Predict the class labels for each sample
        predictions = gmm.predict(data)
        # Get the mean of each Gaussian component
        means = gmm.means_.flatten()
        # Sort the mean values in ascending order
        sorted_means = np.sort(means)
        # Assign 'pos' to rows with higher mean distribution and 'neg' to rows with lower mean distribution
        labels = np.where(predictions == np.argmax(means), 'pos', 'neg')
        return labels
        
    # match two arrays and return seperate lists
    def array_match (labels, names):
        # Create a defaultdict with an empty list as the default value
        result = defaultdict(list)
        # Iterate over the labels and names arrays
        for label, name in zip(labels, names):
            # Add the name to the list for the corresponding label
            result[label].append(name)
        return result
        
        
    
    # clip data
    marker_data = marker_data.apply(clipping)

    # apply transformation if requested
    if transformation is True:
        marker_data = np.arcsinh(marker_data)
        #marker_data = np.log1p(marker_data)
        
    # combine data
    combined_data = pd.concat([marker_data, location], axis=1)

    # intialize the percentiles values
    percentiles.sort()

    # function to identify the corner of the thumbnails
    def cornerFinder (centroid):
        row_start = int(centroid - windowSize // 2)
        row_end = row_start + windowSize
        return [row_start, row_end]

    # function to crop the image and save the image
    def cropImage (rowIndex, corners, imgType, zimg, npercentile, m, maxpercentile, imname):
        #print(str(rowIndex))
        x_start = corners.loc[rowIndex]['x_start']; x_end = corners.loc[rowIndex]['x_end']
        y_start = corners.loc[rowIndex]['y_start']; y_end = corners.loc[rowIndex]['y_end']
        # cropping image
        crop = zimg[y_start:y_end, x_start:x_end]
        # convert the image to unit8
        if globalNorm is True:
            fullN = ((crop/npercentile)*255).clip(0, 255).astype('uint8')
        else:
            fullN = ((crop/maxpercentile)*255).clip(0, 255).astype('uint8')
        # save the cropped image
        if imgType == 'pos':
            path = pathlib.Path(projectDir + '/GATOR/Thumbnails/' + str(m) + '/TruePos/' + str(rowIndex) + "_" + str(imname) + '.tif')
        elif imgType == 'neg':
            path = pathlib.Path(projectDir + '/GATOR/Thumbnails/' + str(m) + '/TrueNeg/' + str(rowIndex) + "_" + str(imname) + '.tif')
        # write file
        tifffile.imwrite(path,fullN)
        # local normalization if requested
        if localNorm is True:
            localN = ((crop/(np.percentile(crop.compute(), 99.99)))*255).clip(0, 255).astype('uint8')
            # save image
            if imgType == 'pos':
                Lpath = pathlib.Path(projectDir + '/GATOR/Thumbnails/localNorm/' + str(m) + '/TruePos/' + str(rowIndex) + "_" + str(imname) + '.tif')
            elif imgType == 'neg':
                Lpath = pathlib.Path(projectDir + '/GATOR/Thumbnails/localNorm/' + str(m) + '/TrueNeg/' + str(rowIndex) + "_" + str(imname) + '.tif')
            # write file
            tifffile.imwrite(Lpath,localN)

    # identify the cells of interest
    def processMarker (marker):
        if verbose is True:
            print('Processing Marker: ' + str(marker))

        moi = combined_data[marker].values

        # figure out marker index or channel in image
        markerIndex = marker_map[marker]
        
        # mean of cells within defined threshold
        lowerPercent = meanPercentile (values=moi, lowPercentile=percentiles[0], highPercentile=percentiles[1])
        higherPercent = meanPercentile (values=moi, lowPercentile=percentiles[2], highPercentile=percentiles[3])
        # Format mean to pass into next GMM
        Pmean = np.array([[lowerPercent], [higherPercent]])
        
        # perform GMM
        labels = simpleGMM (data=moi.reshape(-1, 1), n_components=2, means_init=Pmean,  random_state=random_state)
        # Match the labels and index names to identify which cells are pos and neg
        expCells = array_match (labels=labels, names=data.index)
        # split it
        pos = expCells.get('pos', []) ; neg = expCells.get('neg', [])
            
            
        # determine the percentiles value for the marker of interest
        #low_a = np.percentile(moi, percentiles[0]); low_b = np.percentile(moi, percentiles[1])
        #high_a = np.percentile(moi, percentiles[2]); high_b = np.percentile(moi, percentiles[3])

        # identify the cells that fall within the determined range
        #neg = np.where(moi.between(low_a, low_b))[0]
        #pos = np.where(moi.between(high_a, high_b))[0]

        # shuffle the cells
        random.Random(random_state).shuffle(neg); random.Random(random_state).shuffle(pos)

        # identify the location of pos and neg cells
        neg_location_i = location.iloc[neg]
        pos_location_i = location.iloc[pos]
    
        # divide the pos cells into two bins based on the number of neighbours
        # assumption is that we will find cells that are dense and sparse
        if restrictDensity is True:
            # identify postive cells that are densly packed if user requests
            kdt = BallTree(pos_location_i[[x_coordinate,y_coordinate]], metric='euclidean') 
            ind = kdt.query_radius(pos_location_i[[x_coordinate,y_coordinate]], r=windowSize+5, return_distance=False)
            for i in range(0, len(ind)): ind[i] = np.delete(ind[i], np.argwhere(ind[i] == i))#remove self
            neigh_length = [len(subarray) for subarray in ind]
            if restrictDensityNumber is None:
                # GMM for auto detection
                X = np.array(neigh_length).reshape(-1, 1)
                gmm_neigh = GaussianMixture(n_components=2)
                gmm_neigh.fit(X)
                means = gmm_neigh.means_
                index = np.argmin(means)
                labels_neigh = gmm_neigh.predict(X)
                lower_mean_indices = np.where(labels_neigh == index)[0]
                # subset the postive cells based on the index
                pos_location_i = pos_location_i.iloc[lower_mean_indices]
            else:
                lower_mean_indices = [i for i, x in enumerate(neigh_length) if x < restrictDensityNumber]
                pos_location_i = pos_location_i.iloc[lower_mean_indices]

        # Find corner
        # Negative cells
        r_cornerFinder = lambda x: cornerFinder (centroid=x)
        neg_x = pd.DataFrame(list(map(r_cornerFinder, neg_location_i[x_coordinate].values))) # x direction
        neg_y = pd.DataFrame(list(map(r_cornerFinder, neg_location_i[y_coordinate].values))) # y direction
        neg_x.columns = ["x_start", "x_end"]; neg_y.columns = ["y_start", "y_end"]
        neg_location = pd.concat([neg_x, neg_y], axis=1)
        neg_location.index = neg_location_i.index

        # Positive cells
        r_cornerFinder = lambda x: cornerFinder (centroid=x)
        pos_x = pd.DataFrame(list(map(r_cornerFinder, pos_location_i[x_coordinate].values))) # x direction
        pos_y = pd.DataFrame(list(map(r_cornerFinder, pos_location_i[y_coordinate].values))) # y direction
        pos_x.columns = ["x_start", "x_end"]; pos_y.columns = ["y_start", "y_end"]
        pos_location = pd.concat([pos_x, pos_y], axis=1)
        pos_location.index = pos_location_i.index

        # drop all coordinates with neg values (essentially edges of slide)
        neg_location = neg_location[(neg_location > 0).all(1)]
        pos_location = pos_location[(pos_location > 0).all(1)]

        # subset max number of cells
        if len(neg_location) > maxThumbnails:
            neg_location = neg_location[:maxThumbnails]
        if len(pos_location) > maxThumbnails:
            pos_location = pos_location[:maxThumbnails]

        # identify image name
        imname = pathlib.Path(imagePath).stem

        # load the image
        zimg = da.from_zarr(tifffile.imread(pathlib.Path(imagePath), aszarr=True, level=0, key=markerIndex))
        npercentile = np.percentile(zimg.compute(), 99.99)
        maxpercentile = zimg.max().compute()

        # for older version f tifffile
        #zimg = zarr.open(tifffile.imread(pathlib.Path(imagePath), aszarr=True, level=0, key=markerIndex))
        # = np.percentile(zimg,  99.99)

        # Cut images and write it out
        # neg
        r_cropImage = lambda x: cropImage (rowIndex=x, corners=neg_location, imgType='neg', zimg=zimg, npercentile=npercentile, maxpercentile=maxpercentile, m=marker, imname=imname)
        process_neg = list(map(r_cropImage, list(neg_location.index)))
        # pos
        r_cropImage = lambda x: cropImage (rowIndex=x, corners=pos_location, imgType='pos', zimg=zimg, npercentile=npercentile, maxpercentile=maxpercentile, m=marker, imname=imname)
        process_neg = list(map(r_cropImage, list(pos_location.index)))


    # Run the function for each marker
    r_processMarker = lambda x: processMarker (marker=x)
    final = list(map(r_processMarker, markers))

    # Finish Job
    if verbose is True:
        print('Thumbnails have been generated, head over to "' + str(projectDir) + '/GATOR/Thumbnails" to view results')

# Make the Function CLI compatable
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Thumbnails for single-cell spatial feature matrix and image file")
    parser.add_argument("--spatialTablePath", type=str, help="Path to the single-cell spatial feature matrix.")
    parser.add_argument("--imagePath", type=str, help="Path to the image file. Recognizes `.ome.tif` image file.")
    parser.add_argument("--markerChannelMapPath", type=str, help="Path to the marker channel mapping file.")
    parser.add_argument("--markers", type=str, nargs='+', help="Markers for which `Thumbnails` need to be generated. The function looks for these listed names in the `single-cell spatial feature matrix`.")
    parser.add_argument("--markerColumnName", type=str, default='marker', help="The name of the marker column.")
    parser.add_argument("--channelColumnName", type=str, default='channel', help="The name of the channel column.")
    parser.add_argument("--transformation", type=bool, default=True, help="Performs `arcsinh` transformation on the data. If the `single-cell spatial table` is already transformed (like log transformation), set this to `False`.")
    parser.add_argument("--maxThumbnails", type=int, default=2000, help="Maximum number of Thumbnails to generate.")
    parser.add_argument("--random_state", type=int, default=0, help="Seed used by the random number generator.")
    parser.add_argument("--localNorm", type=bool, default=True, help="It creates a duplicate folder of the Thumbnails, with local normalization performed on the images. Local normalization is the process of dividing each pixel in a thumbnail by the maximum value across the entire thumbnail. This is helpful for visual supervised sorting of the Thumbnails.")
    parser.add_argument("--globalNorm", type=bool, default=False, help="It creates a duplicate folder of the Thumbnails, with global normalization performed on the images. Global normalization is the process of dividing each pixel in a thumbnail by the maximum value of the given marker across the entire image.")
    parser.add_argument("--x_coordinate", type=str, default='X_centroid', help="The column name in `single-cell spatial table` that records the X coordinates for each cell. The default is 'X_centroid'.")
    parser.add_argument("--y_coordinate", type=str, default='Y_centroid', help="The column name in `single-cell spatial table` that records the Y coordinates for each cell. The default is 'Y_centroid'.")
    parser.add_argument("--percentiles", type=int, nargs='+', default=[2, 12, 88, 98], help="Specify the interval of percentile levels of the expression utilized to intialize the GMM. The cells falling within these percentiles are utilized to distinguish between negative cells (first two values) and positive cells (last two values).")
    parser.add_argument("--windowSize", type=int, default=64, help="Size of the Thumbnails.")
    parser.add_argument("--restrictDensity", type=bool, default=True, help="This parameter is utilized to regulate the number of positive cells observed in a given field of view. In the case of markers that do not exhibit a distinct spatial pattern, such as immune cells, it is recommended to train the model using sparse cells in the field of view.")
    parser.add_argument("--restrictDensityNumber", type=int, default=None, help="This parameter is employed in conjunction with `restrictDensity`. By default, the program attempts to automatically identify less dense regions when restrictDensity is set to `True` using a GMM approach. However, `restrictDensityNumber` can be utilized to exert greater control over the process, allowing the user to limit the number of positive cells they wish to observe within the field of view. This parameter requires integers.")
    parser.add_argument("--verbose", type=bool, default=True, help="If True, print detailed information about the process to the console.")    
    parser.add_argument("--projectDir", type=str, default=None, help="Path to output directory. The result will be located at `projectDir/GATOR/Thumbnails/`.")
    args = parser.parse_args()
    generateThumbnails(spatialTablePath=args.spatialTablePath,
                        imagePath=args.imagePath,
                        markerChannelMapPath=args.markerChannelMapPath,
                        markers=args.markers,
                        markerColumnName=args.markerColumnName,
                        channelColumnName=args.channelColumnName,
                        transformation=args.transformation,
                        maxThumbnails=args.maxThumbnails,
                        random_state=args.random_state,
                        localNorm=args.localNorm,
                        globalNorm=args.globalNorm,
                        x_coordinate=args.x_coordinate,
                        y_coordinate=args.y_coordinate,
                        percentiles=args.percentiles,
                        windowSize=args.windowSize,
                        restrictDensity=args.restrictDensity,
                        restrictDensityNumber=args.restrictDensityNumber,
                        verbose=args.verbose,
                        projectDir=args.projectDir)
