#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Sat Dec 17 11:56:32 2022
#@author: Ajit Johnson Nirmal
#Gator Algorithm

"""
!!! abstract "Short Description"
    The gator function identifies positive and negative cells for a marker. To 
    get optimal results, consider adjusting the following parameters:  
        
    1. The `gatorObject` parameter can accept either the loaded gatorObject or a path to the `.h5ad` file.  
    2. The `minAbundance` parameter determines the minimum percentage of a marker's abundance to consider it a failure.  
    3. It is suggested to drop background markers with the `dropMarkers` option as they can interfere with classifiers.  
    4. `RobustScale`: Scaling the data before training the classifier model has been shown to improve results. 
    However, in our experience a simple log transformation was found to be sufficient.   

## Function
"""

import pandas as pd
import anndata as ad
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier
import random
import numpy as np
from sklearn.mixture import GaussianMixture
from collections import defaultdict
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import pathlib
import os
import argparse


def gator (gatorObject,
           gatorScore='gatorScore',
           minAbundance=0.005,
           percentiles=[1, 20, 80, 99],
           dropMarkers = None,
           RobustScale=False,
           log=True,
           stringentThreshold=False,
           x_coordinate='X_centroid',
           y_coordinate='Y_centroid',
           imageid='imageid',
           random_state=0,
           rescaleMethod='minmax',
           label='gatorOutput',
           verbose=True,
           projectDir=None, **kwargs):
    """
Parameters:
    gatorObject (anndata):
        Pass the `gatorObject` loaded into memory or a path to the `gatorObject` 
        file (.h5ad).

    gatorScore (str, optional):
        Include the label used for saving the `gatorScore` within the Gator object.

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
        negative cells by utilizing gatorScores. The stringentThreshold can be utilized 
        to further refine the classification of positive and negative cells. 
        By setting it to True, cells with gatorScore below the mean of the negative 
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
        from Gator will be stored. 

    verbose (bool, optional):
        If True, print detailed information about the process to the console.  

    projectDir (str, optional):
        Provide the path to the output directory. The result will be located at
        `projectDir/GATOR/gatorOutput/`. 

    **kwargs (keyword parameters):
        Additional arguments to pass to the `HistGradientBoostingClassifier()` function.

Returns:
    gatorObject (anndata):
        If projectDir is provided the updated Gator Object will saved within the
        provided projectDir.

Example:

        ```python
        
        # set the working directory & set paths to the example data
        cwd = '/Users/aj/Desktop/gatorExampleData'
        gatorObject = cwd + '/GATOR/gatorObject/exampleImage_gatorPredict.ome.h5ad'
        
        # Run the function
        adata = ga.gator ( gatorObject=gatorObject,
                    gatorScore='gatorScore',
                    minAbundance=0.002,
                    percentiles=[1, 20, 80, 99],
                    dropMarkers = None,
                    RobustScale=False,
                    log=True,
                    x_coordinate='X_centroid',
                    y_coordinate='Y_centroid',
                    imageid='imageid',
                    random_state=0,
                    rescaleMethod='sigmoid',
                    label='gatorOutput',
                    verbose=True,
                    projectDir=cwd)
        
        # Same function if the user wants to run it via Command Line Interface
        python gator.py --gatorObject /Users/aj/Desktop/gatorExampleData/GATOR/gatorObject/exampleImage_gatorPredict.ome.h5ad --projectDir /Users/aj/Desktop/gatorExampleData
                
        
        ```

    """

    #gatorObject = "/Users/aj/Dropbox (Partners HealthCare)/Data/gator/data/ajn_training_data/GATOR/gatorObject/1_6_GatorOutput.h5ad"
    #gatorScore='gatorScore'; minAbundance=0.002; percentiles=[1, 20, 80, 99]; dropMarkers = None;leakData=False;rescaleMethod='sigmoid';verbose = False
    #scaleData=False; log=True; x_coordinate='X_centroid'; y_coordinate='Y_centroid'; imageid='imageid'; random_state=0; label='gatorOutput'
    #gatorObject = '/Users/aj/Dropbox (Partners HealthCare)/Data/gator/data/ajn_training_data/GATOR/gatorPredict/2_28_GatorOutput.h5ad'
    #gatorObject = '/Users/aj/Dropbox (Partners HealthCare)/Data/gator/data/ajn_training_data/GATOR/gatorPredict/4_113_GatorOutput.h5ad'


    # Load the andata object
    if isinstance(gatorObject, str):
        adata = ad.read(gatorObject)
        gatorObject = [gatorObject]
        gatorObjectPath = [pathlib.Path(p) for p in gatorObject]
    else:
        adata = gatorObject.copy()

    # break the function if gatorScore is not detectable
    def check_key_exists(dictionary, key):
        try:
            # Check if the key exists in the dictionary
            value = dictionary[key]
        except KeyError:
            # Return an error if the key does not exist
            return "Error: " + str(gatorScore) + " does not exist, please check!"
    # Test
    check_key_exists(dictionary=adata.uns, key=gatorScore)


    ###########################################################################
    # SOME GENERIC FUNCTIONS
    ###########################################################################

    # used in (step 1)
    def get_columns_with_low_values(df, minAbundance):
        columns_to_keep = []
        for column in df.columns:
            num_rows_with_high_values = len(df[df[column] > 0.6])
            if num_rows_with_high_values / len(df) < minAbundance:
                columns_to_keep.append(column)
        return columns_to_keep
    
    # count the number of pos and neg elements in a list
    def count_pos_neg(lst):
        arr = np.array(lst)
        result = {'pos': np.sum(arr == 'pos'), 'neg': np.sum(arr == 'neg')}
        result['pos'] = result['pos'] if result['pos'] > 0 else 0
        result['neg'] = result['neg'] if result['neg'] > 0 else 0
        return result
    
    # alternative to find if markers failed
    def simpleGMM_failedMarkers (df, n_components, minAbundance, random_state):
        # prepare data
        columns_to_keep = []
        for column in df.columns:
            #print(str(column))
            colValue = df[[column]].values  
            colValue[0] = 0; colValue[1] = 1;   # force the model to converge from 0-1
            gmm = GaussianMixture(n_components=n_components,  random_state=random_state)
            gmm.fit(colValue)
            predictions = gmm.predict(colValue)
            # Get the mean of each Gaussian component
            means = gmm.means_.flatten()
            # Sort the mean values in ascending order
            sorted_means = np.sort(means)
            # Assign 'pos' to rows with higher mean distribution and 'neg' to rows with lower mean distribution
            labels = np.where(predictions == np.argmax(means), 'pos', 'neg')
            # count pos and neg
            counts = count_pos_neg(labels)
            # find if the postive cells is less than the user defined min abundance
            if counts['pos'] / len(df)  < minAbundance:
                columns_to_keep.append(column)
        return columns_to_keep


    # preprocess data (step-4)
    def pre_process (data, log=log):
        # clip outliers
        def clipping (x):
            clip = x.clip(lower =np.percentile(x,0.01), upper=np.percentile(x,99.99)).tolist()
            return clip
        processsed_data = data.apply(clipping)
        if log is True:
            processsed_data = np.log1p(processsed_data)
        return processsed_data
    # preprocess data (step-5)
    def apply_transformation (data):
        # rescale the data
        transformer = RobustScaler().fit(data)
        processsed_data = pd.DataFrame(transformer.transform(data), columns = data.columns, index=data.index)
        return processsed_data

    # GMM
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

        return labels, sorted_means 

    # take in two list (ccategorical and numerical) and return mean values
    def array_mean (labels, values):
        # Create a defaultdict with an empty list as the default value
        result = defaultdict(list)
        # Iterate over the labels and values arrays
        for label, value in zip(labels, values):
            # Add the value to the list for the corresponding label
            result[label].append(value)
        # Calculate the mean for each label and store it in the dictionary
        for label, value_list in result.items():
            result[label] = np.mean(value_list)
        return result

    # match two arrays and return seperate lists
    def array_match (labels, names):
        # Create a defaultdict with an empty list as the default value
        result = defaultdict(list)
        # Iterate over the labels and names arrays
        for label, name in zip(labels, names):
            # Add the name to the list for the corresponding label
            result[label].append(name)
        return result

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

    # return the mean between two percentiles
    def indexPercentile (processed_data, marker, lowPercentile, highPercentile):
        values = processed_data[marker].values
        # Calculate the 1st percentile value
        p1 = np.percentile(values, lowPercentile)
        # Calculate the 20th percentile value
        p20 = np.percentile(values, highPercentile)
        # Select the values between the 1st and 20th percentile using numpy.where()
        filtered_values = np.where((values >= p1) & (values <= p20))
        # Calculate the mean of the filtered values
        idx = processed_data[marker].iloc[filtered_values].index
        return idx

    # Used for rescaling data
    # used to find the mid point of GMM mixtures
    def find_midpoint(data, labels):
      # Convert data and labels to NumPy arrays
      data = np.array(data)
      labels = np.array(labels)
      # Get the indices that would sort the data and labels arrays
      sort_indices = np.argsort(data)
      # Sort the data and labels arrays using the sort indices
      sorted_data = data[sort_indices]
      sorted_labels = labels[sort_indices]
      # Find the index where the 'neg' and 'pos' labels meet
      midpoint_index = np.argmax(sorted_labels == 'pos')
      # Return the value at the midpoint index
      return sorted_data[midpoint_index]

    # Used for reassigning some of the wrong 'nes' and 'pos' within data given a midpoint
    def modify_negatives_vectorized(data, labels, midpoint):
      # Convert data and labels to NumPy arrays
      data = np.array(data)
      labels = np.array(labels)
      # Get the indices that would sort the data and labels arrays
      sort_indices = np.argsort(data)
      # Sort the data and labels arrays using the sort indices
      sorted_data = data[sort_indices]
      sorted_labels = labels[sort_indices]
      # Find the index where the sorted data is greater than or equal to the midpoint value
      midpoint_index = np.argmax(sorted_data >= midpoint)
      # Find all the elements in the sorted labels array with a value of 'neg' after the midpoint index
      neg_mask = np.logical_and(sorted_labels == 'neg', np.arange(len(sorted_data)) >= midpoint_index)
      # Modify the value of the elements to be equal to the midpoint value
      sorted_data[neg_mask] = midpoint
      # Find all the elements in the sorted labels array with a value of 'pos' before the midpoint index
      pos_mask = np.logical_and(sorted_labels == 'pos', np.arange(len(sorted_data)) < midpoint_index)
      # Modify the value of the elements to be equal to the midpoint value plus 0.1
      sorted_data[pos_mask] = midpoint + 0.1
      # Reorder the data array to the original order
      reordered_data = sorted_data[np.argsort(sort_indices)]
      # Return the modified data
      return reordered_data

# =============================================================================
#     def modify_prediction_results(rawData, prediction_results, failedMarkersinData):
#         # Identify the index of the maximum value for each column in rawData
#         max_index = rawData.idxmax()
#         # Iterate through the specified columns of rawData
#         for col in failedMarkersinData:
#             # Get the index of the maximum value
#             max_row = max_index[col]
#             # Modify the corresponding index in prediction_results
#             prediction_results[col].at[max_row] = 'pos'
# =============================================================================

    def get_second_highest_values(df, failedMarkersinData):
        # get the second largest value for each column in the list
        second_highest_values = df[failedMarkersinData].max()
        # convert the series to a dictionary
        second_highest_values_dict = second_highest_values.to_dict()
        return second_highest_values_dict


    # sigmoid scaling to convert the data between 0-1 based on the midpoint
    def sigmoid(x, midpoint):
        return 1 / (1 + np.exp(-(x - midpoint)))

    # rescale based on min-max neg -> 0-4.9 and pos -> 0.5-1
    def scale_data(data, midpoint):
        below_midpoint = data[data <= midpoint]
        above_midpoint = data[data > midpoint]
        indices_below = np.where(data <= midpoint)[0]
        indices_above = np.where(data > midpoint)[0]

        # Scale the group below the midpoint
        min_below = below_midpoint.min()
        max_below = below_midpoint.max()
        range_below = max_below - min_below
        below_midpoint = (below_midpoint - min_below) / range_below

        # Scale the group above the midpoint
        if len(above_midpoint) > 0:
            min_above = above_midpoint.min()
            max_above = above_midpoint.max()
            range_above = max_above - min_above
            above_midpoint = (above_midpoint - min_above) / range_above
        else:
            above_midpoint = []

        # Re-assemble the data in the original order by using the indices of the values in each group
        result = np.empty(len(data))
        result[indices_below] = below_midpoint * 0.499999999
        if len(above_midpoint) > 0:
            result[indices_above] = above_midpoint * 0.50 + 0.50
        return result
    
    # classifies data based on a given midpoint
    def classify_data(data, sorted_means):
        data = np.array(data)
        low = sorted_means[0]
        high = sorted_means[1]
        return np.where(data < low, 'neg', np.where(data > high, 'pos', 'unknown'))
            


    ###########################################################################
    # step-1 : Identify markers that have failed in this dataset
    ###########################################################################
    # 0ld thresholding method
    #failed_markers = get_columns_with_low_values (df=adata.uns[gatorScore],minAbundance=minAbundance)
    
    # New GMM method
    failed_markers = simpleGMM_failedMarkers (df=adata.uns[gatorScore], 
                                              n_components=2, 
                                              minAbundance=minAbundance, 
                                              random_state=random_state)
    
    # to store in adata
    failed_markers_dict = {adata.obs[imageid].unique()[0] : failed_markers}

    if verbose is True:
        print('Failed Markers are: ' + ", ".join(str(x) for x in failed_markers))

    ###########################################################################
    # step-2 : Prepare DATA
    ###########################################################################

    rawData = pd.DataFrame(adata.raw.X, columns= adata.var.index, index = adata.obs.index)
    rawprocessed = pre_process (rawData, log=log)
    # drop user defined markers; note if a marker is dropped it will not be part of the
    # final prediction too. Markers that failed although removed from prediction will
    # still be included in the final predicted output as all negative.
    if dropMarkers is not None:
        if isinstance(dropMarkers, str):
            dropMarkers = [dropMarkers]
        pre_processed_data = rawprocessed.drop(columns=dropMarkers)
    else:
        pre_processed_data = rawprocessed.copy()

    # also drop failed markers
    failedMarkersinData = list(set(pre_processed_data.columns).intersection(failed_markers))

    # final dataset that will be used for prediction
    pre_processed_data = pre_processed_data.drop(columns=failedMarkersinData)

    # isolate the unet probabilities
    probQuant_data = adata.uns[gatorScore]

    # list of markers to process: (combined should match data)
    expression_unet_common = list(set(pre_processed_data.columns).intersection(set(probQuant_data.columns)))
    only_expression = list(set(pre_processed_data.columns).difference(set(probQuant_data.columns)))


    ###########################################################################
    # step-4 : Identify a subset of true positive and negative cells
    ###########################################################################

    # marker = 'CD4'
    def bonafide_cells (marker,
                        expression_unet_common, only_expression,
                        pre_processed_data, probQuant_data, random_state,
                        percentiles):


        if marker in expression_unet_common:
            if verbose is True:
                print("NN marker: " + str(marker))
            # run GMM on probQuant_data
            X = probQuant_data[marker].values.reshape(-1,1)
            # Fit the GMM model to the data
            labels, sorted_means = simpleGMM (data=X, n_components=2, means_init=None, random_state=random_state)
        
            # Identify cells that are above a certain threshold in the probability maps
            if stringentThreshold is True:
                labels = classify_data (data=probQuant_data[marker], sorted_means=sorted_means)

            # find the mean of the pos and neg cells in expression data given the labels
            values = pre_processed_data [marker].values
            Pmeans = array_mean (labels, values)
            # Format mean to pass into next GMM
            Pmean = np.array([[ Pmeans.get('neg')], [Pmeans.get('pos')]])

            # Now run GMM on the expression data
            Y = pre_processed_data[marker].values.reshape(-1,1)
            labelsE, sorted_meansE = simpleGMM (data=Y, n_components=2, means_init=Pmean, random_state=random_state)

            # Match the labels and index names to identify which cells are pos and neg
            expCells = array_match (labels=labels, names=pre_processed_data.index)
            probCells = array_match (labels=labelsE, names=pre_processed_data.index)
            # split it
            expCellsPos = expCells.get('pos', []) ; expCellsNeg = expCells.get('neg', [])
            probCellsPos = probCells.get('pos', []) ; probCellsNeg = probCells.get('neg', [])
            # find common elements
            pos = list(set(expCellsPos).intersection(set(probCellsPos)))
            neg = list(set(expCellsNeg).intersection(set(probCellsNeg)))

            # print no of cells
            if verbose is True:
                print("POS cells: {} and NEG cells: {}.".format(len(pos), len(neg)))

            # check if the length is less than 20 cells and if so add the marker to only_expression
            if len(pos) < 20 or len(neg) < 20: ## CHECK!
                only_expression.append(marker)
                if verbose is True:
                    print ("As the number of POS/NEG cells is low for " + str(marker) + ", GMM will fitted using only expression values.")


        if marker in only_expression:
            if verbose is True:
                print("Expression marker: " + str(marker))
            # Run GMM only on the expression data
            Z = pre_processed_data[marker].values.reshape(-1,1)
            # if user provides manual percentile, use it to intialize the GMM
            if percentiles is not None:
                percentiles.sort()
                F = pre_processed_data[marker].values
                # mean of cells within defined threshold
                lowerPercent = meanPercentile (values=F, lowPercentile=percentiles[0], highPercentile=percentiles[1])
                higherPercent = meanPercentile (values=F, lowPercentile=percentiles[2], highPercentile=percentiles[3])
                # Format mean to pass into next GMM
                Pmean = np.array([[lowerPercent], [higherPercent]])
                labelsOE, sorted_meansOE = simpleGMM (data=Z, n_components=2, means_init=Pmean, random_state=random_state)
            else:
                labelsOE, sorted_meansOE = simpleGMM (data=Z, n_components=2, means_init=None, random_state=random_state)
            # match labels with indexname
            OEcells = array_match (labels=labelsOE, names=pre_processed_data.index)
            # split it
            pos = OEcells.get('pos', []) ; neg = OEcells.get('neg', [])

            # randomly subset 70% of the data to return
            random.seed(random_state); pos = random.sample(pos, k=int(len(pos) * 0.7))
            random.seed(random_state); neg = random.sample(neg, k=int(len(neg) * 0.7))

            # print no of cells
            if verbose is True:
                print("Defined POS cells is {} and NEG cells is {}.".format(len(pos), len(neg)))

            # What happens of POS/NEG is less than 20
            # check if the length is less than 20 cells and if so add the marker to only_expression
            if len(pos) < 20 or len(neg) < 20:  ## CHECK!
                if percentiles is None:
                    percentiles = [1,20,80,99]
                neg = list(indexPercentile (pre_processed_data, marker, lowPercentile=percentiles[0], highPercentile=percentiles[1]))
                pos = list(indexPercentile (pre_processed_data, marker, lowPercentile=percentiles[2], highPercentile=percentiles[3]))
                if verbose is True:
                    print ("As the number of POS/NEG cells is low for " + str(marker) + ", cells falling within the given percentile " + str(percentiles) + ' was used.')

        # return the output
        return marker, pos, neg

    # Run the function on all markers
    if verbose is True:
        print("Intial GMM Fitting")
    r_bonafide_cells = lambda x: bonafide_cells (marker=x,
                                                expression_unet_common=expression_unet_common,
                                                only_expression=only_expression,
                                                pre_processed_data=pre_processed_data,
                                                probQuant_data=probQuant_data,
                                                random_state=random_state,
                                                percentiles=percentiles)
    bonafide_cells_result = list(map(r_bonafide_cells,  pre_processed_data.columns)) # Apply function
    


    ###########################################################################
    # step-5 : Generate training data for the Gradient Boost Classifier
    ###########################################################################

    # bonafide_cells_result = bonafide_cells_result[2]
    def trainingData (bonafide_cells_result, pre_processed_data, RobustScale):
        # uravel the data
        marker = bonafide_cells_result[0]
        pos = bonafide_cells_result[1]
        neg = bonafide_cells_result[2]
        PD = pre_processed_data.copy()

        if verbose is True:
            print('Processing: ' + str(marker))

        # class balance the number of pos and neg cells based on the lowest denominator
        if len(neg) < len(pos):
            pos = random.sample(pos, len(neg))
        else:
            neg = random.sample(neg, len(pos))

        # processed data with pos and neg info


        #PD['label'] = ['pos' if index in pos else 'neg' if index in neg else 'other' for index in PD.index]
        PD['label'] = np.where(PD.index.isin(pos), 'pos', np.where(PD.index.isin(neg), 'neg', 'other'))
        combined_data = PD.copy()

        # scale data if requested
        if RobustScale is True:
            combined_data_labels = combined_data[['label']]
            combined_data = combined_data.drop('label', axis=1)
            combined_data = apply_transformation(combined_data)
            combined_data = pd.concat ([combined_data, combined_data_labels], axis=1)

        # return final output
        return marker, combined_data

    # Run the function
    if verbose is True:
        print("Building the Training Data")
    r_trainingData = lambda x: trainingData (bonafide_cells_result=x,
                                                 pre_processed_data=pre_processed_data,
                                                 RobustScale=RobustScale)
    trainingData_result = list(map(r_trainingData, bonafide_cells_result)) # Apply function


    ###########################################################################
    # step-6 : Train and Predict on all cells
    ###########################################################################

    # trainingData_result = trainingData_result[2]
    def gatorClassifier (trainingData_result,random_state):

        #unravel data
        marker = trainingData_result[0]
        combined_data = trainingData_result[1]

        # prepare the data for predicition
        index_names_to_drop = [index for index in combined_data.index if 'npu' in index or 'nnu' in index]
        predictionData = combined_data.drop(index=index_names_to_drop, inplace=False)
        predictionData = predictionData.drop('label', axis=1)

        if verbose is True:
            print('classifying: ' + str(marker))

        # shuffle the data
        combined_data = combined_data.sample(frac=1) # shuffle it

        # prepare the training data and training labels
        to_train = combined_data.loc[combined_data['label'].isin(['pos', 'neg'])]
        training_data = to_train.drop('label', axis=1)
        training_labels = to_train[['label']]
        trainD = training_data.values
        trainL = training_labels.values
        trainL = [item for sublist in trainL for item in sublist]


        #start = time.time()

        # Function for the classifier
        #mlp = MLPClassifier(**kwargs) # CHECK
        #model = GradientBoostingClassifier()

        model = HistGradientBoostingClassifier(random_state=random_state, **kwargs)
        model.fit(trainD, trainL)
        # predict
        pred = model.predict(predictionData.values)
        prob = model.predict_proba(predictionData.values)
        prob = [item[0] for item in prob]

        #end = time.time()
        #print(end - start)

        # find the mid point based on the predictions (used for rescaling data later)
        midpoint = find_midpoint(data=predictionData[marker].values, labels=pred)

        # return
        return marker, pred, prob, midpoint

    # Run the function
    if verbose is True:
        print("Fitting model for classification:")
    r_gatorClassifier = lambda x: gatorClassifier (trainingData_result=x,random_state=random_state)
    gatorClassifier_result = list(map(r_gatorClassifier, trainingData_result))

    ###########################################################################
    # step-7 : Consolidate the results into a dataframe
    ###########################################################################

    # consolidate results
    markerOrder = []
    for i in range(len(gatorClassifier_result)):
        markerOrder.append(gatorClassifier_result[i][0])

    prediction_results = []
    for i in range(len(gatorClassifier_result)):
        prediction_results.append(gatorClassifier_result[i][1])
    prediction_results = pd.DataFrame(prediction_results, index=markerOrder, columns=pre_processed_data.index).T

    probability_results = []
    for i in range(len(gatorClassifier_result)):
        probability_results.append(gatorClassifier_result[i][2])
    probability_results = pd.DataFrame(probability_results, index=markerOrder, columns=pre_processed_data.index).T

    midpoints_dict = {}
    for i in range(len(gatorClassifier_result)):
        midpoints_dict[markerOrder[i]] = gatorClassifier_result[i][3]

    ###########################################################################
    # step-8 : Final cleaning of predicted results with UNET results
    ###########################################################################

    # bonafide_cells_result_copy = bonafide_cells_result.copy()
    # bonafide_cells_result = bonafide_cells_result[0]

    def anomalyDetector (pre_processed_data, bonafide_cells_result, prediction_results):
        # unravel data
        marker = bonafide_cells_result[0]
        pos = bonafide_cells_result[1]
        neg = bonafide_cells_result[2]

        if verbose is True:
            print("Processing: " + str(marker))
        # prepare data
        X = pre_processed_data.drop(marker, axis=1)
        # scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # model data
        model = LocalOutlierFactor(n_neighbors=20)
        model.fit(X_scaled)
        outlier_scores = model.negative_outlier_factor_
        outliers = pre_processed_data[outlier_scores < -1].index

        # common elements betwenn outliers and true neg
        posttoneg = list(set(outliers).intersection(set(neg)))
        # similarly is there any cells in negative that needs to be relocated to positive?
        negtopos = list(set(pos).intersection(set(prediction_results[prediction_results[marker]=='neg'].index)))

        # mutate the prediction results
        prediction_results.loc[posttoneg, marker] = 'neg'
        prediction_results.loc[negtopos, marker] = 'pos'

        # results
        results = prediction_results[[marker]]
        return results

    # Run the function
    if verbose is True:
        print("Running Anomaly Detection")
    r_anomalyDetector = lambda x: anomalyDetector (bonafide_cells_result = x,
                                                   pre_processed_data = pre_processed_data,
                                                   prediction_results = prediction_results)

    # as the Anomaly Detection uses the rest of the data it cannot be run on 1 marker
    if len(bonafide_cells_result) > 1:
        anomalyDetector_result = list(map(r_anomalyDetector, bonafide_cells_result))
        # final prediction
        prediction_results = pd.concat(anomalyDetector_result, axis=1)


    ###########################################################################
    # step-9 : Reorganizing all predictions into a final dataframe
    ###########################################################################

    # re introduce failed markers
    if len(failedMarkersinData) > 0 :
        for name in failedMarkersinData:
            prediction_results[name] = 'neg'

        # modify the highest value element to be pos
        #modify_prediction_results(rawprocessed, prediction_results, failedMarkersinData)

        # identify midpoints for the failed markers (second largest element)
        max_values_dict = get_second_highest_values (rawprocessed, failedMarkersinData)

        # update midpoints_dict
        midpoints_dict.update(max_values_dict)

        # add the column to pre_processed data for rescaling
        columns_to_concat = rawprocessed[failedMarkersinData]
        pre_processed_data = pd.concat([pre_processed_data, columns_to_concat], axis=1)


    ###########################################################################
    # step-10 : Rescale data
    ###########################################################################

    # marker = 'ECAD'
    def rescaleData (marker, pre_processed_data, prediction_results, midpoints_dict):
        if verbose is True:
            print("Processing: " + str(marker))
        # unravel data
        data = pre_processed_data[marker].values
        labels = prediction_results[marker].values
        midpoint = midpoints_dict.get(marker)

        # reformat data such that all negs and pos are sorted based on the midpoint
        rescaled = modify_negatives_vectorized(data,
                                               labels,
                                               midpoint)

        # sigmoid scaling to convert the data between 0-1 based on the midpoint
        if rescaleMethod == 'sigmoid':
            rescaled_data = sigmoid (rescaled, midpoint=midpoint)

        if rescaleMethod == 'minmax':
            rescaled_data = scale_data(rescaled, midpoint=midpoint)

        # return
        return rescaled_data

    # Run the function
    if verbose is True:
        print("Rescaling the raw data")
    r_rescaleData = lambda x: rescaleData (marker=x,
                                           pre_processed_data=pre_processed_data,
                                           prediction_results=prediction_results,
                                           midpoints_dict=midpoints_dict)
    rescaleData_result = list(map(r_rescaleData, pre_processed_data.columns))
    rescaledData = pd.DataFrame(rescaleData_result, index=pre_processed_data.columns, columns=pre_processed_data.index).T



    ###########################################################################
    # step-8 : create a new adata object with the results
    ###########################################################################


    final_markers = pre_processed_data.columns
    intial_markers = rawData.columns
    ordered_final_markers = [marker for marker in intial_markers if marker in final_markers]
    # final raw data
    rd = rawData[ordered_final_markers].reindex(adata.obs.index)
    # final scaled data
    rescaledData = rescaledData[ordered_final_markers].reindex(adata.obs.index)
    # final pre-processed data
    pre_processed_data = pre_processed_data[ordered_final_markers].reindex(adata.obs.index)


    # reindex prediction results
    prediction_results = prediction_results.reindex(adata.obs.index)
    #probability_results = probability_results.reindex(adata.obs.index)

    # create AnnData object
    bdata = ad.AnnData(rd, dtype=np.float64)
    bdata.obs = adata.obs
    bdata.raw = bdata
    bdata.X = rescaledData
    # add the pre-processed data as a layer
    bdata.layers["preProcessed"] = pre_processed_data
    bdata.uns = adata.uns
    bdata.uns['failedMarkers'] = failed_markers_dict

    # save the prediction results in anndata object
    bdata.uns[str(label)] = prediction_results
    #bdata.uns[str(label)] = probability_results

    # Save data if requested
    if projectDir is not None:
        finalPath = pathlib.Path(projectDir + '/GATOR/gatorOutput')
        if not os.path.exists(finalPath):
            os.makedirs(finalPath)
        if len(gatorObjectPath) > 1:
            imid = 'gatorOutput'
        else:
            imid = gatorObjectPath[0].stem
        bdata.write(finalPath / f'{imid}.h5ad')
        
    # Finish Job
    if verbose is True:
        print('Gator ran successfully, head over to "' + str(projectDir) + '/GATOR/gatorOutput" to view results')
        
    return bdata

# Make the Function CLI compatable
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Gator')
    parser.add_argument('--gatorObject', type=str, help='Single or combined Gator object')
    parser.add_argument('--gatorScore', type=str, default='gatorScore', help='Include the label used for saving the `gatorScore` within the Gator object')
    parser.add_argument('--minAbundance', type=float, default=0.002, help='Specify the minimum percentage of cells that should express a specific marker in order to determine if the marker is considered a failure')
    parser.add_argument('--percentiles', type=list, default=[1, 20, 80, 99], help='Specify the interval of percentile levels of the expression utilized to intialize the GMM')
    parser.add_argument('--dropMarkers', type=list, default=None, help='Specify a list of markers to be removed from the analysis')
    parser.add_argument('--RobustScale', type=bool, default=False, help='When set to True, the data will be subject to Robust Scaling before the Gradient Boosting Classifier is trained')
    parser.add_argument('--log', type=bool, default=True, help='Apply `log1p` transformation on the data, unless it has already been log transformed in which case set it to `False`')
    parser.add_argument('--stringentThreshold', type=bool, default=False, help='Threshold to refine the classification of positive and negative cells')
    parser.add_argument('--x_coordinate', type=str, default='X_centroid', help='The column name in `single-cell spatial table` that records the X coordinates for each cell')
    parser.add_argument('--y_coordinate', type=str, default='Y_centroid', help='The column name in `single-cell spatial table` that records the Y coordinates for each cell')
    parser.add_argument('--imageid', type=str, default='imageid', help='The name of the column that holds the unique image ID')
    parser.add_argument('--random_state', type=int, default=0, help='Seed used by the random number generator')
    parser.add_argument('--rescaleMethod', type=str, default='minmax', help='Choose between `sigmoid` and `minmax`')
    parser.add_argument('--label', type=str, default='gatorOutput', help='Assign a label for the object within `adata.uns` where the predictions from Gator will be stored')
    parser.add_argument('--verbose', type=bool, default=True, help='Enables the display of step-by-step events on the console')
    parser.add_argument('--projectDir', type=str, default=None, help='Provide the path to the output directory')
    args = parser.parse_args()
    gator(gatorObject=args.gatorObject,
          gatorScore=args.gatorScore,
          minAbundance=args.minAbundance,
          percentiles=args.percentiles,
          dropMarkers=args.dropMarkers,
          RobustScale=args.RobustScale,
          log=args.log,
          stringentThreshold=args.stringentThreshold, 
          x_coordinate=args.x_coordinate,
          y_coordinate=args.y_coordinate,
          imageid=args.imageid,
          random_state=args.random_state,
          rescaleMethod=args.rescaleMethod,
          label=args.label,
          verbose=args.verbose,
          projectDir=args.projectDir)
