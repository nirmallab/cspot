#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 11:56:32 2022
@author: Ajit Johnson Nirmal
Gator Algorithm
"""

import pandas as pd
import anndata as ad
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import BallTree
import random
import numpy as np
from sklearn.mixture import GaussianMixture
from collections import defaultdict
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import pathlib
import os


def gator (gatorObject,
           probQuant='probQuant',
           minAbundance=0.002,
           percentiles=[1, 20, 80, 99],
           drop_markers = None, 
           leakData=True,
           scaleData=True,
           log=True,
           x_coordinate='X_centroid',
           y_coordinate='Y_centroid',
           imageid='imageid',
           seed=0,
           rescaleMethod='minmax',
           label='gatorPredict', 
           silent=True,
           outputDir=None, **kwargs):
    
    #gatorObject = "/Users/aj/Dropbox (Partners HealthCare)/Data/gator/data/ajn_training_data/GATOR/gatorObject/1_6_GatorOutput.h5ad"
    #probQuant='probQuant'; minAbundance=0.002; percentiles=[1, 20, 80, 99]; drop_markers = None;leakData=False;rescaleMethod='minmax';silent = False
    #scaleData=False; log=True; x_coordinate='X_centroid'; y_coordinate='Y_centroid'; imageid='imageid'; seed=0; label='gatorPredict'
    #gatorObject = '/Users/aj/Dropbox (Partners HealthCare)/Data/gator/data/ajn_training_data/GATOR/gatorPredict/2_28_GatorOutput.h5ad'
    #gatorObject = '/Users/aj/Dropbox (Partners HealthCare)/Data/gator/data/ajn_training_data/GATOR/gatorPredict/4_113_GatorOutput.h5ad'


    # Load the andata object    
    if isinstance(gatorObject, str):
        adata = ad.read(gatorObject)
        gatorObject = [gatorObject]
        gatorObjectPath = [pathlib.Path(p) for p in gatorObject]
    else:
        adata = gatorObject.copy()
    
    # break the function if probQuant is not detectable
    def check_key_exists(dictionary, key):
        try:
            # Check if the key exists in the dictionary
            value = dictionary[key]
        except KeyError:
            # Return an error if the key does not exist
            return "Error: " + str(probQuant) + " does not exist, please check!"
    # Test 
    check_key_exists(dictionary=adata.uns, key=probQuant)  
    
    
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

        return labels
    
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
         
    # Leaking expression inforamtion into neighbouring cells function (step-4)
    def leakDataF (bonafideCells, processed_data, neighbours, marker, seed):
        cells = processed_data.loc[bonafideCells].reset_index(drop=True)#[marker].values  
        cells_neigh = neighbours[neighbours['cell'].isin(bonafideCells)]['neigh'].values        
        # find original expression profile of both the cell and its neighbour
        cells_neigh = processed_data.loc[cells_neigh].reset_index(drop=True)#[marker].values
        # formula leak of 5-40%
        # x = (a + leak*b) / 1+ leak
        random.seed(seed)
        leak = [(random.randint(5, 40) / 100) for _ in range(len(cells))]
        nLeak = [x + 1 for x in leak] # normalizing the cell size        
        def transform_dataframes(cells, cells_neigh, leak, nLeak):
            # Multiply leak vector to all columns of cells_neigh
            cells_neigh_transformed = cells_neigh.multiply(leak, axis=0)
            # Add cells_neigh_transformed to cells element-wise
            cells_transformed = cells.add(cells_neigh_transformed)
            # Divide each column of cells_transformed by 1+leak
            cells_final = cells_transformed.divide(nLeak, axis=0)            
            return cells_final
        # Run the function
        leakData_result = transform_dataframes (cells, cells_neigh, leak, nLeak)
        return leakData_result
    
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

    
    ###########################################################################
    # step-1 : Identify markers that have failed in this dataset
    ###########################################################################

    failed_markers = get_columns_with_low_values (df=adata.uns[probQuant],minAbundance=minAbundance)
    
    if silent is False:
        print('Failed Markers are: ' + ", ".join(str(x) for x in failed_markers))

    ###########################################################################
    # step-2 : Prepare DATA
    ###########################################################################
    
    rawData = pd.DataFrame(adata.raw.X, columns= adata.var.index, index = adata.obs.index)
    rawprocessed = pre_process (rawData, log=log)
    # drop user defined markers; note if a marker is dropped it will not be part of the 
    # final prediction too. Markers that failed although removed from prediction will
    # still be included in the final predicted output as all negative. 
    if drop_markers is not None: 
        if isinstance(drop_markers, str):
            drop_markers = [drop_markers]
        pre_processed_data = rawprocessed.drop(columns=drop_markers)
    else:
        pre_processed_data = rawprocessed.copy()
    
    # also drop failed markers
    failedMarkersinData = list(set(pre_processed_data.columns).intersection(failed_markers))
    
    # final dataset that will be used for prediction
    pre_processed_data = pre_processed_data.drop(columns=failedMarkersinData)
    
    # isolate the unet probabilities
    probQuant_data = adata.uns[probQuant]
    
    # list of markers to process: (combined should match data)
    expression_unet_common = list(set(pre_processed_data.columns).intersection(set(probQuant_data.columns)))
    only_expression = list(set(pre_processed_data.columns).difference(set(probQuant_data.columns)))


    ###########################################################################
    # step-3 : Build the spatial neighbourhood graph
    ###########################################################################
    
    if leakData is True: 
        spatial_data = adata.obs[[x_coordinate,y_coordinate ]]
        # identify the nearest neighbour
        tree = BallTree(spatial_data, leaf_size= 2)
        ind = tree.query(spatial_data, k=2, return_distance= False)
        neighbours = pd.DataFrame(ind.tolist()) # neighbour DF
        neighbours = neighbours.dropna(how='all')
        # map the cell id for the neighbours
        phenomap = dict(zip(list(range(len(ind))), spatial_data.index)) # Used for mapping
        for i in neighbours.columns:
            neighbours[i] = neighbours[i].dropna().map(phenomap, na_action='ignore')
        neighbours.columns = ['cell', 'neigh']
    else:
        neighbours = None
        
        
    ###########################################################################
    # step-4 : Identify a subset of true positive and negative cells    
    ###########################################################################
    
    # marker = 'CD4'
    def bonafide_cells (marker, 
                        expression_unet_common, only_expression, 
                        pre_processed_data, probQuant_data, seed,
                        percentiles):
        
        
        if marker in expression_unet_common:
            if silent is False:
                print("NN marker: " + str(marker))
            # run GMM on probQuant_data
            X = probQuant_data[marker].values.reshape(-1,1)
            # Fit the GMM model to the data
            labels = simpleGMM (data=X, n_components=2, means_init=None, random_state=seed)
            
            # find the mean of the pos and neg cells in expression data given the labels
            values = pre_processed_data [marker].values
            Pmeans = array_mean (labels, values)
            # Format mean to pass into next GMM
            Pmean = np.array([[ Pmeans.get('neg')], [Pmeans.get('pos')]])
             
            # Now run GMM on the expression data
            Y = pre_processed_data[marker].values.reshape(-1,1)
            labelsE = simpleGMM (data=Y, n_components=2, means_init=Pmean, random_state=seed)
            
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
            if silent is False:
                print("POS cells: {} and NEG cells: {}.".format(len(pos), len(neg)))
            
            # check if the length is less than 20 cells and if so add the marker to only_expression
            if len(pos) < 20 or len(neg) < 20: ## CHECK!
                only_expression.append(marker)
                if silent is False:
                    print ("As the number of POS/NEG cells is low for " + str(marker) + ", GMM will fitted using only expression values.")
            

        if marker in only_expression:
            if silent is False:
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
                labelsOE = simpleGMM (data=Z, n_components=2, means_init=Pmean, random_state=seed)
            else:
                labelsOE = simpleGMM (data=Z, n_components=2, means_init=None, random_state=seed)
            # match labels with indexname
            OEcells = array_match (labels=labelsOE, names=pre_processed_data.index)
            # split it
            pos = OEcells.get('pos', []) ; neg = OEcells.get('neg', [])   
            
            # randomly subset 70% of the data to return
            random.seed(seed); pos = random.sample(pos, k=int(len(pos) * 0.7))
            random.seed(seed); neg = random.sample(neg, k=int(len(neg) * 0.7)) 
                
            # print no of cells
            if silent is False:
                print("Defined POS cells is {} and NEG cells is {}.".format(len(pos), len(neg)))
            
            # What happens of POS/NEG is less than 20
            # check if the length is less than 20 cells and if so add the marker to only_expression
            if len(pos) < 20 or len(neg) < 20:  ## CHECK!
                if percentiles is None:
                    percentiles = [1,20,80,99]
                neg = list(indexPercentile (pre_processed_data, marker, lowPercentile=percentiles[0], highPercentile=percentiles[1]))
                pos = list(indexPercentile (pre_processed_data, marker, lowPercentile=percentiles[2], highPercentile=percentiles[3]))
                if silent is False:
                    print ("As the number of POS/NEG cells is low for " + str(marker) + ", cells falling within the given percentile " + str(percentiles) + ' was used.')      

        # return the output
        return marker, pos, neg
    
    # Run the function on all markers
    if silent is False:
        print("Intial GMM Fitting")
    r_bonafide_cells = lambda x: bonafide_cells (marker=x, 
                                                expression_unet_common=expression_unet_common, 
                                                only_expression=only_expression, 
                                                pre_processed_data=pre_processed_data, 
                                                probQuant_data=probQuant_data, 
                                                seed=seed,
                                                percentiles=percentiles)
    bonafide_cells_result = list(map(r_bonafide_cells,  pre_processed_data.columns)) # Apply function  


    ###########################################################################
    # step-5 : Generate training data for the Gradient Boost Classifier
    ###########################################################################
    
    # bonafide_cells_result = bonafide_cells_result[0]
    def trainingData (bonafide_cells_result, pre_processed_data, neighbours, leakData, scaleData):
        # uravel the data
        marker = bonafide_cells_result[0]
        pos = bonafide_cells_result[1]
        neg = bonafide_cells_result[2]
        PD = pre_processed_data.copy()
        
        if silent is False:
            print('Processing: ' + str(marker))
        
        # class balance the number of pos and neg cells based on the lowest denominator
        if len(neg) < len(pos):
            pos = random.sample(pos, len(neg))
        else:
            neg = random.sample(neg, len(pos))
             
        # processed data with pos and neg info
        
        
        #PD['label'] = ['pos' if index in pos else 'neg' if index in neg else 'other' for index in PD.index]
        PD['label'] = np.where(PD.index.isin(pos), 'pos', np.where(PD.index.isin(neg), 'neg', 'other'))


        
        if leakData is True:
            # Now based on the bonafide cells, leak the neighbour information into these cells
            # Leak information into negative cells
            negLeakedDF = leakDataF (bonafideCells=neg, 
                                   processed_data=pre_processed_data, 
                                   neighbours=neighbours,
                                   marker=marker,
                                   seed=seed)
            new_index_names = ['nnu_{}'.format(idx) for idx in negLeakedDF.index]
            negLeakedDF.index = new_index_names
            negLeakedDF['label'] = 'neg'
            
            # Leak information into positive cells
            posLeakedDF = leakDataF (bonafideCells=pos, 
                                   processed_data=pre_processed_data, 
                                   neighbours=neighbours,
                                   marker=marker,
                                   seed=seed)   
            new_index_names = ['npu_{}'.format(idx) for idx in posLeakedDF.index]
            posLeakedDF.index = new_index_names
            posLeakedDF['label'] = 'pos'
            
            # combine all cells togeather
            combined_data = pd.concat([PD, negLeakedDF, posLeakedDF])
        else:
            combined_data = PD
        
        # scale data if requested
        if scaleData is True:
            combined_data_labels = combined_data[['label']]
            combined_data = combined_data.drop('label', axis=1)
            combined_data = apply_transformation(combined_data)
            combined_data = pd.concat ([combined_data, combined_data_labels], axis=1)
        
        # return final output
        return marker, combined_data
    
    # Run the function
    if silent is False:
        print("Building the Training Data")
    r_trainingData = lambda x: trainingData (bonafide_cells_result=x, 
                                                 pre_processed_data=pre_processed_data, 
                                                 neighbours=neighbours, 
                                                 leakData = leakData,
                                                 scaleData=scaleData)
    trainingData_result = list(map(r_trainingData, bonafide_cells_result)) # Apply function  


    ###########################################################################
    # step-6 : Train and Predict on all cells
    ###########################################################################
    
    # trainingData_result = trainingData_result[2]
    def gatorClassifier (trainingData_result):
        
        #unravel data
        marker = trainingData_result[0]
        combined_data = trainingData_result[1]
        
        # prepare the data for predicition
        index_names_to_drop = [index for index in combined_data.index if 'npu' in index or 'nnu' in index]
        predictionData = combined_data.drop(index=index_names_to_drop, inplace=False)
        predictionData = predictionData.drop('label', axis=1)

        if silent is False:
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
        
        model = HistGradientBoostingClassifier()
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
    if silent is False:
        print("Fitting model for classification:")
    r_gatorClassifier = lambda x: gatorClassifier (trainingData_result=x)
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
        
        if silent is False:
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
    if silent is False:
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
        if silent is False:
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
    if silent is False:
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
    bdata = ad.AnnData(rd)
    bdata.obs = adata.obs
    bdata.raw = bdata
    bdata.X = rescaledData
    # add the pre-processed data as a layer
    bdata.layers["preProcessed"] = pre_processed_data
    bdata.uns = adata.uns
    
    # save the prediction results in anndata object
    bdata.uns[str(label)] = prediction_results
    #bdata.uns[str(label)] = probability_results
        
    # Save data if requested
    if outputDir is not None:
        finalPath = pathlib.Path(outputDir + '/GATOR/gatorPredict')
        if not os.path.exists(finalPath):
            os.makedirs(finalPath)
        if len(gatorObjectPath) > 1:
            imid = 'gatorPredict'
        else:
            imid = gatorObjectPath[0].stem 
        bdata.write(finalPath / f'{imid}.h5ad')

    return bdata


    
    
    
    
    
    
    
    
    
    