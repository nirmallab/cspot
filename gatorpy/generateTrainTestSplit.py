#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Thu Aug 18 16:37:29 2022
#@author: Ajit Johnson Nirmal
#Function to generate Masks for UNET model


"""
!!! abstract "Short Description"
    The function generates a mask for the deep learning model training, using 
    automated approaches. Splitting the data into training, validation and 
    test sets is also included in the function, making it easier to feed the 
    data directly into the deep learning algorithm. Note that manually drawing 
    the mask on thumbnails is the ideal approach, however for scalability 
    purposes, automation is used.


## Function
"""

# Libs
import pathlib
import cv2 as cv
import random
import numpy as np
import tifffile
import argparse

# Function
def generateTrainTestSplit (thumbnailFolder, 
                            projectDir, 
                            file_extension=None,
                            verbose=True,
                            TruePos='TruePos', NegToPos='NegToPos',
                            TrueNeg='TrueNeg', PosToNeg='PosToNeg'):
    """
Parameters:
    thumbnailFolder (list):
        List of folders that contains the human sorted Thumbnails that is to be used
        for generating training data and split them train test and validation cohorts.

    projectDir (str):
        Path to output directory.

    file_extension (str, optional):
        If there are non-image files in the thumbnailFolder, the user can specify
        a file extension to only select those files for processing. The default is None.

    verbose (bool, optional):
        If True, print detailed information about the process to the console. 

    TruePos (str, optional):
        Name of the folder that holds the Thumbnails classified as True Positive.
        The default is 'TruePos'.

    NegToPos (str, optional):
        Name of the folder that holds the Thumbnails classified as True Negative.
        The default is 'NegToPos'.

    TrueNeg (str, optional):
        Name of the folder that holds the Thumbnails that were moved from `True Positive`
        to `True Negative`. The default is 'TrueNeg'.

    PosToNeg (str, optional):
        Name of the folder that holds the Thumbnails that were moved from `True Negative`
        to `True Positive`. The default is 'PosToNeg'.

Returns:
    masks (images):
        Segmentation masks are generated for every Thumbnail and split into Train,
        Test and Validation cohorts.

Example:

        ```python
        
        # High level working directory
        cwd = '/Users/aj/Desktop/gatorExampleData'
        
        # Folder where the raw Thumbnails are stored
        thumbnailFolder = [cwd + '/GATOR/Thumbnails/CD3D',
                           cwd + '/GATOR/Thumbnails/ECAD']
        projectDir = cwd
        
        # The function accepts the four pre-defined folders. If you had renamed them, please change it using the parameter below.
        # If you had deleted any of the folders and are not using them, replace the folder name with `None` in the parameter.
        ga.generateTrainTestSplit ( thumbnailFolder, 
                                    projectDir, 
                                    file_extension=None,
                                    verbose=True,
                                    TruePos='TruePos', NegToPos='NegToPos',
                                    TrueNeg='TrueNeg', PosToNeg='PosToNeg')
        
        # Same function if the user wants to run it via Command Line Interface
        python generateTrainTestSplit.py --thumbnailFolder /Users/aj/Desktop/gatorExampleData/GATOR/Thumbnails/CD3D /Users/aj/Desktop/gatorExampleData/GATOR/Thumbnails/ECAD --projectDir /Users/aj/Desktop/gatorExampleData/
        
        ```

    """

    # Function takes in path to two folders, processes the images in those folders,
    # and saves them into a different folder that contains Train, Validation and Test samples
    #TruePos='TruePos'; NegToPos='NegToPos'; TrueNeg='TrueNeg'; PosToNeg='PosToNeg'; verbose=True

    # convert the folder into a list
    if isinstance (thumbnailFolder, str):
        thumbnailFolder = [thumbnailFolder]

    # convert all path names to pathlib
    thumbnailFolder = [pathlib.Path(p) for p in thumbnailFolder]
    projectDir = pathlib.Path(projectDir)

    # find all markers passed
    all_markers = [i.stem for i in thumbnailFolder]

    # create directories to save
    for i in all_markers:
        if not (projectDir / 'GATOR/TrainingData/' / f"{i}" /  'training').exists ():
            (projectDir / 'GATOR/TrainingData/' / f"{i}" /  'training').mkdir(parents=True, exist_ok=True)

        if not (projectDir / 'GATOR/TrainingData/' / f"{i}" /  'validation').exists ():
            (projectDir / 'GATOR/TrainingData/' / f"{i}" /  'validation').mkdir(parents=True, exist_ok=True)

        if not (projectDir / 'GATOR/TrainingData/' / f"{i}" /  'test').exists ():
            (projectDir / 'GATOR/TrainingData/' / f"{i}" /  'test').mkdir(parents=True, exist_ok=True)

    # standard format
    if file_extension is None:
        file_extension = '*'
    else:
        file_extension = '*' + str(file_extension)

    # Filter on pos cells
    def pos_filter (path):
        image = cv.imread(str(path.resolve()), cv.IMREAD_GRAYSCALE)
        blur = cv.GaussianBlur(image, ksize=(3,3), sigmaX=1, sigmaY=1)
        ret3,th3 = cv.threshold(blur,0,1,cv.THRESH_OTSU)
        mask = th3 + 1
        return [mask, image]

    # Filter on neg cells
    def neg_filter (path):
        image = cv.imread(str(path.resolve()), cv.IMREAD_GRAYSCALE)
        mask = np.ones(image.shape, dtype=np.uint8)
        return [mask, image]

    # identify the files within all the 4 folders
    def findFiles (folderIndex):
        if verbose is True:
            print ('Processing: ' + str(thumbnailFolder[folderIndex].stem))
        marker_name = str(thumbnailFolder[folderIndex].stem)

        baseFolder = thumbnailFolder[folderIndex]

        if TruePos is not None:
            pos = list(pathlib.Path.glob(baseFolder / TruePos, file_extension))
        if NegToPos is not None:
            negtopos = list(pathlib.Path.glob(baseFolder / NegToPos, file_extension))
        positive_cells = pos + negtopos

        if TrueNeg is not None:
            neg = list(pathlib.Path.glob(baseFolder / TrueNeg, file_extension))
        if PosToNeg is not None:
            postoneg = list(pathlib.Path.glob(baseFolder / PosToNeg, file_extension))
        negative_cells = neg + postoneg

        # prepare the Training, Validataion and Test Cohorts
        if len(positive_cells) > 0:
            train_pos = random.sample(positive_cells, round(len(positive_cells) * 0.6))
            remanining_pos = list(set(positive_cells) - set(train_pos))
            val_pos = random.sample(remanining_pos, round(len(remanining_pos) * 0.5)) # validation
            test_pos = list(set(remanining_pos) - set(val_pos)) # test
        else:
            train_pos = []; val_pos = []; test_pos = []
        if len(negative_cells) > 0:
            train_neg = random.sample(negative_cells, round(len(negative_cells) * 0.6))
            remanining_neg = list(set(negative_cells) - set(train_neg))
            val_neg = random.sample(remanining_neg, round(len(remanining_neg) * 0.5))
            test_neg = list(set(remanining_neg) - set(val_neg))
        else:
            train_neg = []; val_neg = []; test_neg = []


        # loop through training dataset and save images and masks
        newname_train = list(range(len(train_pos) + len(train_neg))); random.shuffle(newname_train)
        train_pos_name = newname_train[:len(train_pos)]; train_neg_name = newname_train[len(train_pos):]

        if len (train_pos_name) > 0:
            for i, j in zip( train_pos_name, train_pos):
                m, im = pos_filter (j)
                # save image
                fPath = projectDir / 'GATOR/TrainingData/' / f"{marker_name}" / 'training' / f"{i}_img.tif"
                tifffile.imwrite(fPath,im)
                # associated mask
                fPath = projectDir / 'GATOR/TrainingData/' / f"{marker_name}" / 'training' / f"{i}_mask.tif"
                tifffile.imwrite(fPath, m)

        if len (train_neg_name) > 0:
            for k, l in zip( train_neg_name, train_neg):
                m, im = neg_filter (l)
                # save image
                fPath = projectDir / 'GATOR/TrainingData/' / f"{marker_name}" / 'training' / f"{k}_img.tif"
                tifffile.imwrite(fPath, im)
                # associated mask
                fPath = projectDir / 'GATOR/TrainingData/' / f"{marker_name}" / 'training' / f"{k}_mask.tif"
                tifffile.imwrite(fPath, m)


        # loop through validation dataset and save images and masks
        newname_train = list(range(len(val_pos) + len(val_neg))); random.shuffle(newname_train)
        train_pos_name = newname_train[:len(val_pos)]; train_neg_name = newname_train[len(val_pos):]

        if len (train_pos_name) > 0:
            for i, j in zip( train_pos_name, val_pos):
                m, im = pos_filter (j)
                # save image
                fPath = projectDir / 'GATOR/TrainingData/' / f"{marker_name}" / 'validation' / f"{i}_img.tif"
                tifffile.imwrite(fPath, im)
                # associated mask
                fPath = projectDir / 'GATOR/TrainingData/' / f"{marker_name}" / 'validation' / f"{i}_mask.tif"
                tifffile.imwrite(fPath, m)

        if len (train_neg_name) > 0:
            for k, l in zip( train_neg_name, val_neg):
                m, im = neg_filter (l)
                # save image
                fPath = projectDir / 'GATOR/TrainingData/' / f"{marker_name}" / 'validation' / f"{k}_img.tif"
                tifffile.imwrite(fPath, im)
                # associated mask
                fPath = projectDir / 'GATOR/TrainingData/' / f"{marker_name}" / 'validation' / f"{k}_mask.tif"
                tifffile.imwrite(fPath, m)


        # loop through test dataset and save images and masks
        newname_train = list(range(len(test_pos) + len(test_neg))); random.shuffle(newname_train)
        train_pos_name = newname_train[:len(test_pos)]; train_neg_name = newname_train[len(test_pos):]

        if len (train_pos_name) > 0:
            for i, j in zip( train_pos_name, test_pos):
                m, im = pos_filter (j)
                # save image
                fPath = projectDir / 'GATOR/TrainingData/' / f"{marker_name}" / 'test' / f"{i}_img.tif"
                tifffile.imwrite(fPath, im)
                # associated mask
                fPath = projectDir / 'GATOR/TrainingData/' / f"{marker_name}" / 'test' / f"{i}_mask.tif"
                tifffile.imwrite(fPath, m)

        if len (train_neg_name) > 0:
            for k, l in zip( train_neg_name, test_neg):
                m, im = neg_filter (l)
                # save image
                fPath = projectDir / 'GATOR/TrainingData/' / f"{marker_name}" / 'test' / f"{k}_img.tif"
                tifffile.imwrite(fPath, im)
                # associated mask
                fPath = projectDir / 'GATOR/TrainingData/' / f"{marker_name}" / 'test' / f"{k}_mask.tif"
                tifffile.imwrite(fPath, m)

    # apply function to all folders
    r_findFiles = lambda x: findFiles (folderIndex=x)
    process_folders = list(map(r_findFiles, list(range(len(thumbnailFolder)))))

    # Print
    if verbose is True:
        print('Training data has been generated, head over to "' + str(projectDir) + '/GATOR/TrainingData" to view results')


# Make the Function CLI compatable
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate train, test, and validation cohorts from human sorted thumbnails.')
    parser.add_argument('--thumbnailFolder', type=str, nargs='+', help='List of folders that contains the human sorted Thumbnails that is to be used for generating training data and split them train test and validation cohorts.')
    parser.add_argument('--projectDir', type=str, help='Path to output directory.')
    parser.add_argument('--file_extension', type=str, default=None, help='If there are non-image files in the thumbnailFolder, the user can specify a file extension to only select those files for processing.')
    parser.add_argument("--verbose", type=bool, default=True, help="If True, print detailed information about the process to the console.")    
    parser.add_argument('--TruePos', type=str, default='TruePos', help='Name of the folder that holds the Thumbnails classified as True Positive.')
    parser.add_argument('--NegToPos', type=str, default='NegToPos', help='Name of the folder that holds the Thumbnails classified as True Negative.')
    parser.add_argument('--TrueNeg', type=str, default='TrueNeg', help='Name of the folder that holds the Thumbnails that were moved from `True Positive` to `True Negative`.')
    parser.add_argument('--PosToNeg', type=str, default='PosToNeg', help='Name of the folder that holds the Thumbnails that were moved from `True Negative` to `True Positive`.')
    args = parser.parse_args()
    generateTrainTestSplit(thumbnailFolder=args.thumbnailFolder,
                           projectDir=args.projectDir,
                           file_extension=args.file_extension,
                           verbose=args.verbose,
                           TruePos=args.TruePos,
                           NegToPos=args.NegToPos, 
                           TrueNeg=args.TrueNeg,
                           PosToNeg=args.PosToNeg)
