# -*- coding: utf-8 -*-
#Created on Wed Aug 31 15:08:38 2022
#@author: Ajit Johnson Nirmal
#When LocalNorm is used to curate the thumbnails, use this function to clone the curation on the real files

"""
!!! abstract "Short Description"
    The purpose of the `cloneFolder` function is to copy user actions from one 
    folder to another. For example, if a user manually arranges thumbnails in 
    the `localNorm` folder, this function can replicate those changes to the 
    raw thumbnails.


## Function
"""

# lib
import pathlib
from os import walk
import os
import shutil
import argparse

# function
def cloneFolder (copyFolder, 
                 applyFolder,
                 TruePos='TruePos', 
                 TrueNeg='TrueNeg',
                 PosToNeg='PosToNeg', 
                 NegToPos='NegToPos',
                 verbose=True):
    """
Parameters:
    copyFolder (list):
        List of folders from which the user wants to replicate the file structure.

    applyFolder (list):
        List of folders where the replicated file structure should be applied,
        in the same order as the `copyFolder` list.

    TruePos (str, optional):
        Name of the folder that holds the Thumbnails classified as True Positive.

    TrueNeg (str, optional):
        Name of the folder that holds the Thumbnails classified as True Negative.

    PosToNeg (str, optional):
        Name of the folder that holds the Thumbnails that were moved from `True Positive`
        to `True Negative`.

    NegToPos (str, optional):
        Name of the folder that holds the Thumbnails that were moved from `True Negative`
        to `True Positive`.

    verbose (bool, optional):
        If True, print detailed information about the process to the console.  

Returns:
    folder (cloned folders):  
        The file structure of the source Folder is replicated in the destination Folder.

Example:
        ```python
        
        # High level working directory
        projectDir = '/Users/aj/Documents/cspotExampleData'
        
        # list of folders to copy settings from
        copyFolder = [projectDir + '/CSPOT/Thumbnails/localNorm/CD3D',
                      projectDir + '/CSPOT/Thumbnails/localNorm/ECAD']
        # list of folders to apply setting to
        applyFolder = [projectDir + '/CSPOT/Thumbnails/CD3D',
                       projectDir + '/CSPOT/Thumbnails/ECAD']
        # note: Every copyFolder should have a corresponding applyFolder. The order matters! 
        
        # The function accepts the four pre-defined folders. If you had renamed them, please change it using the parameter below.
        cs.cloneFolder (copyFolder, 
                        applyFolder, 
                        TruePos='TruePos', TrueNeg='TrueNeg', 
                        PosToNeg='PosToNeg', NegToPos='NegToPos')
        
                
        # Same function if the user wants to run it via Command Line Interface
        python cloneFolder.py \
            --copyFolder /Users/aj/Desktop/cspotExampleData/CSPOT/Thumbnails/localNorm/CD3D /Users/aj/Desktop/cspotExampleData/CSPOT/Thumbnails/localNorm/ECAD \
            --applyFolder /Users/aj/Desktop/cspotExampleData/CSPOT/Thumbnails/CD3D /Users/aj/Desktop/cspotExampleData/CSPOT/Thumbnails/ECAD
        
        ```

    """

    #TruePos='TruePos'; TrueNeg='TrueNeg'; PosToNeg='PosToNeg'; NegToPos='NegToPos'

    # Convert the path to list
    if isinstance (copyFolder, str):
        copyFolder = [copyFolder]
    if isinstance (applyFolder, str):
        applyFolder = [applyFolder]

    # Quick Check!
    if len(copyFolder) is not len(applyFolder):
        raise ValueError('The number of copyFolder and applyFolder should match, please check!' )

    # function to delete images
    def deleteFile(files, location):
        for f in files:
            # full path
            #full_path = location  + f
            full_path = pathlib.Path.joinpath(location, f)
            if os.path.exists(full_path):
                os.remove(full_path)

    # Function to move images
    def moveFile(files, from_loc, to_loc):
        for f in files:
            # full path
            full_path_from = pathlib.Path.joinpath(from_loc, f) # from_loc + f
            full_path_to = pathlib.Path.joinpath(to_loc, f) # to_loc + f
            # move file
            if os.path.exists(full_path_from):
                shutil.move(full_path_from, full_path_to)

    # path lib of all folder
    all_folders = [pathlib.Path(p) for p in copyFolder]

    # copy from location
    pos_aug_location = [pathlib.Path(p + '/' + str(TruePos)) for p in copyFolder]
    neg_aug_location = [pathlib.Path(p + '/' + str(TrueNeg)) for p in copyFolder]
    pos2neg_aug_location = [pathlib.Path(p + '/' + str(PosToNeg)) for p in copyFolder]
    neg2pos_aug_location = [pathlib.Path(p + '/' + str(NegToPos)) for p in copyFolder]

    # copy to location
    pos_real_location = [pathlib.Path(p + '/' + str(TruePos)) for p in applyFolder]
    neg_real_location = [pathlib.Path(p + '/' + str(TrueNeg)) for p in applyFolder]
    pos2neg_real_location = [pathlib.Path(p + '/' + str(PosToNeg)) for p in applyFolder]
    neg2pos_real_location = [pathlib.Path(p + '/' + str(NegToPos)) for p in applyFolder]



    # function
    def processFolder (folderIndex):
        if verbose is True:
            print ('Processing: ' + str(all_folders[folderIndex].stem))

        # create a list of all file names in the applyFolder
        pos_files = next(walk(pos_real_location[folderIndex]), (None, None, []))[2]
        neg_files = next(walk(neg_real_location[folderIndex]), (None, None, []))[2]

        # Find file names within each of the copyFolder
        pos = next(walk(pos_aug_location[folderIndex]), (None, None, []))[2]
        neg = next(walk(neg_aug_location[folderIndex]), (None, None, []))[2]
        pos2neg = next(walk(pos2neg_aug_location[folderIndex]), (None, None, []))[2]
        neg2pos = next(walk(neg2pos_aug_location[folderIndex]), (None, None, []))[2]  # [] if no file

        # Find images to delete
        pos_del = list(set(pos_files).difference(pos + pos2neg))
        neg_del = list(set(neg_files).difference(neg + neg2pos))

        # delete files
        deleteFile(files=pos_del, location=pos_real_location[folderIndex])
        deleteFile(files=neg_del, location=neg_real_location[folderIndex])

        # move files
        moveFile (files=pos2neg, from_loc=pos_real_location[folderIndex], to_loc=pos2neg_real_location[folderIndex])
        moveFile (files=neg2pos, from_loc=neg_real_location[folderIndex], to_loc=neg2pos_real_location[folderIndex])

        # print the number of files
        posaug = len(next(walk(pos_aug_location[folderIndex]), (None, None, []))[2])
        posreal = len(next(walk(pos_real_location[folderIndex]), (None, None, []))[2])
        negaug = len(next(walk(neg_aug_location[folderIndex]), (None, None, []))[2])
        negreal = len(next(walk(neg_real_location[folderIndex]), (None, None, []))[2])
        postonegaug = len(next(walk(pos2neg_aug_location[folderIndex]), (None, None, []))[2])
        postonegreal = len(next(walk(pos2neg_real_location[folderIndex]), (None, None, []))[2])
        negtoposaug = len(next(walk(neg2pos_aug_location[folderIndex]), (None, None, []))[2])
        negtoposreal = len(next(walk(neg2pos_real_location[folderIndex]), (None, None, []))[2])

        #print ('No of Files in TruePos-> copyFolder: ' + str(posaug) + ' ; applyFolder: '+ str(posreal))
        #print ('No of Files in TrueNeg-> copyFolder: ' + str(negaug) + ' ; applyFolder: '+ str(negreal))
        #print ('No of Files in PosToNeg-> copyFolder: ' + str(postonegaug) + ' ; applyFolder: '+ str(postonegreal))
        #print ('No of Files in NegToPos-> copyFolder: ' + str(negtoposaug) + ' ; applyFolder: '+ str(negtoposreal))

    # apply function to all folders
    r_processFolder = lambda x: processFolder (folderIndex=x)
    process_folders = list(map(r_processFolder, list(range(len(copyFolder)))))

    # Finish Job
    if verbose is True:
        print('Cloning Folder is complete, head over to /CSPOT/Thumbnails" to view results')

# Make the Function CLI compatable
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clone folder file structure')
    parser.add_argument('--copyFolder', metavar='copyFolder', type=str, nargs='+', help='List of folders from which to replicate the file structure')
    parser.add_argument('--applyFolder', metavar='applyFolder', type=str, nargs='+', help='List of folders where the replicated file structure should be applied, in the same order as the copyFolder list')
    parser.add_argument('--TruePos', dest='TruePos', type=str, default='TruePos', help='Name of the folder that holds the Thumbnails classified as True Positive')
    parser.add_argument('--TrueNeg', dest='TrueNeg', type=str, default='TrueNeg', help='Name of the folder that holds the Thumbnails classified as True Negative')
    parser.add_argument('--PosToNeg', dest='PosToNeg', type=str, default='PosToNeg', help='Name of the folder that holds the Thumbnails that were moved from True Positive to True Negative')
    parser.add_argument('--NegToPos', dest='NegToPos', type=str, default='NegToPos', help='Name of the folder that holds the Thumbnails that were moved from True Negative to True Positive')
    parser.add_argument("--verbose", type=bool, default=True, help="If True, print detailed information about the process to the console.")   
    args = parser.parse_args()
    cloneFolder(copyFolder=args.copyFolder,
                applyFolder=args.applyFolder,
                TruePos=args.TruePos,
                TrueNeg=args.TrueNeg,
                PosToNeg=args.PosToNeg,
                NegToPos=args.NegToPos,
                verbose=args.verbose)
