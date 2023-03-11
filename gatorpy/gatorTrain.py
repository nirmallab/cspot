

"""
!!! abstract "Short Description"
    The function trains a deep learning model for each marker in the provided 
    training data. To train the `gatorModel`, simply direct the function to the 
    `TrainingData` folder. To train only specific models, specify the folder names 
    using the `trainMarkers` parameter. The `projectDir` remains constant and the 
    program will automatically create subfolders to save the trained models.
  

## Function
"""


# Libs
import argparse
import pathlib
if __name__ == '__main__':
    from UNet import *
else:
    from .UNet import UNet2D

# Function
def gatorTrain(trainingDataPath,
               projectDir,
               trainMarkers=None,
               artefactPath=None,
               imSize=64,
               nChannels=1,
               nClasses=2,
               nExtraConvs=0,
               nLayers=3,
               featMapsFact=2,
               downSampFact=2,
               ks=3,
               nOut0=16,
               stdDev0=0.03,
               batchSize=16,
               epochs=100,
               verbose=True):
    """

Parameters:
    trainingDataPath (str):
        The file path leading to the directory that holds the training data.
    
    projectDir (str):
        Path to output directory. The result will be located at `projectDir/GATOR/gatorModel/`.
    
    trainMarkers (list):
        Generate models for a specified list of markers. By default, models are c
        reated for all data in the TrainingData folder. If the user wants to
        limit it to a specific list, they can pass in the folder names (e.g. ['CD3D', 'CD4'])
    
    artefactPath (str):
        Path to the directory where the artefacts data is loaded from.
    
    imSize (int, optional):
        Image size (assumed to be square).
    
    nChannels (int, optional):
        Number of channels in the input image.
    
    nClasses (int, optional):
        Number of classes in the classification problem.
    
    nExtraConvs (int, optional):
        Number of extra convolutional layers to add to the model.
    
    nLayers (int, optional):
        Total number of layers in the model.
    
    featMapsFact (int, optional):
        Factor to multiply the number of feature maps by in each layer.
    
    downSampFact (int, optional):
        Factor to down-sample the feature maps by in each layer.
    
    ks (int, optional):
        Kernel size for the convolutional layers.
    
    nOut0 (int, optional):
        Number of filters in the first layer.
    
    stdDev0 (float, optional):
        Standard deviation for the initializer for the first layer.
    
    batchSize (int, optional):
        Batch size for training.
    
    epochs (int, optional):
        Number of training epochs.

    verbose (bool, optional):
        If True, print detailed information about the process to the console.  

Returns:
    
    Model (images and model):  
        The result will be located at `projectDir/GATOR/gatorModel/`.
    

Example:
    
    ```python
    
    # set the working directory & set paths to the example data
    cwd = '/Users/aj/Desktop/gatorExampleData'
    trainingDataPath = cwd + '/GATOR/TrainingData'
    projectDir = cwd
    
    # Run the Function
    ga.gatorTrain(trainingDataPath=trainingDataPath,
                   projectDir=projectDir,
                   trainMarkers=None,
                   artefactPath=None,
                   imSize=64,
                   nChannels=1,
                   nClasses=2,
                   nExtraConvs=0,
                   nLayers=3,
                   featMapsFact=2,
                   downSampFact=2,
                   ks=3,
                   nOut0=16,
                   stdDev0=0.03,
                   batchSize=16,
                   epochs=1)
    
    # Same function if the user wants to run it via Command Line Interface
    python gatorTrain.py --trainingDataPath /Users/aj/Desktop/gatorExampleData/GATOR/TrainingData --projectDir /Users/aj/Desktop/gatorExampleData/ --epochs 1
    
    ```


    """

    # Start here
    # convert to path
    trainingDataPath = pathlib.Path(trainingDataPath)
    # identify all the data folders within the given TrainingData folder
    directories = [x for x in trainingDataPath.iterdir() if x.is_dir()]
    # keep only folders that the user have requested
    if trainMarkers is not None:
        if isinstance(trainMarkers, str):
            trainMarkers = [trainMarkers]
        directories = [x for x in directories if x.stem in trainMarkers]

    # optional artifacts
    if artefactPath is not None:
        artefactPath = pathlib.Path(artefactPath)
        artefactTrainPath = pathlib.Path(artefactPath / 'training')
        artefactValidPath = pathlib.Path(artefactPath / 'validation')
    else:
        artefactPath = ''
        artefactTrainPath = ''
        artefactValidPath = ''
    # Need to run the training for each marker

    def gatorTrainInternal(trainingDataPath,
                           projectDir,
                           artefactPath,
                           imSize,
                           nChannels,
                           nClasses,
                           nExtraConvs,
                           nLayers,
                           featMapsFact,
                           downSampFact,
                           ks,
                           nOut0,
                           stdDev0,
                           batchSize,
                           epochs):
        # process the file name
        finalName = trainingDataPath.stem
        
        # paths for loading data
        trainPath = pathlib.Path(trainingDataPath / 'training')
        validPath = pathlib.Path(trainingDataPath / 'validation')
        testPath = pathlib.Path(trainingDataPath / 'test')

        # Paths for saving data
        logPath = pathlib.Path(
            projectDir + '/GATOR/gatorTrain/' + finalName + '/tempTFLogs/')
        modelPath = pathlib.Path(projectDir + '/GATOR/gatorModel/' + finalName)
        pmPath = pathlib.Path(projectDir + '/GATOR/gatorTrain/' +
                              finalName + '/TFprobMaps/')

        # set up the model
        UNet2D.setup(imSize=imSize,
                     nClasses=nClasses,
                     nChannels=nChannels,
                     nExtraConvs=nExtraConvs,
                     nDownSampLayers=nLayers,
                     featMapsFact=featMapsFact,
                     downSampFact=downSampFact,
                     kernelSize=ks,
                     nOut0=nOut0,
                     stdDev0=stdDev0,
                     batchSize=batchSize)

        # train the model
        UNet2D.train(trainPath=trainPath,
                     validPath=validPath,
                     testPath=testPath,
                     artTrainPath=artefactTrainPath,
                     artValidPath=artefactValidPath,
                     logPath=logPath,
                     modelPath=modelPath,
                     pmPath=pmPath,
                     restoreVariables=False,
                     nSteps=epochs,
                     gpuIndex=0,
                     testPMIndex=2)

    # Run the function on all markers
    def r_gatorTrainInternal(x): return gatorTrainInternal(trainingDataPath=x,
                                                           projectDir=projectDir,
                                                           artefactPath=artefactPath,
                                                           imSize=imSize,
                                                           nChannels=nChannels,
                                                           nClasses=nClasses,
                                                           nExtraConvs=nExtraConvs,
                                                           nLayers=nLayers,
                                                           featMapsFact=featMapsFact,
                                                           downSampFact=downSampFact,
                                                           ks=ks,
                                                           nOut0=nOut0,
                                                           stdDev0=stdDev0,
                                                           batchSize=batchSize,
                                                           epochs=epochs)

    gatorTrainInternal_result = list(
        map(r_gatorTrainInternal,  directories))  # Apply function
    
    # Finish Job
    if verbose is True:
        print('Gator Models have been generated, head over to "' + str(projectDir) + '/GATOR/gatorModel" to view results')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a deep learning model using data for each marker, stored in separate directories.')
    parser.add_argument('--trainingDataPath', type=str, help='The file path leading to the directory that holds the training data for each marker.')
    parser.add_argument('--projectDir', type=str, help='Path to output directory.')
    parser.add_argument("--trainMarkers", type=str, nargs="*", default=None, help="Generate models for a specified list of markers.")
    parser.add_argument('--artefactPath', type=str, help='Path to the directory where the artefacts data is loaded from.', default=None)
    parser.add_argument('--imSize', type=int, help='Image size (assumed to be square).', default=64)
    parser.add_argument('--nChannels', type=int, help='Number of channels in the input image.', default=1)
    parser.add_argument('--nClasses', type=int, help='Number of classes in the classification problem.', default=2)
    parser.add_argument('--nExtraConvs', type=int, help='Number of extra convolutional layers to add to the model.', default=0)
    parser.add_argument('--nLayers', type=int, help='Total number of layers in the model.', default=3)
    parser.add_argument('--featMapsFact', type=int, help='Factor to multiply the number of feature maps by in each layer.', default=2)
    parser.add_argument('--downSampFact', type=int, help='Factor to down-sample the feature maps by in each layer.', default=2)
    parser.add_argument('--ks', type=int, help='Kernel size for the convolutional layers.', default=3)
    parser.add_argument('--nOut0', type=int, help='Number of filters in the first layer.', default=16)
    parser.add_argument('--stdDev0', type=float, help='Standard deviation for the initializer for the first layer.', default=0.03)
    parser.add_argument('--batchSize', type=int, help='Batch size for training.', default=16)
    parser.add_argument('--epochs', type=int, help='Number of training epochs.', default=100)
    parser.add_argument("--verbose", type=bool, default=True, help="If True, print detailed information about the process to the console.")    

    args = parser.parse_args()
    gatorTrain(trainingDataPath=args.trainingDataPath,
               projectDir=args.projectDir,
               trainMarkers=args.trainMarkers,
               artefactPath=args.artefactPath,
               imSize=args.imSize,
               nChannels=args.nChannels,
               nClasses=args.nClasses,
               nExtraConvs=args.nExtraConvs,
               nLayers=args.nLayers,
               featMapsFact=args.featMapsFact,
               downSampFact=args.downSampFact,
               ks=args.ks,
               nOut0=args.nOut0,
               stdDev0=args.stdDev0,
               batchSize=args.batchSize,
               epochs=args.epochs,
               verbose=args.verbose)
