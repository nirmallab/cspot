

# Libs
import argparse, pathlib
if __name__ == '__main__':
    from UNet import *
else:
    from .UNet import UNet2D

# Function
def gatorTrain ( trainingDataPath, 
    			 outputDir,
    			 artefactPath=None, 
    			 modelName=None, 
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
    			 epochs=100):
	"""
    
Parameters:
    trainingDataPath (str):  
        The file path leading to the directory that holds the training data for each marker.
		
    artefactPath (str):  
        Path to the directory where the artefacts data is loaded from.

    outputDir (str):  
        PatPath to output directory.
	
    modelName (str, optional):  
        Name of the model to be used. If None, the folder name will be used.
   
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

Returns:
    Deep learning model (images and model):  

Example:

	```python
	
	
	
	```

	"""

	# convert to path
	trainingDataPath = pathlib.Path(trainingDataPath)
	
	
	# resolve model name
	if modelName is None:
		finalName = trainingDataPath.stem
	else:
		finalName = modelName
		
	# optional artifacts
	if artefactPath is not None:
		artefactPath = pathlib.Path(artefactPath)
		artefactTrainPath = pathlib.Path(artefactPath / 'training')
		artefactValidPath = pathlib.Path(artefactPath / 'validation')
	else:
		artefactPath=''; artefactTrainPath=''; artefactValidPath=''

	# paths for loading data
	trainPath = pathlib.Path(trainingDataPath / 'training')
	validPath = pathlib.Path(trainingDataPath / 'validation')
	testPath = pathlib.Path(trainingDataPath / 'test')


	# Paths for saving data
	logPath = pathlib.Path(outputDir + '/GATOR/gatorTrain/' + finalName + '/tempTFLogs/' )
	modelPath = pathlib.Path(outputDir + '/GATOR/gatorModel/' + finalName)
	pmPath = pathlib.Path(outputDir + '/GATOR/gatorTrain/' + finalName + '/TFprobMaps/')
	
	# set up the model
	UNet2D.setup(   imSize=imSize,
					nClasses=nClasses,
					nChannels = nChannels,
					nExtraConvs=nExtraConvs,
					nDownSampLayers=nLayers,
					featMapsFact=featMapsFact,
					downSampFact=downSampFact,
					kernelSize=ks,
					nOut0=nOut0,
					stdDev0=stdDev0,
					batchSize=batchSize)  # 4 (Ki67 CD8a) , 3 (CD45 ecad CD4 CD3D)
		
	
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

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a deep learning model using data for each marker, stored in separate directories.')
	parser.add_argument('--trainingDataPath', type=str, help='The file path leading to the directory that holds the training data for each marker.')
	parser.add_argument('--outputDir', type=str, help='Path to output directory.')
	parser.add_argument('--artefactPath', type=str, help='Path to the directory where the artefacts data is loaded from.', default=None)
	parser.add_argument('--modelName', type=str, help='Name of the model to be used. If None, the folder name will be used.', default=None)
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
	args = parser.parse_args()
	gatorTrain( trainingDataPath=args.trainingDataPath,
    			artefactPath=args.artefactPath,
    			outputDir=args.outputDir,
    			modelName=args.modelName,
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
    			epochs=args.epochs)
			

			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			

