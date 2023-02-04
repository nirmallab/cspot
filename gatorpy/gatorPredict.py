
# Libs
import os, argparse
import pandas as pd
import pathlib

# tools libs
from skimage import io as skio
import tensorflow.compat.v1 as tf
import tifffile
import numpy as np
from skimage.transform import resize

# from other .py scripts
if __name__ == '__main__':
    from toolbox.imtools import im2double
    from UNet import *
else:
    from .toolbox.imtools import im2double
    from .UNet import UNet2D

# Function
def gatorPredict(imagePath,
                 gatorModelPath,
                 outputDir, 
                 markerChannelMapPath, 
                 markerColumnName='marker', 
                 channelColumnName='channel', 
                 modelColumnName='gatormodel', 
                 GPU=0):
    
    """
Parameters:
    imagePath (str):  
        The path to the .tif file that needs to be processed. 
     
    gatorModelPath (str):  
        The path to the `gatorModel` folder. 
    
    outputDir (str):  
        The path to the output directory where the processed images will be saved.
     
    markerChannelPath (str, optional):  
        The path to the marker panel list, which contains information about the markers used in the image. This argument is required.
     
    markerColumnName (str, optional):  
        The name of the column in the marker panel list that contains the marker names. The default value is 'marker'.
     
    channelColumnName (str, optional):  
        The name of the column in the marker panel list that contains the channel names. The default value is 'channel'.
     
    modelColumnName (str, optional):  
        The name of the column in the marker panel list that contains the model names. The default value is 'gatormodel'.
     
    GPU (int, optional):  
        An optional argument to explicitly select the GPU to use. The default value is -1, meaning that the GPU will be selected automatically.

Returns:
    Predicted Probability Masks (images):  

Example:

	```python
	
    
	```
     
     """
    
    fileName =pathlib.Path(imagePath).stem

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


    # identify the gator model column name (doing this to make it easier for people who confuse between gatormodel and gatormodels)
    if modelColumnName not in columnnames:
        if modelColumnName != 'gatormodel':
            raise ValueError('modelColumnName not found in markerChannelMap, please check')
        if 'gatormodels' in columnnames:
            channelCol = 'gatormodels'
        else:
            raise ValueError('modelColumnName not found in markerChannelMap, please check')
    else:
        modelCol = modelColumnName

    # remove rowa that have nans in modelCol
    runMenu = maper.dropna(subset=[modelCol], inplace=False)[[channelCol,markerCol,modelCol]]

    # shortcuts
    numMarkers = len(runMenu)
    len(runMenu.columns)

    I = skio.imread(imagePath, img_num=0, plugin='tifffile')


    probPath = pathlib.Path(outputDir + '/GATOR/gatorPredict/')
    modelPath = pathlib.Path(gatorModelPath)

    if not os.path.exists(probPath):
        os.makedirs(probPath,exist_ok=True)




    def data(runMenu, 
             imagePath, 
             modelPath, 
             outputDir, 
             dsFactor=1, 
             GPU=0):
        
        # Loop through the rows of the DataFrame
        for index, row in runMenu.iterrows():
            channel = row[channelColumnName]
            markerName = row[markerColumnName]
            gatormodel = row[modelColumnName]
            print('Running gator model ' + str(gatormodel) + ' on channel ' + str(channel) + ' corresponding to marker ' + str(markerName) )


            tf.reset_default_graph()
            UNet2D.singleImageInferenceSetup(pathlib.Path(modelPath / gatormodel), GPU, -1, -1)

            fileName = os.path.basename(imagePath)
            fileNamePrefix = fileName.split(os.extsep, 1)
            fileType = fileNamePrefix[1]
            if fileType == 'ome.tif' or fileType == 'ome.tiff' or fileType == 'btf':
                I = skio.imread(imagePath, img_num=int(channel-1), plugin='tifffile')
            elif fileType == 'tif':
                I = tifffile.imread(imagePath, key=int(channel-1))

            if I.dtype == 'float32':
                I=np.uint16(I)
            rawVert = I.shape[0]
            rawHorz = I.shape[1]
            rawI = I

            hsize = int(float(rawVert * float(dsFactor)))
            vsize = int(float(rawHorz * float(dsFactor)))
            # I = resize(I, (hsize, vsize))
            cells = I
            maxLimit = np.max(I)
            I=I/65535*255

            rawI = im2double(rawI) / np.max(im2double(rawI))

            append_kwargs = {
                'bigtiff': True,
                'metadata': None,
                'append': True,
            }
            save_kwargs = {
                'bigtiff': True,
                'metadata': None,
                'append': False,
            }

            PM = np.uint8(255 * UNet2D.singleImageInference(I, 'accumulate',1))
            PM = resize(PM, (rawVert, rawHorz))
            yield np.uint8(255 * PM)

    with tifffile.TiffWriter(probPath / (fileName + '_GatorOutput.ome.tif')) as tiff:
        tiff.write(data(runMenu, imagePath, modelPath, probPath, dsFactor=1, GPU=0), shape=(numMarkers,I.shape[0],I.shape[1]), dtype='uint8', metadata={'Channel': {'Name': runMenu.marker.tolist()}, 'axes': 'CYX'})

        UNet2D.singleImageInferenceCleanup()


# =============================================================================
#     logPath = ''
#     scriptPath = os.path.dirname(os.path.realpath(__file__))
# 
#     pmPath = ''
# 
#     if os.system('nvidia-smi') == 0:
#         if args.GPU == -1:
#             print("automatically choosing GPU")
#             GPU = GPUselect.pick_gpu_lowest_memory()
#         else:
#             GPU = args.GPU
#         print('Using GPU ' + str(GPU))
# 
#     else:
#         if sys.platform == 'win32':  # only 1 gpu on windows
#             if args.GPU == -1:
#                 GPU = 0
#                 print('using default GPU')
#             else:
#                 GPU = args.GPU
#             print('Using GPU ' + str(GPU))
#         else:
#             GPU = 0
#             print('Using CPU')
#     os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % GPU
# =============================================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gator Predict function')
    parser.add_argument('--imagePath', type=str, required=True, help='The path to the .tif file that needs to be processed.')
    parser.add_argument('--gatorModelPath', type=str, required=True, help='The path to the `gatorModel` folder.')
    parser.add_argument('--outputDir', type=str, help='The path to the output directory where the processed images will be saved.')
    parser.add_argument('--markerChannelMapPath', type=str, required=True, help='The path to the marker panel list, which contains information about the markers used in the image.')
    parser.add_argument('--markerColumnName', type=str, default='marker', help='The name of the column in the marker panel list that contains the marker names. The default value is `marker`.')
    parser.add_argument('--channelColumnName', type=str, default='channel', help='The name of the column in the marker panel list that contains the channel names. The default value is `channel`.')
    parser.add_argument('--modelColumnName', type=str, default='gatormodel', help='The name of the column in the marker panel list that contains the model names. The default value is `gatormodel`.')
    parser.add_argument('--GPU', type=int, default=0, help='An optional argument to explicitly select the GPU to use. The default value is 0, meaning that the GPU will be selected automatically.')

    args = parser.parse_args()
    gatorPredict(imagePath=args.imagePath, 
                 gatorModelPath=args.gatorModelPath, 
                 outputDir=args.outputDir, 
                 markerChannelMapPath=args.markerChannelMapPath, 
                 markerColumnName=args.markerColumnName, 
                 channelColumnName=args.channelColumnName, 
                 modelColumnName=args.modelColumnName, 
                 GPU=args.GPU)






