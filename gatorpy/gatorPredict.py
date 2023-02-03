
# Libs
import os, argparse
import pandas as pd
import pathlib

from UNet import *


# Function
def gatorPredict(imagePath, 
                 outputDir, 
                 markerChannelMapPath, 
                 markerColumnName='marker', 
                 channelColumnName='channel', 
                 modelColumnName='gatormodel', 
                 GPU = 0):
    
    """
     --imagePath (str): The path to the .tif file that needs to be processed. This argument is required.
    
     --outputDir (str): The path to the output directory where the processed images will be saved.
     
     --markerChannelPath (str, optional): The path to the marker panel list, which contains information about the markers used in the image. This argument is required.
     
     --markerColumnName (str, optional): The name of the column in the marker panel list that contains the marker names. The default value is 'marker'.
     
     --channelColumnName (str, optional): The name of the column in the marker panel list that contains the channel names. The default value is 'channel'.
     
     --modelColumnName (str, optional): The name of the column in the marker panel list that contains the model names. The default value is 'gatormodel'.
         
     --scalingFactor (int, optional): A factor by which the image size will be increased/decreased. The default value is 1, meaning no change in size.
     
     --GPU (int, optional): An optional argument to explicitly select the GPU to use. The default value is -1, meaning that the GPU will be selected automatically.
     
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
    modelPath = pathlib.Path(outputDir + '/GATOR/gatorModels/')

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", help="path to the .tif file")
    parser.add_argument("--markerChannelPath", help="path to the marker panel list")
    parser.add_argument("--markerColumnName", type=str, help="marker colulmn name in the marker panel list", default = 'marker')
    parser.add_argument("--channelColumnName", type=str, help="channel column name in the marker panel list", default = 'channel')
    parser.add_argument("--modelColumnName", type=str, help="model column name in the marker panel list", default = 'gatormodel')
    parser.add_argument('--outputDir', type=str, help='Path to output directory.')
    parser.add_argument("--scalingFactor", help="factor by which to increase/decrease image size by", type=float,
                        default=1)
    parser.add_argument("--GPU", help="explicitly select GPU", type=int, default=-1)
    args = parser.parse_args()



    logPath = ''
    scriptPath = os.path.dirname(os.path.realpath(__file__))

    pmPath = ''

    if os.system('nvidia-smi') == 0:
        if args.GPU == -1:
            print("automatically choosing GPU")
            GPU = GPUselect.pick_gpu_lowest_memory()
        else:
            GPU = args.GPU
        print('Using GPU ' + str(GPU))

    else:
        if sys.platform == 'win32':  # only 1 gpu on windows
            if args.GPU == -1:
                GPU = 0
                print('using default GPU')
            else:
                GPU = args.GPU
            print('Using GPU ' + str(GPU))
        else:
            GPU = 0
            print('Using CPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % GPU


    gatorPredict(imagePath = args.imagePath,
              markerChannelMapPath=args.markerChannelPath,
              outputDir=args.outputDir,
              markerColumnName=args.markerColumnName,
              channelColumnName=args.channelColumnName,
              modelColumnName=args.modelColumnName,
              GPU=args.GPU)
