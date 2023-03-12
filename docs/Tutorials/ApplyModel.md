# 🐊 Run the GATOR Prediction Algorithm on new images

**Download the [executable notebook](https://github.com/nirmalLab/gatorpy/blob/main/docs/Tutorials/notebooks/ApplyModel.ipynb) and [trained models]().**  
For the purpose of this tutorial, replace the `gatorModel` folder within the exemplar data with the newly downloaded `gatorModel` directory. 
  
Make sure you have completed `BuildModels` Tutorial before you try to execute this Jupyter Notebook!
  
Please keep in mind that the sample data is used for demonstration purposes only and has been simplified and reduced in size. It is solely intended for educational purposes on how to execute `Gator` and will not yeild any meaningful results.


**Running the Gator Prediction Algorithm involves the following steps:**
- Run the `gatorPredict` function on a new image. It will produce an image with probability masks
- Run the `generateGatorScore` function on the probability masks to generate the `gatorScores`
- Run the `gatorObject` to create an anndata object with the `gatorScores` and pre-computed `single-cell table`
- Lastly, run `gator`  on the gatorObject
  
**Note: To make things easy, all of the above steps can be run with a single command `gatorPipeline`.**  
Typically, in production settings, `gatorPipeline` would be utilized, whereas step-by-step analysis would be employed for troubleshooting, model validation, and similar tasks that necessitate greater granularity or control.


```python
# import packages in jupyter notebook (not needed for command line interface users)
import gatorpy as ga
```


```python
# set the working directory & set paths to the example data
cwd = '/Users/aj/Desktop/gatorExampleData'

# Apply gatorModels
imagePath = cwd + '/image/exampleImage.tif'
markerChannelMapPath = cwd + '/markers.csv'
gatorModelPath = cwd + '/GATOR/gatorModel/'

# Generate gatorScores
probabilityMaskPath = cwd + '/GATOR/gatorPredict/exampleImage_gatorPredict.ome.tif'
segmentationPath = cwd + '/segmentation/exampleSegmentationMask.tif'
```

## Run the Gator Pipeline


```python
# Manadatory parameters for running the gator pipeline
projectDir = '/Users/aj/Desktop/gatorExampleData'
imagePath = projectDir + '/image/exampleImage.tif'
markerChannelMapPath = projectDir + '/markers.csv'
gatorModelPath = projectDir + '/GATOR/gatorModel/'
segmentationPath = projectDir + '/segmentation/exampleSegmentationMask.tif'
spatialTablePath = projectDir + '/quantification/exampleSpatialTable.csv'

# Run the pipeline
ga.gatorPipeline(   
                    # parameters for gatorPredict
                    imagePath=imagePath,
                    gatorModelPath=gatorModelPath,
                    markerChannelMapPath=markerChannelMapPath,
                    # parameters for generateGatorScore
                    segmentationMaskPath=segmentationPath,
                    # parameters for gatorObject
                    spatialTablePath=spatialTablePath,
                    # parameters to run gator
                    # ..
                    # common parameters
                    verbose=False,
                    projectDir=projectDir)

```

    loading data
    WARNING:tensorflow:From /opt/anaconda3/envs/gator/lib/python3.9/site-packages/keras/layers/normalization/batch_normalization.py:561: _colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.


    /Users/aj/Dropbox (Partners HealthCare)/nirmal lab/softwares/gatorpy/gatorpy/UNet.py:137: UserWarning: `tf.layers.batch_normalization` is deprecated and will be removed in a future version. Please use `tf.keras.layers.BatchNormalization` instead. In particular, `tf.control_dependencies(tf.GraphKeys.UPDATE_OPS)` should not be used (consult the `tf.keras.layers.BatchNormalization` documentation).
      bn = tf.nn.leaky_relu(tf.layers.batch_normalization(c00+shortcut, training=UNet2D.tfTraining))
    /Users/aj/Dropbox (Partners HealthCare)/nirmal lab/softwares/gatorpy/gatorpy/UNet.py:159: UserWarning: `tf.layers.batch_normalization` is deprecated and will be removed in a future version. Please use `tf.keras.layers.BatchNormalization` instead. In particular, `tf.control_dependencies(tf.GraphKeys.UPDATE_OPS)` should not be used (consult the `tf.keras.layers.BatchNormalization` documentation).
      lbn = tf.nn.leaky_relu(tf.layers.batch_normalization(
    /Users/aj/Dropbox (Partners HealthCare)/nirmal lab/softwares/gatorpy/gatorpy/UNet.py:162: UserWarning: `tf.layers.dropout` is deprecated and will be removed in a future version. Please use `tf.keras.layers.Dropout` instead.
      return tf.layers.dropout(lbn, 0.15, training=UNet2D.tfTraining)
    /Users/aj/Dropbox (Partners HealthCare)/nirmal lab/softwares/gatorpy/gatorpy/UNet.py:224: UserWarning: `tf.layers.batch_normalization` is deprecated and will be removed in a future version. Please use `tf.keras.layers.BatchNormalization` instead. In particular, `tf.control_dependencies(tf.GraphKeys.UPDATE_OPS)` should not be used (consult the `tf.keras.layers.BatchNormalization` documentation).
      tf.layers.batch_normalization(tf.nn.conv2d(cc, luXWeights2, strides=[1, 1, 1, 1], padding='SAME'),


    loading data
    loading data
    0
    1
    INFO:tensorflow:Restoring parameters from /Users/aj/Desktop/gatorExampleData/GATOR/gatorModel/ECAD/model.ckpt


    /Users/aj/Dropbox (Partners HealthCare)/nirmal lab/softwares/gatorpy/gatorpy/UNet.py:245: UserWarning: `tf.layers.batch_normalization` is deprecated and will be removed in a future version. Please use `tf.keras.layers.BatchNormalization` instead. In particular, `tf.control_dependencies(tf.GraphKeys.UPDATE_OPS)` should not be used (consult the `tf.keras.layers.BatchNormalization` documentation).
      return tf.layers.batch_normalization(


    Model restored.
    Inference...
    loading data
    loading data
    loading data
    0
    1
    INFO:tensorflow:Restoring parameters from /Users/aj/Desktop/gatorExampleData/GATOR/gatorModel/CD3D/model.ckpt
    Model restored.
    Inference...


    /Users/aj/Dropbox (Partners HealthCare)/nirmal lab/softwares/gatorpy/gatorpy/gator.py:390: RuntimeWarning: invalid value encountered in divide
      below_midpoint = (below_midpoint - min_below) / range_below


**Same function if the user wants to run it via Command Line Interface**
```
python gatorPipeline.py \
        --imagePath /Users/aj/Desktop/gatorExampleData/image/exampleImage.tif \
        --gatorModelPath /Users/aj/Desktop/gatorExampleData/GATOR/gatorModel/ \
        --markerChannelMapPath /Users/aj/Desktop/gatorExampleData/markers.csv \
        --segmentationMaskPath /Users/aj/Desktop/gatorExampleData/segmentation/exampleSegmentationMask.tif \
        --spatialTablePath /Users/aj/Desktop/gatorExampleData/quantification/exampleSpatialTable.csv \
        --projectDir /Users/aj/Desktop/gatorExampleData \
        --verbose True
```

## Step-1: Apply the generated Models on the Image of interest (Pixel Level)

The function `gatorPredict` is employed to make predictions about the expression of a specified marker on cells in new images using the models generated by `gatorTrain`. This calculation is done at the pixel level, resulting in an output image where the number of channels corresponds to the number of models applied to the input image. The parameter `markerChannelMapPath` is used to associate the image channel number with the relevant model to be applied.


```python
outputDir = cwd

ga.gatorPredict( imagePath=imagePath,
                 gatorModelPath=gatorModelPath,
                 outputDir=outputDir, 
                 markerChannelMapPath=markerChannelMapPath, 
                 markerColumnName='marker', 
                 channelColumnName='channel', 
                 modelColumnName='gatormodel', 
                 GPU=-1)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    /Users/aj/Dropbox (Partners HealthCare)/nirmal lab/softwares/gatorpy/docs/Tutorials/notebooks/ApplyModel.ipynb Cell 8 in <cell line: 3>()
          <a href='vscode-notebook-cell:/Users/aj/Dropbox%20%28Partners%20HealthCare%29/nirmal%20lab/softwares/gatorpy/docs/Tutorials/notebooks/ApplyModel.ipynb#X10sZmlsZQ%3D%3D?line=0'>1</a> outputDir = cwd
    ----> <a href='vscode-notebook-cell:/Users/aj/Dropbox%20%28Partners%20HealthCare%29/nirmal%20lab/softwares/gatorpy/docs/Tutorials/notebooks/ApplyModel.ipynb#X10sZmlsZQ%3D%3D?line=2'>3</a> ga.gatorPredict( imagePath=imagePath,
          <a href='vscode-notebook-cell:/Users/aj/Dropbox%20%28Partners%20HealthCare%29/nirmal%20lab/softwares/gatorpy/docs/Tutorials/notebooks/ApplyModel.ipynb#X10sZmlsZQ%3D%3D?line=3'>4</a>                  gatorModelPath=gatorModelPath,
          <a href='vscode-notebook-cell:/Users/aj/Dropbox%20%28Partners%20HealthCare%29/nirmal%20lab/softwares/gatorpy/docs/Tutorials/notebooks/ApplyModel.ipynb#X10sZmlsZQ%3D%3D?line=4'>5</a>                  outputDir=outputDir, 
          <a href='vscode-notebook-cell:/Users/aj/Dropbox%20%28Partners%20HealthCare%29/nirmal%20lab/softwares/gatorpy/docs/Tutorials/notebooks/ApplyModel.ipynb#X10sZmlsZQ%3D%3D?line=5'>6</a>                  markerChannelMapPath=markerChannelMapPath, 
          <a href='vscode-notebook-cell:/Users/aj/Dropbox%20%28Partners%20HealthCare%29/nirmal%20lab/softwares/gatorpy/docs/Tutorials/notebooks/ApplyModel.ipynb#X10sZmlsZQ%3D%3D?line=6'>7</a>                  markerColumnName='marker', 
          <a href='vscode-notebook-cell:/Users/aj/Dropbox%20%28Partners%20HealthCare%29/nirmal%20lab/softwares/gatorpy/docs/Tutorials/notebooks/ApplyModel.ipynb#X10sZmlsZQ%3D%3D?line=7'>8</a>                  channelColumnName='channel', 
          <a href='vscode-notebook-cell:/Users/aj/Dropbox%20%28Partners%20HealthCare%29/nirmal%20lab/softwares/gatorpy/docs/Tutorials/notebooks/ApplyModel.ipynb#X10sZmlsZQ%3D%3D?line=8'>9</a>                  modelColumnName='gatormodel', 
         <a href='vscode-notebook-cell:/Users/aj/Dropbox%20%28Partners%20HealthCare%29/nirmal%20lab/softwares/gatorpy/docs/Tutorials/notebooks/ApplyModel.ipynb#X10sZmlsZQ%3D%3D?line=9'>10</a>                  GPU=-1)


    TypeError: gatorPredict() got an unexpected keyword argument 'outputDir'


**Same function if the user wants to run it via Command Line Interface**
```
python gatorPredict.py --imagePath /Users/aj/Desktop/gatorExampleData/image/exampleImage.tif --gatorModelPath /Users/aj/Desktop/gatorExampleData/GATOR/gatorModel/ --outputDir /Users/aj/Desktop/gatorExampleData --markerChannelMapPath /Users/aj/Desktop/gatorExampleData/markers.csv
```

## Step-2: Calculate the gatorScore (Single-cell Level)

After calculating pixel-level probability scores, the next step is to aggregate them to the single-cell level. This can be done by computing the mean or median probability scores using pre-computed segmentation masks. The marker names, if available, should already be included in the probabilityMask image. If the marker names are lost due to file manipulation, the user can provide them through the markerNames parameter.


```python
ga.generateGatorScore(probabilityMaskPath=probabilityMaskPath,
                      segmentationMaskPath=segmentationPath,
                      feature='median',
                      outputDir=cwd)
```

    Quantifying the probability masks


**Same function if the user wants to run it via Command Line Interface**
```
python generateDLScore.py --probabilityMaskPath /Users/aj/Desktop/gatorExampleData/dlPredict/exampleProbabiltyMap.ome.tif --segmentationMask /Users/aj/Desktop/gatorExampleData/segmentation/exampleSegmentationMask.tif --markerNames ECAD CD45 CD4 CD3D CD8A CD45R Ki67 --outputDir /Users/aj/Desktop/gatorExampleData/

#### If you head over to `GATOR/DLScore/`, you will find the `.csv` file with the DLScores for every cell.


```python
# this tutorial ends here. Move to the Apply Model Tutorial
```
