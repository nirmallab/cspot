# 🐊 Build a CSPOT Model 


The [executable notebook can be **downloaded here**](https://github.com/nirmalLab/gatorpy/blob/main/docs/Tutorials/notebooks/BuildModels.ipynb)  
  
**When following the tutorial, it is crucial to read the text as simply running the cells will not work!**
  
Please keep in mind that the [sample data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QDZ6XO) is used for demonstration purposes only and has been simplified and reduced in size. It is solely intended for educational purposes on how to execute `cspot` and will not yeild any meaningful results.

**Training a CSPOT Model involves the following steps:**
- For any given marker: Identify a image that could be used to generate postive and negative thumbnails
- Run the `generateThumbnails` function on the image to auto generate postive and negative thumbnails
- Go through the auto generated  thumbnails and remove any wrong assignments
- On the user sorted set of thumbnails run the `generateTrainTestSplit` function to prepare the data for training a deep learning model
- Lastly, run `csTrain` 

<hr>


```python
# import packages in jupyter notebook (not needed for command line interface users)
import cspot as cs
```

    WARNING:tensorflow:From /Users/aj/miniconda3/envs/cspot/lib/python3.9/site-packages/tensorflow/python/compat/v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term


**CSPOT auto generates subfolders and so always set a single folder as `projectDir` and cspot will use that for all subsequent steps.**  
In this case we will set the downloaded sample data as our `projectDir`. My sample data is on my desktop as seen below.


```python
# set the working directory & set paths to the example data
projectDir = '/Users/aj/Documents/cspotExampleData'
imagePath = projectDir + '/image/exampleImage.tif'
spatialTablePath = projectDir + '/quantification/exampleSpatialTable.csv'
markerChannelMapPath = projectDir + '/markers.csv'
```

## Step-1: Generate Thumbnails for Training Data

The first step would be to train a model to recognize the marker of interest. In this example the data contains 11 channels `DNA1, ECAD, CD45, CD4, CD3D, CD8A, CD45R, KI67` and as we are not interested in training a model to recognize DNA or background (`DNA1`), we will only need to generate training data for  `ECAD1, CD45, CD4, CD3D, CD8A & KI67`. However for proof of concept, let us just train a model for `ECAD` and `CD3D`.

To do so, the first step is to create examples of `postive` and `negative` examples for each marker of interest. To facilitate this process, we can use the `generateThumbnails` function in `CSPOT`. Under the hood the function auto identifies the cells that has high and low expression of the marker of interest and cuts out small thumbnails from the image.


```python
cs.generateThumbnails ( spatialTablePath=spatialTablePath, 
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
                        projectDir=projectDir)

```

    Processing Marker: ECAD
    Processing Marker: CD3D
    Thumbnails have been generated, head over to "/Users/aj/Documents/cspotExampleData/CSPOT/Thumbnails" to view results


**Same function if the user wants to run it via Command Line Interface**
```
python generateThumbnails.py \
            --spatialTablePath /Users/aj/Documents/cspotExampleData/quantification/exampleSpatialTable.csv \
            --imagePath /Users/aj/Documents/cspotExampleData/image/exampleImage.tif \
            --markerChannelMapPath /Users/aj/Documents/cspotExampleData/markers.csv \
            --markers ECAD CD3D \
            --maxThumbnails 100 \
            --projectDir /Users/aj/Documents/cspotExampleData/
```

**The output from the above function will be stored under `CSPOT/Thumbnails/`.**  
  
There are a number of parameters that function need to provided as seen above. Detailed explanations are avaialable in the documentation. Briefly, the function takes in the single-cell table (`spatialTablePath`) with X and Y coordinates, the full image (`imagePath`) and lastly a list of `markers` for which thumbnails need to be generated. Please note as the program does not know which channels in the image corresponds to the `markers`, hence, the `markerChannelMapPath` is used to supply a `.csv` file that maps the channels to the marker information. The `markerChannelMap` follow 1-indexing convention- so the first channel is represented by the number `1`. 
  
You would have also notices that I have set `maxThumbnails=100`. This basically means that even if more than 100 cells are identified, only 100 random cells will be used to generate the thumbnails. I generally generate about `2000` cells, however based on our estimates about 250 postive and 250 negative examples should be suffcient. As this is for illustration purpose only, I have set it to `100`.  
  
Now that the thumbnails are generated, one would manually go through the `TruePos` folder and `TrueNeg` folder and move files around as necessary. If there are any truly negative thumbnails in the `TruePos` folder, move it to `PosToNeg` folder. Similarly, if there are any truly positive thumbnails in `TrueNeg` folder, move it to `NegToPos` folder. You will often notice that imaging artifacts are captured in the `TruePos` folder and there will also likely be a number of true positives in the `TrueNeg` folder as the field of view (64x64) is larger than what the program used to identify those thumbnails (just the centroids of single cells at the center of that thumbnail).    
  
While you are manually sorting the postives and negative thumbnails, please keep in mind that you are looking for high-confident positives and high-confident negatives. It is absolutely okay to delete off majority of the thumbnails that you are not confident about. This infact makes it easy and fast as you are looking to only keep only thumbnails that are readily sortable.  
  
Lastly, I generally use a whole slide image to generate these thumbnails as there will be enough regions with high expression and no expression of the marker of interest. If you look at the thumbnails of this dummy example, you will notice that most thumbnails of `TrueNeg` for `ECAD` does contain some level of `ECAD` as there is not enough regions to sample from. 

## Step-1a (optional)

You might have noticed in the above example, I had set `localNorm=True`, which is on by default. This parameter essentially creates a mirror duplicate copy of all the thumbnails and saves it under a folder named `localNorm`. The difference being that each thumbnail is normalized to the maximum intensity pixel in that thumbnail. It helps to visually sort out the true positives and negatives faster and more reliably. As we will not use the thumbnails in the `localNorm` for training the deep learning model, we want to make sure all the manual sorting that we did in the `localNorm` folder is copied over to the real training data. I have written an additional function to help with this. Any moving or deleting of files that you did in the `localNorm` folder will be copied over to the real training data.  
  
Randomly shift and delete some files from `TruePos` -> `PosToNeg` and `TrueNeg` -> `NegToPos`   for   `CD3D` for the purpose of illustration and run the  `cloneFolder` function to see what happens.


```python
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
```

    Processing: CD3D
    Processing: ECAD
    Cloning Folder is complete, head over to /CSPOT/Thumbnails" to view results


**Same function if the user wants to run it via Command Line Interface**
```
python cloneFolder.py \
            --copyFolder /Users/aj/Desktop/cspotExampleData/CSPOT/Thumbnails/localNorm/CD3D /Users/aj/Desktop/cspotExampleData/CSPOT/Thumbnails/localNorm/ECAD \
            --applyFolder /Users/aj/Desktop/cspotExampleData/CSPOT/Thumbnails/CD3D /Users/aj/Desktop/cspotExampleData/CSPOT/Thumbnails/ECAD
```

**If you head over to the training data thumbails you will notice that the files have been shifited around exactly as in the `localNorm` folder.**

## Step-2: Generate Masks for Training Data

To train the deep learning model, in addition to the raw thumbnails a mask is needed. The mask lets the model know where the cell is located. Ideally one would manually draw on the thumbnails to locate where the positive cells are, however for the pupose of scalability we will use automated approaches to generate the Mask for us. The following function will generate the mask and split the data into `training, validation and test` that can be directly fed into the deep learning algorithm.


```python
thumbnailFolder = [projectDir + '/CSPOT/Thumbnails/CD3D',
                   projectDir + '/CSPOT/Thumbnails/ECAD']

# The function accepts the four pre-defined folders. If you had renamed them, please change it using the parameter below.
# If you had deleted any of the folders and are not using them, replace the folder name with `None` in the parameter.
cs.generateTrainTestSplit ( thumbnailFolder, 
                            projectDir=projectDir,
                            file_extension=None,
                            TruePos='TruePos', NegToPos='NegToPos',
                            TrueNeg='TrueNeg', PosToNeg='PosToNeg')
```

    Processing: CD3D
    Processing: ECAD
    Training data has been generated, head over to "/Users/aj/Documents/cspotExampleData/CSPOT/TrainingData" to view results


**Same function if the user wants to run it via Command Line Interface**
```
python generateTrainTestSplit.py \
            --thumbnailFolder /Users/aj/Desktop/cspotExampleData/CSPOT/Thumbnails/CD3D /Users/aj/Desktop/cspotExampleData/CSPOT/Thumbnails/ECAD \
            --projectDir /Users/aj/Desktop/cspotExampleData/
```

If you head over to `CSPOT/TrainingData/`, you will notice that each of the supplied marker above will have a folder with the associated `training, validataion and test` data that is required by the deep-learning algorithm to generate the model. 

## Step-3: Train the CSPOT Model

The function trains a deep learning model for each marker in the provided training data. To train the `cspotModel`, simply direct the function to the `TrainingData` folder. To train only specific models, specify the folder names using the `trainMarkers` parameter. The 'outputDir' remains constant and the program will automatically create subfolders to save the trained models.


```python
trainingDataPath = projectDir + '/CSPOT/TrainingData'

cs.csTrain(trainingDataPath=trainingDataPath,
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
```

    WARNING:tensorflow:From /Users/aj/miniconda3/envs/cspot/lib/python3.9/site-packages/keras/src/layers/normalization/batch_normalization.py:883: _colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.


    /Users/aj/miniconda3/envs/cspot/lib/python3.9/site-packages/cspot/UNet.py:137: UserWarning: `tf.layers.batch_normalization` is deprecated and will be removed in a future version. Please use `tf.keras.layers.BatchNormalization` instead. In particular, `tf.control_dependencies(tf.GraphKeys.UPDATE_OPS)` should not be used (consult the `tf.keras.layers.BatchNormalization` documentation).
      bn = tf.nn.leaky_relu(tf.layers.batch_normalization(c00+shortcut, training=UNet2D.tfTraining))
    /Users/aj/miniconda3/envs/cspot/lib/python3.9/site-packages/cspot/UNet.py:159: UserWarning: `tf.layers.batch_normalization` is deprecated and will be removed in a future version. Please use `tf.keras.layers.BatchNormalization` instead. In particular, `tf.control_dependencies(tf.GraphKeys.UPDATE_OPS)` should not be used (consult the `tf.keras.layers.BatchNormalization` documentation).
      lbn = tf.nn.leaky_relu(tf.layers.batch_normalization(
    /Users/aj/miniconda3/envs/cspot/lib/python3.9/site-packages/cspot/UNet.py:162: UserWarning: `tf.layers.dropout` is deprecated and will be removed in a future version. Please use `tf.keras.layers.Dropout` instead.
      return tf.layers.dropout(lbn, 0.15, training=UNet2D.tfTraining)
    /Users/aj/miniconda3/envs/cspot/lib/python3.9/site-packages/cspot/UNet.py:224: UserWarning: `tf.layers.batch_normalization` is deprecated and will be removed in a future version. Please use `tf.keras.layers.BatchNormalization` instead. In particular, `tf.control_dependencies(tf.GraphKeys.UPDATE_OPS)` should not be used (consult the `tf.keras.layers.BatchNormalization` documentation).
      tf.layers.batch_normalization(tf.nn.conv2d(cc, luXWeights2, strides=[1, 1, 1, 1], padding='SAME'),
    /Users/aj/miniconda3/envs/cspot/lib/python3.9/site-packages/cspot/UNet.py:245: UserWarning: `tf.layers.batch_normalization` is deprecated and will be removed in a future version. Please use `tf.keras.layers.BatchNormalization` instead. In particular, `tf.control_dependencies(tf.GraphKeys.UPDATE_OPS)` should not be used (consult the `tf.keras.layers.BatchNormalization` documentation).
      return tf.layers.batch_normalization(


    /Users/aj/Documents/cspotExampleData/CSPOT/TrainingData/CD3D/training
    Training for 8 steps
    Found 122 training images
    Found 41 validation images
    Found 41 test images
    Of these, 0 are artefact training images
     and  0 artefact validation images
    Using 0 and 1 for mean and standard deviation.
    saving data
    saving data
    Using 16.0 and 0.0 for global max and min intensities.
    Class balance ratio is 18.959115759448537
    WARNING:tensorflow:From /Users/aj/miniconda3/envs/cspot/lib/python3.9/site-packages/tensorflow/python/util/dispatch.py:1176: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    WARNING:tensorflow:From /Users/aj/miniconda3/envs/cspot/lib/python3.9/site-packages/tensorflow/python/util/dispatch.py:1176: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    WARNING:tensorflow:From /Users/aj/miniconda3/envs/cspot/lib/python3.9/site-packages/tensorflow/python/util/dispatch.py:1176: calling reduce_min_v1 (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
    Instructions for updating:
    keep_dims is deprecated, use keepdims instead
    WARNING:tensorflow:From /Users/aj/miniconda3/envs/cspot/lib/python3.9/site-packages/tensorflow/python/util/dispatch.py:1176: calling reduce_max_v1 (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
    Instructions for updating:
    keep_dims is deprecated, use keepdims instead
    WARNING:tensorflow:From /Users/aj/miniconda3/envs/cspot/lib/python3.9/site-packages/tensorflow/python/util/dispatch.py:1176: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Deprecated in favor of operator or tf.math.divide.
    step 00000, e: 0.506249, epoch: 1
    Model saved in file: /Users/aj/Documents/cspotExampleData/CSPOT/cspotModel/CD3D/model.ckpt
    step 00001, e: 0.500656, epoch: 1
    step 00002, e: 0.481774, epoch: 1
    step 00003, e: 0.466444, epoch: 1
    step 00004, e: 0.441243, epoch: 1
    step 00005, e: 0.492766, epoch: 1
    step 00006, e: 0.522770, epoch: 2
    step 00007, e: 0.508577, epoch: 2
    saving data
    loading data
    INFO:tensorflow:Restoring parameters from /Users/aj/Documents/cspotExampleData/CSPOT/cspotModel/CD3D/model.ckpt
    Model restored.
    /Users/aj/Documents/cspotExampleData/CSPOT/TrainingData/ECAD/training
    Training for 8 steps
    Found 120 training images
    Found 40 validation images
    Found 40 test images
    Of these, 0 are artefact training images
     and  0 artefact validation images
    Using 0 and 1 for mean and standard deviation.
    saving data
    saving data
    Using 70.0 and 6.0 for global max and min intensities.
    Class balance ratio is 6.6801200018750295
    step 00000, e: 0.498890, epoch: 1
    Model saved in file: /Users/aj/Documents/cspotExampleData/CSPOT/cspotModel/ECAD/model.ckpt
    step 00001, e: 0.494717, epoch: 1
    step 00002, e: 0.510676, epoch: 1
    step 00003, e: 0.480390, epoch: 1
    step 00004, e: 0.478806, epoch: 1
    step 00005, e: 0.516272, epoch: 1
    step 00006, e: 0.512058, epoch: 2
    step 00007, e: 0.480931, epoch: 2
    saving data
    loading data
    INFO:tensorflow:Restoring parameters from /Users/aj/Documents/cspotExampleData/CSPOT/cspotModel/ECAD/model.ckpt
    Model restored.
    CSPOT Models have been generated, head over to "/Users/aj/Documents/cspotExampleData/CSPOT/cspotModel" to view results


**Same function if the user wants to run it via Command Line Interface**
```
python csTrain.py \
        --trainingDataPath /Users/aj/Documents/cspotExampleData/CSPOT/TrainingData \
        --projectDir /Users/aj/Documents/cspotExampleData/ \
        --epochs 1
```


```python
# this tutorial ends here. Move to the Run CSPOT Algorithm Tutorial
```


```python

```
