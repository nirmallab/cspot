# 🎯 Run CSPOT with Docker

1. Install Docker on your local machine if you haven't already done so.
2. Open a terminal or command prompt on your machine.


## Download CSPOT from Docker Hub
```
docker pull nirmallab/cspot:latest

```

**Run Docker**
Running cspot via docker follows the same principles as running cspot via Command Line Interface. 
  
If you are comfortable using Docker and would like to execute the commands in your preferred way, please feel free to do so. However, if you are new to Docker and would like step-by-step instructions, please follow the tutorial below.
  
**Download the [sample data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QDZ6XO)**. Please keep in mind that the sample data is used for demonstration purposes only and has been simplified and reduced in size. It is solely intended for educational purposes on how to execute `cspot` and will not yeild any meaningful results.
  
**The purpose of this tutorial is solely to demonstrate how to run cspot using Docker. If you require detailed explanations of each step, please refer to the other tutorials.**

## Step-1: Generate Thumbnails for Training Data

To use your own data, it is recommended to follow the same folder structure as the sample data. However, if that is not possible, you should place all the required data within a single folder. This is because we need to tell Docker where to find all the raw data, and specifying a single directory makes it easier to manage the data within the container.
  
```
# specify the directory where the sample data lives and Run the docker command
export projectDir="/Users/aj/Documents/cspotExampleData"
docker run -it --mount type=bind,source=$projectDir,target=/$projectDir \
                nirmallab/cspot:latest \
                python /app/generateThumbnails.py \
                --spatialTablePath $projectDir/quantification/exampleSpatialTable.csv \
                --imagePath $projectDir/image/exampleImage.tif \
                --markerChannelMapPath $projectDir/markers.csv \
                --markers ECAD CD3D \
                --maxThumbnails 100 \
                --projectDir $projectDir

```

## Step-2: Generate Masks for Training Data

```
export projectDir="/Users/aj/Documents/cspotExampleData"
docker run -it --mount type=bind,source=$projectDir,target=/$projectDir \
                nirmallab/cspot:latest \
                python /app/generateTrainTestSplit.py \
                --thumbnailFolder $projectDir/CSPOT/Thumbnails/CD3D $projectDir/CSPOT/Thumbnails/ECAD\
                --projectDir $projectDir
```

## Step-3: Train the CSPOT Model

```
export projectDir="/Users/aj/Documents/cspotExampleData"
docker run -it --mount type=bind,source=$projectDir,target=/$projectDir \
                nirmallab/cspot:latest \
                python /app/csTrain.py \
                --trainingDataPath $projectDir/CSPOT/TrainingData \
                --projectDir $projectDir \
                --epochs=1
```

## Step-4: Run the CSPOT Algorithm

Note that the `markers.csv` requests to predict on all markers and so replace the current `cspotModel` folder with these models that are [available for download](https://github.com/nirmallab/cspot/tree/main/docs/Tutorials/manuscriptModels/).   
  
To keep things simple, we're running the entire pipeline with a single command instead of going through the step-by-step process. Nevertheless, you can apply the same principles to each function separately.

```
export projectDir="/Users/aj/Documents/cspotExampleData"
docker run -it --mount type=bind,source=$projectDir,target=/$projectDir \
                nirmallab/cspot:latest \
                python /app/csPipeline.py \
                --imagePath $projectDir/image/exampleImage.tif \
                --csModelPath $projectDir/CSPOT/cspotModel/ \
                --markerChannelMapPath $projectDir/markers.csv \
                --segmentationMaskPath $projectDir/segmentation/exampleSegmentationMask.tif \
                --spatialTablePath $projectDir/quantification/exampleSpatialTable.csv \
                --projectDir $projectDir \
                --verbose False

```

## Step-5: Merge multiple CSPOT objects (optional)

```
export projectDir="/Users/aj/Documents/cspotExampleData"
docker run -it --mount type=bind,source=$projectDir,target=/$projectDir \
                nirmallab/cspot:latest \
                python /app/mergecsObject.py \
                --csObjects $projectDir/CSPOT/cspotOutput/exampleImage_cspotPredict.ome.h5ad $projectDir/CSPOT/cspotOutput/exampleImage_cspotPredict.ome.h5ad \
                --projectDir $projectDir

```

## Step-6: CSPOT Phenotyping

```
export projectDir="/Users/aj/Dcouments/cspotExampleData"
docker run -it --mount type=bind,source=$projectDir,target=/$projectDir \
                            nirmallab/cspot:latest \
                            python /app/csPhenotype.py \
                            --csObject $projectDir/CSPOT/csObject/exampleImage_cspotPredict.ome.h5ad \
                            --phenotype $projectDir/phenotype_workflow.csv \
                            --projectDir $projectDir
```


```python
# Tutorial ends here. Refer to other tutorials for detailed explanation of each step!
```


```python

```
