# 🐊 GATOR 
## Run the final `gator` algorithm
Please keep in mind that the sample data is used for demonstration purposes only and has been simplified and reduced in size. It is solely intended for educational purposes on how to execute `Gator` and will not yeild any meaningful results.

**Download the [exemplar dataset](https://github.com/nirmalLab/gatorpy/tree/main/docs/Tutorials/gatorExampleData) and [executable notebook](https://github.com/nirmalLab/gatorpy/blob/main/docs/Tutorials/notebooks/RunGator.ipynb)**
  
Make sure you have completed `Build Model and Apply Model` Tutorial before you try to execute this Jupyter Notebook!


```python
# import packages
import gatorpy as ga
import os
```

**We need `two` basic input to run the third module of the gator algorithm**
- Pre-computed quantification table (spatial feature table)
- DL Scores


```python
# set the working directory & set paths to the example data
cwd = '/Users/aj/Desktop/gatorExampleData'
# Module specific paths
spatialTablePath = cwd + '/quantification/exampleSpatialTable.csv'
gatorScorePath = cwd + '/GATOR/gatorScore/exampleImage_gatorPredict.ome.csv'
```

## Step-1: Create a Gator object

We'll use the anndata framework to create a gator object to store all information in one file, making it easier to keep track of intermediate files generated in subsequent steps.  This helps streamline the data analysis process and reduces the risk of losing or misplacing information.


```python
# please note that there are a number of defaults in the below function that assumes certain structure within the spatialTable.
# Please confirm it is similar with user data or modifiy the parameters accordingly
# check out the documentation for further details
adata = ga.gatorObject (spatialTablePath=spatialTablePath,
                        gatorScorePath=gatorScorePath,
                        CellId='CellID',
                        uniqueCellId=True,
                        split='X_centroid',
                        removeDNA=True,
                        remove_string_from_name=None,
                        log=True,
                        dropMarkers=None,
                        outputDir=cwd)
```

    Loading exampleSpatialTable.csv


**Same function if the user wants to run it via Command Line Interface**
```
python gatorObject.py --spatialTablePath /Users/aj/Desktop/gatorExampleData/quantification/exampleSpatialTable.csv --gatorScorePath /Users/aj/Desktop/gatorExampleData/GATOR/DLScore/exampleProbabiltyMap.ome.csv --outputDir /Users/aj/Desktop/gatorExampleData

If you had provided `outputDir` the object would be stored in `GATOR/gatorObject/`, else, the object will be returned to memory

## Step-2: Run the Gator Algorithm

The `gator` algorithm is ready to run after pre-processing. To get optimal results, consider adjusting the following parameters:
  
1. The `minAbundance` parameter determines the minimum percentage of a marker's abundance to consider it a failure.
2. It is suggested to drop background markers with the `dropMarkers` option as they can interfere with classifiers.
3. `RobustScale`: Scaling the data before training the classifier model has been shown to improve results. However, in our experience a simple log transformation was found to be sufficient. 


```python
# set the working directory & set paths to the example data
cwd = '/Users/aj/Desktop/gatorExampleData'
gatorObject = cwd + '/GATOR/gatorObject/exampleImage_gatorPredict.ome.h5ad'
```


```python
adata = ga.gator ( gatorObject=gatorObject,
                    gatorScore='gatorScore',
                    minAbundance=0.002,
                    percentiles=[1, 20, 80, 99],
                    dropMarkers = None,
                    RobustScale=False,
                    log=True,
                    x_coordinate='X_centroid',
                    y_coordinate='Y_centroid',
                    imageid='imageid',
                    random_state=0,
                    rescaleMethod='sigmoid',
                    label='gatorOutput',
                    silent=True,
                    outputDir=cwd)
```

**Same function if the user wants to run it via Command Line Interface**
```
python gator.py --gatorObject /Users/aj/Desktop/gatorExampleData/GATOR/gatorObject/exampleImage_gatorPredict.ome.h5ad --outputDir /Users/aj/Desktop/gatorExampleData
```

If `outputDir` is provided, modified anndata object with results (stored in `adata.uns['gatorOutput']`) will be saved in `GATOR/gatorOutput/`. The gator-scaled data (stored in `adata.X`) considers cells above 0.5 as positive and below 0.5 as negative for the marker.

## Step-3: Merge multiple Gator objects (optional)

Use `mergeGatorObject` to combine multiple gatorObjects into a dataset for analysis when multiple images need to be analyzed.
  
Note that merging gatorObjects requires merging multiple sections, not simple concatenation. Use parameters to specify which parts of the gatorObjects to merge.


```python
# set the working directory & set paths to the example data
cwd = '/Users/aj/Desktop/gatorExampleData'
gatorObjects = [cwd + '/GATOR/gatorOutput/exampleImage_gatorPredict.ome.h5ad',
                cwd + '/GATOR/gatorOutput/exampleImage_gatorPredict.ome.h5ad']
```


```python
# For this tutorial, supply the same gatorObject twice for merging, but multiple gatorObjects can be merged in ideal conditions.
adata = ga.mergeGatorObject ( gatorObjects=gatorObjects,
                              fileName='mergedGatorObject',
                              layers=['preProcessed'],
                              uns= ['gatorOutput','gatorScore'],
                              outputDir=cwd)
```

    Extracting data
    Extracting data from: exampleSpatialTable
    Extracting data from: exampleSpatialTable


    /opt/anaconda3/envs/gator/lib/python3.9/site-packages/anndata/_core/anndata.py:1828: UserWarning: Observation names are not unique. To make them unique, call `.obs_names_make_unique`.
      utils.warn_names_duplicates("obs")


**Same function if the user wants to run it via Command Line Interface**
```
python mergeGatorObject.py --gatorObjects /Users/aj/Desktop/gatorExampleData/GATOR/gatorOutput/exampleImage_gatorPredict.ome.h5ad /Users/aj/Desktop/gatorExampleData/GATOR/gatorOutput/exampleImage_gatorPredict.ome.h5ad --outputDir /Users/aj/Desktop/gatorExampleData
```

If `outputDir` is provided, modified anndata object with results will be saved in `GATOR/mergedGatorObject/`.


```python
# this tutorial ends here. Move to the Phenotyping cells Tutorial
```


```python

```
