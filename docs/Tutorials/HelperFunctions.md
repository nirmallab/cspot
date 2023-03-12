# 🐊 GATOR Helper Functions

## Export the results to `.csv`
Once the Gator pipeline has been executed, all the output is stored within the gatorObject. An efficient way to export the results of gatorScore, gatorOutput, and the rescaled data is by using a function that saves them as a CSV file. This allows for easy sharing and analysis of the data in other programs.


```python
# import packages
import gatorpy as ga
```


```python
# path to files needed for gatorExport
projectDir = '/Users/aj/Desktop/gatorExampleData'
gatorObject = projectDir + '/GATOR/gatorOutput/exampleImage_gatorPredict.ome.h5ad'
```


```python
ga.gatorExport(gatorObject,
               projectDir,
               fileName=None,
               raw=False,
               CellID='CellID',
               verbose=True)

```

    Contents of the gatorObject have been exported to "/Users/aj/Desktop/gatorExampleData/GATOR/gatorExport"


**Same function if the user wants to run it via Command Line Interface**
```
python gatorExport.py --gatorObject /Users/aj/Desktop/gatorExampleData/GATOR/gatorOutput/exampleImage_gatorPredict.ome.h5ad --projectDir /Users/aj/Desktop/gatorExampleData
```

## We also provide a helper functions to vizualize the identified postive and negative cells for each marker.

The `addPredictions` function serves as a link between `gatorpy` and `scimap` package. It's useful for evaluating model performance. The function transforms results stored in `anndata.uns` to `anndata.obs` so they can be visualized using the `scimap` package's `sm.pl.image viewer` function. This displays `positive` and `negative` cells overlaid on the raw image.
      
The `addPredictions` function can take in two methods.  `gatorOutput` displays the result of running the `gator` function,  while `gatorScore` shows the raw output produced by the `gatorScore`  function, which returns a probability score. The `midpoint` parameter,  with a default value of 0.5, can be adjusted to define what is considered a `positive` result, when method is set to `gatorScore`.


```python
# Path to gatorObject
gatorObject = projectDir + '/GATOR/gatorOutput/exampleImage_gatorPredict.ome.h5ad'

adata = ga.addPredictions (gatorObject, 
                    method='gatorOutput',
                    gatorOutput='gatorOutput',
                    gatorScore='gatorScore', 
                    midpoint=0.5)

```


```python
# check the results
adata.obs.columns
```


    Index(['X_centroid', 'Y_centroid', 'Area', 'MajorAxisLength',
           'MinorAxisLength', 'Eccentricity', 'Solidity', 'Extent', 'Orientation',
           'CellID', 'imageid', 'p_p_CD45R', 'p_p_CD8A', 'p_p_CD4', 'p_p_KI67',
           'p_p_ECAD', 'p_p_CD45', 'p_p_CD3D'],
          dtype='object')


As it can be seen the addition of `p_CD45, p_CD4, p_CD8A, p_CD45R, p_KI67, p_ECAD, p_CD3D` to `adata.obs`. These columns can be vizualized with `scimap`. 

## We recommend creating a new environment to install scimap

**Download and install the scimap package. We recommend creating a new conda/python environment**

```
# create new conda env (assuming you have conda installed): executed in the conda command prompt or terminal
conda create --name scimap -y python=3.8
conda activate scimap

```

**Install `scimap` within the conda environment.**

```
pip install scimap

# install jupyter notebook if you want to simply execute this notebook.
pip install notebook

```

**Once `scimap` is installed the following function can be used to vizualize the results**


```python
# import
import scimap as sm
import anndata as ad

# import the gatorObject
cwd = '/Users/aj/Desktop/gatorExampleData'
gatorObject = cwd + '/GATOR/gatorOutput/exampleImage_gatorPredict.ome.h5ad'
adata = ad.read(gatorObject)

# Path to the raw image
image_path = '/Users/aj/Desktop/gatorExampleData/image/exampleImage.tif'
sm.image_viewer(image_path, adata, overlay='p_CD45')

```
