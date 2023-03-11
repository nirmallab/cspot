# 🐊 GATOR 
## Assign phenotypes to each cell. Clustering data may not always be ideal, so we developed a cell type assignment algorithm that does a hierarchical assignment process iteratively.
#### Please keep in mind that the sample data is used for demonstration purposes only and has been simplified and reduced in size. It is solely intended for educational purposes on how to execute `Gator` and will not yeild any meaningful results.

#### Download the [exemplar dataset](https://github.com/nirmalLab/gatorpy/tree/main/docs/Tutorials/gatorExampleData) and [executable notebook](https://github.com/nirmalLab/gatorpy/blob/main/docs/Tutorials/notebooks/CellPhenotyping.ipynb)
#### Make sure you have completed `Build Model, Apply Model and Run Gator Algorithm` Tutorial before you try to execute this Jupyter Notebook!


```python
# import packages
import gatorpy as ga
import os
import pandas as pd
```

### We need `two` basic input to run the third module of the gator algorithm
- The Gator Object
- A Phenotyping workflow based on prior knowledge


```python
# set the working directory & set paths to the example data
cwd = '/Users/aj/Desktop/gatorExampleData'
# Module specific paths
gatorObject = cwd + '/GATOR/gatorObject/exampleImage_gatorPredict.ome.h5ad'
```


```python
# load the phenotyping workflow
phenotype = pd.read_csv(str(cwd) + '/phenotype_workflow.csv')
# view the table:
phenotype.style.format(na_rep='')
```




<style type="text/css">
</style>
<table id="T_58459">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_58459_level0_col0" class="col_heading level0 col0" >Unnamed: 0</th>
      <th id="T_58459_level0_col1" class="col_heading level0 col1" >Unnamed: 1</th>
      <th id="T_58459_level0_col2" class="col_heading level0 col2" >ECAD</th>
      <th id="T_58459_level0_col3" class="col_heading level0 col3" >CD45</th>
      <th id="T_58459_level0_col4" class="col_heading level0 col4" >CD4</th>
      <th id="T_58459_level0_col5" class="col_heading level0 col5" >CD3D</th>
      <th id="T_58459_level0_col6" class="col_heading level0 col6" >CD8A</th>
      <th id="T_58459_level0_col7" class="col_heading level0 col7" >KI67</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_58459_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_58459_row0_col0" class="data row0 col0" >all</td>
      <td id="T_58459_row0_col1" class="data row0 col1" >Immune</td>
      <td id="T_58459_row0_col2" class="data row0 col2" ></td>
      <td id="T_58459_row0_col3" class="data row0 col3" >anypos</td>
      <td id="T_58459_row0_col4" class="data row0 col4" >anypos</td>
      <td id="T_58459_row0_col5" class="data row0 col5" >anypos</td>
      <td id="T_58459_row0_col6" class="data row0 col6" >anypos</td>
      <td id="T_58459_row0_col7" class="data row0 col7" ></td>
    </tr>
    <tr>
      <th id="T_58459_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_58459_row1_col0" class="data row1 col0" >all</td>
      <td id="T_58459_row1_col1" class="data row1 col1" >ECAD+</td>
      <td id="T_58459_row1_col2" class="data row1 col2" >pos</td>
      <td id="T_58459_row1_col3" class="data row1 col3" ></td>
      <td id="T_58459_row1_col4" class="data row1 col4" ></td>
      <td id="T_58459_row1_col5" class="data row1 col5" ></td>
      <td id="T_58459_row1_col6" class="data row1 col6" ></td>
      <td id="T_58459_row1_col7" class="data row1 col7" ></td>
    </tr>
    <tr>
      <th id="T_58459_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_58459_row2_col0" class="data row2 col0" >ECAD+</td>
      <td id="T_58459_row2_col1" class="data row2 col1" >KI67+ ECAD+</td>
      <td id="T_58459_row2_col2" class="data row2 col2" ></td>
      <td id="T_58459_row2_col3" class="data row2 col3" ></td>
      <td id="T_58459_row2_col4" class="data row2 col4" ></td>
      <td id="T_58459_row2_col5" class="data row2 col5" ></td>
      <td id="T_58459_row2_col6" class="data row2 col6" ></td>
      <td id="T_58459_row2_col7" class="data row2 col7" >pos</td>
    </tr>
    <tr>
      <th id="T_58459_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_58459_row3_col0" class="data row3 col0" >Immune</td>
      <td id="T_58459_row3_col1" class="data row3 col1" >CD4+ T</td>
      <td id="T_58459_row3_col2" class="data row3 col2" ></td>
      <td id="T_58459_row3_col3" class="data row3 col3" ></td>
      <td id="T_58459_row3_col4" class="data row3 col4" >allpos</td>
      <td id="T_58459_row3_col5" class="data row3 col5" >allpos</td>
      <td id="T_58459_row3_col6" class="data row3 col6" ></td>
      <td id="T_58459_row3_col7" class="data row3 col7" ></td>
    </tr>
    <tr>
      <th id="T_58459_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_58459_row4_col0" class="data row4 col0" >Immune</td>
      <td id="T_58459_row4_col1" class="data row4 col1" >CD8+ T</td>
      <td id="T_58459_row4_col2" class="data row4 col2" ></td>
      <td id="T_58459_row4_col3" class="data row4 col3" ></td>
      <td id="T_58459_row4_col4" class="data row4 col4" ></td>
      <td id="T_58459_row4_col5" class="data row4 col5" >allpos</td>
      <td id="T_58459_row4_col6" class="data row4 col6" >allpos</td>
      <td id="T_58459_row4_col7" class="data row4 col7" ></td>
    </tr>
    <tr>
      <th id="T_58459_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_58459_row5_col0" class="data row5 col0" >Immune</td>
      <td id="T_58459_row5_col1" class="data row5 col1" >Non T CD4+ cells</td>
      <td id="T_58459_row5_col2" class="data row5 col2" ></td>
      <td id="T_58459_row5_col3" class="data row5 col3" ></td>
      <td id="T_58459_row5_col4" class="data row5 col4" >pos</td>
      <td id="T_58459_row5_col5" class="data row5 col5" >neg</td>
      <td id="T_58459_row5_col6" class="data row5 col6" ></td>
      <td id="T_58459_row5_col7" class="data row5 col7" ></td>
    </tr>
  </tbody>
</table>




### As it can be seen from the table above,
### (1) The `first column` has to contain the cell that are to be classified.
### (2) The `second column` indicates the phenotype a particular cell will be assigned if it satifies the conditions in the row.
### (3) `Column three` and onward represent protein markers. If the protein marker is known to be expressed for that cell type, then it is denoted by either `pos`, `allpos`. If the protein marker is known to not express for a cell type it can be denoted by `neg`, `allneg`. If the protein marker is irrelevant or uncertain to express for a cell type, then it is left empty. `anypos` and `anyneg` are options for using a set of markers and if any of the marker is positive or negative, the cell type is denoted accordingly.

### To give users maximum flexibility in identifying desired cell types, we have implemented various classification arguments as described above for strategical classification. They include

- allpos
- allneg
- anypos
- anyneg
- pos
- neg

### `pos` : "Pos" looks for cells positive for a given marker. If multiple markers are annotated as `pos`, all must be positive to denote the cell type. For example, a Regulatory T cell can be defined as `CD3+CD4+FOXP3+` by passing `pos` to each marker. If one or more markers don't meet the criteria (e.g. CD4-), the program will classify it as `Likely-Regulatory-T cell`, pending user confirmation. This is useful in cases of technical artifacts or when cell types (such as cancer cells) are defined by marker loss (e.g. T-cell Lymphomas).

### `neg` : Same as `pos` but looks for negativity of the defined markers. 

### `allpos` : "Allpos" requires all defined markers to be positive. Unlike `pos`, it doesn't classify cells as `Likely-cellType`, but strictly annotates cells positive for all defined markers.

### `allneg` : Same as `allpos` but looks for negativity of the defined markers. 

### `anypos` : "Anypos" requires only one of the defined markers to be positive. For example, to define macrophages, a cell could be designated as such if any of `CD68`, `CD163`, or `CD206` is positive.

### `anyneg` : Same as `anyneg` but looks for negativity of the defined markers. 


```python
adata = ga.gatorPhenotype ( gatorObject=gatorObject,
                            phenotype=phenotype,
                            midpoint = 0.5,
                            label="phenotype",
                            imageid='imageid',
                            pheno_threshold_percent=None,
                            pheno_threshold_abs=None,
                            fileName=None,
                            outputDir=cwd)
```

    Phenotyping Immune
    Phenotyping ECAD+
    -- Subsetting ECAD+
    Phenotyping KI67+ ECAD+
    -- Subsetting Immune
    Phenotyping CD4+ T
    Phenotyping CD8+ T
    Phenotyping Non T CD4+ cells
    Consolidating the phenotypes across all groups


    /Users/aj/Dropbox (Partners HealthCare)/nirmal lab/softwares/gatorpy/gatorpy/gatorPhenotype.py:255: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      allpos_score['score'] = allpos_score.max(axis=1)
    /Users/aj/Dropbox (Partners HealthCare)/nirmal lab/softwares/gatorpy/gatorpy/gatorPhenotype.py:255: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      allpos_score['score'] = allpos_score.max(axis=1)


**Same function if the user wants to run it via Command Line Interface**
```
python gatorPhenotype.py --gatorObject /Users/aj/Desktop/gatorExampleData/GATOR/gatorObject/exampleImage_gatorPredict.ome.h5ad --phenotype /Users/aj/Desktop/gatorExampleData/phenotype_workflow.csv --outputDir /Users/aj/Desktop/gatorExampleData
```

#### If you had provided `outputDir` the object would be stored in `GATOR/gatorPhenotyped/`, else, the object will be returned to memory.


```python
# check the identified phenotypes
adata.obs['phenotype'].value_counts()
```




    KI67+ ECAD+    6159
    CD4+ T         5785
    CD8+ T          816
    Name: phenotype, dtype: int64



#### We also provide some helper functions to vizualize the identified postive and negative cells for each marker. 

#### The `addPredictions` function serves as a link between `gatorpy` and `scimap` package. It's useful for evaluating model performance. The function transforms results stored in `anndata.uns` to `anndata.obs` so they can be visualized using the `scimap` package's `sm.pl.image viewer` function. This displays `positive` and `negative` cells overlaid on the raw image.
    
#### The `addPredictions` function can take in two methods.  `gatorOutput` displays the result of running the `gator` function,  while `gatorScore` shows the raw output produced by the `gatorScore`  function, which returns a probability score. The `midpoint` parameter,  with a default value of 0.5, can be adjusted to define what is considered a `positive` result, when method is set to `gatorScore`.


```python
# set the working directory & set paths to the example data
cwd = '/Users/aj/Desktop/gatorExampleData'
# Module specific paths
gatorObject = cwd + '/GATOR/gatorOutput/exampleImage_gatorPredict.ome.h5ad'

adata = ga.addPredictions (gatorObject, 
                    method='gatorOutput',
                    gatorOutput='gatorOutput',
                    gatorScore='gatorScore', 
                    midpoint=0.5,
                    outputDir=cwd + '/GATOR/gatorOutput/')
```


```python
# check the results
adata.obs.columns
```




    Index(['X_centroid', 'Y_centroid', 'Area', 'MajorAxisLength',
           'MinorAxisLength', 'Eccentricity', 'Solidity', 'Extent', 'Orientation',
           'CellID', 'imageid', 'p_CD45', 'p_CD4', 'p_CD8A', 'p_CD45R', 'p_KI67',
           'p_ECAD', 'p_CD3D'],
          dtype='object')



#### As it can be seen the addition of `p_CD45, p_CD4, p_CD8A, p_CD45R, p_KI67, p_ECAD, p_CD3D` to `adata.obs`. These columns can be vizualized with `scimap`. 

## We recommend creating a new environment to install scimap

#### Download and install the scimap package. We recommend creating a new conda/python environment

```
# create new conda env (assuming you have conda installed): executed in the conda command prompt or terminal
conda create --name scimap -y python=3.8
conda activate scimap

```

#### Install `scimap` within the conda environment.

```
pip install scimap

# install jupyter notebook if you want to simply execute this notebook.
pip install notebook

```

### Once `scimap` is installed the following function can be used to vizualize the results


```python
# import
import scimap as sm

# import the gatorObject
cwd = '/Users/aj/Desktop/gatorExampleData'
gatorObject = cwd + '/GATOR/gatorOutput/exampleImage_gatorPredict.ome.h5ad'
adata = ad.read(gatorObject)

# Path to the raw image
image_path = '/Users/aj/Desktop/gatorExampleData/image/exampleImage.tif'
image_viewer(image_path, adata, overlay='p_CD45')

```
