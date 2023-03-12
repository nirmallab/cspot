# -*- coding: utf-8 -*-
# Created on Mon Nov  9 21:00:57 2020
# @author: Ajit Johnson Nirmal
"""
!!! abstract "Short Description"
    Users can utilize the `gatorToCSV` function to store the contents of the gatorObject as a CSV file. 
    It is important to keep in mind that since the object contains a range of intermediate files, multiple CSV files will be produced.

## Function
"""

# Import
import pandas as pd
import numpy as np
import argparse
import sys
import pathlib
import anndata as ad



# Function
def gatorToCSV (gatorObject, data_type='raw', output_dir=None, file_name=None, CellID='CellID'):
    """
Parameters:
    gatorObject : AnnData object loaded into memory or path to AnnData object.

    data_type : string, optional  
        Three options are available:  
        1) 'raw' - The raw data will be returned.  
        2) 'log' - The raw data converted to log scale using `np.log1p` will be returned.  
        3) 'scaled' - If you have scaled the data using the `sm.pp.rescale`, that will be
        returned. Please note, if you have not scaled the data, whatever is within
        `gatorObject.X` will be returned.
        
    output_dir : string, optional  
        Path to output directory.
    
    file_name : string, optional
        Name the output csv file. Use in combination with `output_dir` parameter. If no
        file name is provided a default name `scimap_to_csv_file.csv` will be used. 
    
    CellID : string, optional  
        Name of the column which contains the CellID. Default is `CellID`.  

Returns:
    merged : DataFrame  
        A single dataframe containing the expression and metadata will be returned.
        
Example:
```python
    data = ga.gatorToCSV (gatorObject, data_type='raw')
```
    """
    
    # Load the andata object    
    if isinstance(gatorObject, str):
        if file_name is None:
            imid = str(gatorObject.rsplit('/', 1)[-1])
        else: 
            imid = str(file_name)
        gatorObject = ad.read(gatorObject)
    else:
        if file_name is None:
            imid = "gatorToCSV.csv"
        else: 
            imid = str(file_name)
        gatorObject = gatorObject
    
    # Expression matrix
    if data_type == 'raw':
        data = pd.DataFrame(gatorObject.raw.X, index=gatorObject.obs.index, columns=gatorObject.var.index)
    if data_type == 'log':
        data = pd.DataFrame(np.log1p(gatorObject.raw.X), index=gatorObject.obs.index, columns=gatorObject.var.index)
    if data_type == 'scaled':
        data = pd.DataFrame(gatorObject.X, index=gatorObject.obs.index, columns=gatorObject.var.index)
    
    # Metadata
    meta = pd.DataFrame(gatorObject.obs)
    
    # Merge the two dataframes
    merged = pd.concat([data, meta], axis=1, sort=False)
        
    # Add a column to save cell-id
    #merged['CellID'] = merged.index
    # make cellID the first column
    if CellID in merged.columns:
        first_column = merged.pop(CellID)
        merged.insert(0, CellID, first_column)
    else:
        merged['CellID'] = merged.index
        first_column = merged.pop(CellID)
        merged.insert(0, CellID, first_column)
    
    # reset index
    merged = merged.reset_index(drop=True)

    # Save data if requested
    if output_dir is not None:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        merged.to_csv(output_dir / f'{imid}.csv', index=False)
    else:    
        # Return data
        return merged


