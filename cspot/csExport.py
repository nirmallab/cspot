# -*- coding: utf-8 -*-
# Created on Mon Nov  9 21:00:57 2020
# @author: Ajit Johnson Nirmal
"""
!!! abstract "Short Description"
    Users can utilize the `csExport` function to store the contents of the csObject to a `.CSV` file. 
      
    Keep in mind that the presence of multiple intermediate files in the object will result in the production of several CSV files.

## Function
"""

# Import
import pandas as pd
import argparse
import pathlib
import anndata as ad
import os


# Function
def csExport(
    csObject, projectDir, fileName=None, raw=False, CellID="CellID", verbose=True
):
    """
Parameters:
    csObject (anndata):
        Pass the `csObject` loaded into memory or a path to the `csObject` 
        file (.h5ad).
        
    projectDir (str, optional):
        Provide the path to the output directory. The result will be located at
        `projectDir/CSPOT/csExport/`. 

    fileName (str, optional):
        Specify the name of the CSV output file. If you don't provide a file name, t
        he default name `csExport.csv` will be assigned.
        
    raw (bool, optional):
        If `True` raw data will be returned instead of the CSPOT scaled data.
        
    CellId (str, optional):
        Specify the column name that holds the cell ID (a unique name given to each cell).

    verbose (bool, optional):
        If True, print detailed information about the process to the console.  

Returns:
    CSV files (.csv):
        The `.csv` files can be found under `projectDir/CSPOT/csExport/`

Example:
        ```python
        # path to files needed for csExport
        projectDir = '/Users/aj/Documents/cspotExampleData'
        csObject = projectDir + '/CSPOT/csObject/exampleImage_cspotPredict.ome.h5ad'
        
        cs.csExport(csObject,
               projectDir,
               fileName=None,
               raw=False,
               CellID='CellID',
               verbose=True)
        
        # Same function if the user wants to run it via Command Line Interface
        python csExport.py \
            --csObject /Users/aj/Documents/cspotExampleData/CSPOT/cspotOutput/exampleImage_cspotPredict.ome.h5ad \
            --projectDir /Users/aj/Documents/cspotExampleData
        
        ```

    """

    # Load the andata object
    if isinstance(csObject, str):
        if fileName is None:
            imid = pathlib.Path(csObject).stem
        else:
            imid = str(fileName)
        csObject = ad.read(csObject)
    else:
        if fileName is None:
            imid = "csExport"
        else:
            imid = str(fileName)
        csObject = csObject

    # Expression matrix & obs data
    if raw is True:
        data = pd.DataFrame(
            csObject.raw.X, index=csObject.obs.index, columns=csObject.var.index
        )
    else:
        data = pd.DataFrame(
            csObject.X, index=csObject.obs.index, columns=csObject.var.index
        )
    meta = pd.DataFrame(csObject.obs)
    # Merge the two dataframes
    merged = pd.concat([data, meta], axis=1, sort=False)

    # Add a column to save cell-id
    # make cellID the first column
    if CellID in merged.columns:
        first_column = merged.pop(CellID)
        merged.insert(0, CellID, first_column)
    else:
        merged["CellID"] = merged.index
        first_column = merged.pop(CellID)
        merged.insert(0, CellID, first_column)

    # reset index
    merged = merged.reset_index(drop=True)

    # create a folder to hold the results
    folderPath = pathlib.Path(projectDir + "/CSPOT/csExport/")
    folderPath.mkdir(exist_ok=True, parents=True)

    # extract some of the data stored in .uns and save
    if hasattr(csObject, "uns") and "cspotOutput" in csObject.uns:
        cspotOutput = csObject.uns["cspotOutput"]
        cspotOutput.index = merged["CellID"]
        cspotOutput.to_csv(folderPath / "cspotOutput.csv")
    if hasattr(csObject, "uns") and "csScore" in csObject.uns:
        csScore = csObject.uns["csScore"]
        csScore.index = merged["CellID"]
        csScore.to_csv(folderPath / "csScore.csv")

    # scaled data
    merged.to_csv(folderPath / f"{imid}.csv", index=False)

    # Finish Job
    if verbose is True:
        print(
            'Contents of the csObject have been exported to "'
            + str(projectDir)
            + '/CSPOT/csExport"'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CSPOT results to CSV.")
    parser.add_argument(
        "--csObject", type=str, help="Path to the csObject file (.h5ad)", required=True
    )
    parser.add_argument(
        "--projectDir", type=str, help="Path to the output directory", required=True
    )
    parser.add_argument(
        "--fileName",
        type=str,
        help="Name of the CSV output file",
        default="csExport.csv",
    )
    parser.add_argument(
        "--raw",
        type=bool,
        default=False,
        help="Return raw data instead of CSPOT scaled data",
    )
    parser.add_argument(
        "--CellID",
        type=str,
        help="Column name that holds the cell ID",
        default="CellID",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Print detailed information about the process",
    )
    args = parser.parse_args()

    csExport(
        csObject=args.csObject,
        projectDir=args.projectDir,
        fileName=args.fileName,
        raw=args.raw,
        CellID=args.CellID,
        verbose=args.verbose,
    )
