# -*- coding: utf-8 -*-
#Created on Wed Jan 25 15:53:02 2023
#@author: ajn16
#CSPOT Phenotype


"""
!!! abstract "Short Description"
    The csPhenotype function requires a phenotype workflow document to guide 
    the algorithm in performing classification.  
    
    The phenotype workflow document is imported as a `dataframe` and passed to the 
    `phenotype` argument. It should follow the following structure:  
        
    (1) The `first column` has to contain the cell that are to be classified.  
    (2) The `second column` indicates the phenotype a particular cell will be assigned 
    if it satifies the conditions in the row.  
    (3) `Column three` and onward represent protein markers. If the protein marker 
    is known to be expressed for that cell type, then it is denoted by either `pos`, 
    `allpos`. If the protein marker is known to not express for a cell type it 
    can be denoted by `neg`, `allneg`. If the protein marker is irrelevant or 
    uncertain to express for a cell type, then it is left empty. `anypos` and 
    `anyneg` are options for using a set of markers and if any of the marker is 
    positive or negative, the cell type is denoted accordingly.  
        
    To give users maximum flexibility in identifying desired cell types, 
    we have implemented various classification arguments as described above 
    for strategical classification. They include  
    
    - allpos
    - allneg
    - anypos
    - anyneg
    - pos
    - neg
    
    `pos` : "Pos" looks for cells positive for a given marker. If multiple 
    markers are annotated as `pos`, all must be positive to denote the cell type. 
    For example, a Regulatory T cell can be defined as `CD3+CD4+FOXP3+` by passing 
    `pos` to each marker. If one or more markers don't meet the criteria (e.g. CD4-), 
    the program will classify it as `Likely-Regulatory-T cell`, pending user 
    confirmation. This is useful in cases of technical artifacts or when cell 
    types (such as cancer cells) are defined by marker loss (e.g. T-cell Lymphomas).  
    
    `neg` : Same as `pos` but looks for negativity of the defined markers.  
    
    `allpos` : "Allpos" requires all defined markers to be positive. Unlike 
    `pos`, it doesn't classify cells as `Likely-cellType`, but strictly annotates 
    cells positive for all defined markers.  
    
    `allneg` : Same as `allpos` but looks for negativity of the defined markers.  
    
    `anypos` : "Anypos" requires only one of the defined markers to be positive. 
    For example, to define macrophages, a cell could be designated as such if 
    any of `CD68`, `CD163`, or `CD206` is positive.  
    
    `anyneg` : Same as `anyneg` but looks for negativity of the defined markers. 
            
## Function
"""

# Library
import numpy as np
import pandas as pd
import anndata as ad
import argparse
import pathlib
import os


# Function
def csPhenotype (csObject,
                    phenotype,
                    midpoint = 0.5,
                    label="phenotype",
                    imageid='imageid',
                    pheno_threshold_percent=None,
                    pheno_threshold_abs=None,
                    fileName=None,
                    verbose=True,
                    projectDir=None):
    """
Parameters:
    csObject (anndata):
        Single or combined CSPOT object.

    phenotype (dataframe, str):
        A phenotyping workflow strategy either as a `dataframe` loaded into memory or 
        a path to the `.csv` file. 

    midpoint (float, optional):
        By default, CSPOT normalizes the data in a way that cells with a value
        above 0.5 are considered positive. However, if you desire more selective
        threshold, the parameter can be adjusted accordingly. 

    label (str, optional):
        Specify the column name under which the final phenotype classification
        will be saved. 

    imageid (str, optional):
        The name of the column that holds the unique image ID. 

    pheno_threshold_percent (float, optional):
        The user-defined threshold, which can be set between 0-100, is used to
        recategorize any phenotype that falls below it as 'unknown'.
        This function is commonly used to address low background false positives.

    pheno_threshold_abs (int, optional):
        This function serves a similar purpose as the `pheno_threshold_percent`,
        but it accepts an absolute number as input. For example, if the user
        inputs 10, any phenotype that contains less than 10 cells will be
        recategorized as 'unknown'. 

    fileName (string, optional):
        File Name to be used while saving the CSPOT object.

    verbose (bool, optional):
        If True, print detailed information about the process to the console.  

    projectDir (string, optional):
        Provide the path to the output directory.
        
Returns:
    csObject (anndata):
        Modified CSPOT object with the Phenotypes is returned. If `projectDir` is 
        provided, it will be saved in the defined directory.

Example:
        ```python
        
        # set the Project directory
        projectDir = '/Users/aj/Documents/cspotExampleData'
        # Path to the CSPOT Object
        csObject = projectDir + '/CSPOT/csObject/exampleImage_cspotPredict.ome.h5ad'
        
        # load the phenotyping workflow
        phenotype = pd.read_csv(str(projectDir) + '/phenotype_workflow.csv')
        
        # Run Function
        adata = cs.csPhenotype ( csObject=csObject,
                            phenotype=phenotype,
                            midpoint = 0.5,
                            label="phenotype",
                            imageid='imageid',
                            pheno_threshold_percent=None,
                            pheno_threshold_abs=None,
                            fileName=None,
                            projectDir=projectDir)
        
        # Same function if the user wants to run it via Command Line Interface
        python csPhenotype.py \
            --csObject /Users/aj/Documents/cspotExampleData/CSPOT/csObject/exampleImage_cspotPredict.ome.h5ad \
            --phenotype /Users/aj/Documents/cspotExampleData/phenotype_workflow.csv \
            --projectDir /Users/aj/Documents/cspotExampleData

        
        ```

    """

    # Load data
    if isinstance(csObject, str):
        adata = ad.read(csObject)
    else:
        adata = csObject.copy()
    
    # load phenotype
    if isinstance(phenotype, pd.DataFrame):
        phenotype = phenotype
    else:
        phenotype = pd.read_csv(pathlib.Path(phenotype))

    # Create a dataframe from the adata object
    data = pd.DataFrame(adata.X, columns = adata.var.index, index= adata.obs.index)

    # Function to calculate the phenotype scores
    def phenotype_cells (data,phenotype,midpoint,group):

        # Subset the phenotype based on the group
        phenotype = phenotype[phenotype.iloc[:,0] == group]

        # Parser to parse the CSV file into four categories
        def phenotype_parser (p, cell):
            # Get the index and subset the phenotype row being passed in
            location = p.iloc[:,1] == cell
            idx = [i for i, x in enumerate(location) if x][0]
            phenotype = p.iloc[idx,:]
            # Calculate
            pos = phenotype[phenotype == 'pos'].index.tolist()
            neg = phenotype[phenotype == 'neg'].index.tolist()
            anypos = phenotype[phenotype == 'anypos'].index.tolist()
            anyneg = phenotype[phenotype == 'anyneg'].index.tolist()
            allpos = phenotype[phenotype == 'allpos'].index.tolist()
            allneg = phenotype[phenotype == 'allneg'].index.tolist()
            return {'pos': pos, 'neg': neg ,'anypos': anypos, 'anyneg': anyneg, 'allpos': allpos, 'allneg': allneg}
            #return pos, neg, anypos, anyneg

        # Run the phenotype_parser function on all rows
        p_list = phenotype.iloc[:,1].tolist()
        r_phenotype = lambda x: phenotype_parser(cell=x, p=phenotype) # Create lamda function
        all_phenotype = list(map(r_phenotype, p_list)) # Apply function
        all_phenotype = dict(zip(p_list, all_phenotype)) # Name the lists

        # Define function to check if there is any marker that does not satisfy the midpoint
        def gate_satisfation_lessthan (marker, data, midpoint):
            fail = np.where(data[marker] < midpoint, 1, 0) # 1 is fail
            return fail
        # Corresponding lamda function
        r_gate_satisfation_lessthan = lambda x: gate_satisfation_lessthan(marker=x, data=data, midpoint=midpoint)

        # Define function to check if there is any marker that does not satisfy the midpoint
        def gate_satisfation_morethan (marker, data, midpoint):
            fail = np.where(data[marker] > midpoint, 1, 0)
            return fail
        # Corresponding lamda function
        r_gate_satisfation_morethan = lambda x: gate_satisfation_morethan(marker=x, data=data, midpoint=midpoint)

        def prob_mapper (data, all_phenotype, cell, midpoint):
            if verbose is True:
                print("Phenotyping " + str(cell))

            # Get the appropriate dict from all_phenotype
            p = all_phenotype[cell]

            # Identiy the marker used in each category
            pos = p.get('pos')
            neg = p.get('neg')
            anypos = p.get('anypos')
            anyneg = p.get('anyneg')
            allpos = p.get('allpos')
            allneg = p.get('allneg')

            # Perform computation for each group independently
            # Positive marker score
            if len(pos) != 0:
                pos_score = data[pos].mean(axis=1).values
                pos_fail = list(map(r_gate_satisfation_lessthan, pos)) if len(pos) > 1 else []
                pos_fail = np.amax(pos_fail, axis=0) if len(pos) > 1 else []
            else:
                pos_score = np.repeat(0, len(data))
                pos_fail = []

            # Negative marker score
            if len(neg) != 0:
                neg_score = (1-data[neg]).mean(axis=1).values
                neg_fail = list(map(r_gate_satisfation_morethan, neg)) if len(neg) > 1 else []
                neg_fail = np.amax(neg_fail, axis=0) if len(neg) > 1 else []
            else:
                neg_score = np.repeat(0, len(data))
                neg_fail = []

            # Any positive score
            anypos_score = np.repeat(0, len(data)) if len(anypos) == 0 else data[anypos].max(axis=1).values

            # Any negative score
            anyneg_score = np.repeat(0, len(data)) if len(anyneg) == 0 else (1-data[anyneg]).max(axis=1).values

            # All positive score
            if len(allpos) != 0:
                allpos_score = data[allpos]
                allpos_score['score'] = allpos_score.max(axis=1)
                allpos_score.loc[(allpos_score < midpoint).any(axis = 1), 'score'] = 0
                allpos_score = allpos_score['score'].values + 0.01 # A small value is added to give an edge over the matching positive cell
            else:
                allpos_score = np.repeat(0, len(data))


            # All negative score
            if len(allneg) != 0:
                allneg_score = 1- data[allneg]
                allneg_score['score'] = allneg_score.max(axis=1)
                allneg_score.loc[(allneg_score < midpoint).any(axis = 1), 'score'] = 0
                allneg_score = allneg_score['score'].values + 0.01
            else:
                allneg_score = np.repeat(0, len(data))


            # Total score calculation
            # Account for differences in the number of categories used for calculation of the final score
            number_of_non_empty_features = np.sum([len(pos) != 0,
                                                len(neg) != 0,
                                                len(anypos) != 0,
                                                len(anyneg) != 0,
                                                len(allpos) != 0,
                                                len(allneg) != 0])

            total_score = (pos_score + neg_score + anypos_score + anyneg_score + allpos_score + allneg_score) / number_of_non_empty_features

            return {cell: total_score, 'pos_fail': pos_fail ,'neg_fail': neg_fail}
            #return total_score, pos_fail, neg_fail


        # Apply the fuction to get the total score for all cell types
        r_prob_mapper = lambda x: prob_mapper (data=data, all_phenotype=all_phenotype, cell=x, midpoint=midpoint) # Create lamda function
        final_scores = list(map(r_prob_mapper, [*all_phenotype])) # Apply function
        final_scores = dict(zip([*all_phenotype], final_scores)) # Name the lists

        # Combine the final score to annotate the cells with a label
        final_score_df = pd.DataFrame()
        for i in [*final_scores]:
            df = pd.DataFrame(final_scores[i][i])
            final_score_df= pd.concat([final_score_df, df], axis=1)
        # Name the columns
        final_score_df.columns = [*final_scores]
        final_score_df.index = data.index
        # Add a column called unknown if all markers have a value less than the midpoint (0.5)
        unknown = group + str('-rest')
        final_score_df[unknown] = (final_score_df < midpoint).all(axis=1).astype(int)

        # Name each cell
        labels = final_score_df.idxmax(axis=1)

        # Group all failed instances (i.e. when multiple markers were given
        # any one of the marker fell into neg or pos zones of the midpoint)
        pos_fail_all = pd.DataFrame()
        for i in [*final_scores]:
            df = pd.DataFrame(final_scores[i]['pos_fail'])
            df.columns = [i] if len(df) != 0 else []
            pos_fail_all= pd.concat([pos_fail_all, df], axis=1)
        pos_fail_all.index = data.index if len(pos_fail_all) != 0 else []
        # Same for Neg
        neg_fail_all = pd.DataFrame()
        for i in [*final_scores]:
            df = pd.DataFrame(final_scores[i]['neg_fail'])
            df.columns = [i] if len(df) != 0 else []
            neg_fail_all= pd.concat([neg_fail_all, df], axis=1)
        neg_fail_all.index = data.index if len(neg_fail_all) != 0 else []


        # Modify the labels with the failed annotations
        if len(pos_fail_all) != 0:
            for i in pos_fail_all.columns:
                labels[(labels == i) & (pos_fail_all[i] == 1)] = 'likely-' + i
        # Do the same for negative
        if len(neg_fail_all) != 0:
            for i in neg_fail_all.columns:
                labels[(labels == i) & (neg_fail_all[i] == 1)] = 'likely-' + i

        # Retun the labels
        return labels

    # Create an empty dataframe to hold the labeles from each group
    phenotype_labels = pd.DataFrame()

    # Loop through the groups to apply the phenotype_cells function
    for i in phenotype.iloc[:,0].unique():

        if phenotype_labels.empty:
            phenotype_labels = pd.DataFrame(phenotype_cells(data = data, group = i, phenotype=phenotype, midpoint=midpoint))
            phenotype_labels.columns = [i]

        else:
            # Find the column with the cell-type of interest
            column_of_interest = [] # Empty list to hold the column name
            try:
                column_of_interest = phenotype_labels.columns[phenotype_labels.eq(i).any()]
            except:
                pass
            # If the cell-type of interest was not found just add NA
            if len(column_of_interest) == 0:
                phenotype_labels[i] = np.nan
            else:
                #cells_of_interest = phenotype_labels[phenotype_labels[column_of_interest] == i].index
                cells_of_interest = phenotype_labels[phenotype_labels[column_of_interest].eq(i).any(axis=1)].index
                d = data.loc[cells_of_interest]
                if verbose is True:
                    print("-- Subsetting " + str(i))
                phenotype_l = pd.DataFrame(phenotype_cells(data = d, group = i, phenotype=phenotype, midpoint=midpoint), columns = [i])
                phenotype_labels = phenotype_labels.merge(phenotype_l, how='outer', left_index=True, right_index=True)

    # Rearrange the rows back to original
    phenotype_labels = phenotype_labels.reindex(data.index)
    phenotype_labels = phenotype_labels.replace('-rest', np.nan, regex=True)
    if verbose is True:
        print("Consolidating the phenotypes across all groups")
        phenotype_labels_Consolidated = phenotype_labels.fillna(method='ffill', axis = 1)
    phenotype_labels[label] = phenotype_labels_Consolidated.iloc[:,-1].values

    # replace nan to 'other cells'
    phenotype_labels[label] = phenotype_labels[label].fillna('Unknown')

    # Apply the phenotype threshold if given
    if pheno_threshold_percent or pheno_threshold_abs is not None:
        p = pd.DataFrame(phenotype_labels[label])
        q = pd.DataFrame(adata.obs[imageid])
        p = q.merge(p, how='outer', left_index=True, right_index=True)

        # Function to remove phenotypes that are less than the given threshold
        def remove_phenotype(p, ID, pheno_threshold_percent, pheno_threshold_abs):
            d = p[p[imageid] == ID]
            x = pd.DataFrame(d.groupby([label]).size())
            x.columns = ['val']
            # FInd the phenotypes that are less than the given threshold
            if pheno_threshold_percent is not None:
                fail = list(x.loc[x['val'] < x['val'].sum() * pheno_threshold_percent/100].index)
            if pheno_threshold_abs is not None:
                fail = list(x.loc[x['val'] < pheno_threshold_abs].index)
            d[label] = d[label].replace(dict(zip(fail, np.repeat('Unknown',len(fail)))))
            # Return
            return d

        # Apply function to all images
        r_remove_phenotype = lambda x: remove_phenotype (p=p, ID=x,
                                                         pheno_threshold_percent=pheno_threshold_percent,
                                                         pheno_threshold_abs=pheno_threshold_abs) # Create lamda function
        final_phrnotypes= list(map(r_remove_phenotype, list(p[imageid].unique()))) # Apply function

        final_phrnotypes = pd.concat(final_phrnotypes, join='outer')
        phenotype_labels = final_phrnotypes.reindex(adata.obs.index)


    # Return to adata
    adata.obs[label] = phenotype_labels[label]

    # Save data if requested
    if projectDir is not None:
        
        finalPath = pathlib.Path(projectDir + '/CSPOT/csPhenotype/')
        if not os.path.exists(finalPath):
            os.makedirs(finalPath)
        # determine file name
        if fileName is None:
            if isinstance (csObject, str):
                imid = pathlib.Path(csObject).stem
            else:
                imid = 'csPhenotype'
        else:
            imid = fileName
    
        adata.write(finalPath / f'{imid}.h5ad')
        # Print
        if verbose is True:
            print('Modified csObject is stored at "' + str(projectDir) + '/CSPOT/csPhenotype')

    else:
        # Return data
        return adata
    return adata


# Make the Function CLI compatable
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSPOT Phenotype')
    parser.add_argument('--csObject', type=str, help='Single or combined CSPOT object')
    parser.add_argument('--phenotype', type=str, help='A phenotyping workflow strategy')
    parser.add_argument('--midpoint', type=float, default=0.5, help='Normalization threshold')
    parser.add_argument('--label', type=str, default='phenotype', help='Column name for final phenotype classification')
    parser.add_argument('--imageid', type=str, default='imageid', help='Name of column that holds unique image ID')
    parser.add_argument('--pheno_threshold_percent', type=float, default=None, help='User-defined threshold for recategorizing phenotypes as "unknown"')
    parser.add_argument('--pheno_threshold_abs', type=int, default=None, help='User-defined threshold for recategorizing phenotypes as "unknown"')
    parser.add_argument('--fileName', type=str, default=None, help="File name for saving the modified csObject.")
    parser.add_argument("--verbose", type=bool, default=True, help="If True, print detailed information about the process to the console.")
    parser.add_argument('--projectDir', type=str, help="Directory to save the modified csObject.")
    args = parser.parse_args()
    csPhenotype(csObject=args.csObject,
                   phenotype=args.phenotype,
                   midpoint=args.midpoint,
                   label=args.label,
                   imageid=args.imageid,
                   pheno_threshold_percent=args.pheno_threshold_percent,
                   pheno_threshold_abs=args.pheno_threshold_abs,
                   fileName=args.fileName, 
                   verbose=args.verbose,
                   projectDir=args.projectDir)
    
    
    
