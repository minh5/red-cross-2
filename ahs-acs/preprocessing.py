import os
import requests
import time

import matplotlib.pyplot as plt
from census import Census
import numpy as np
import pandas as pd
from slugify import slugify
from us import states


# using Census API to retrieve block groups for querying
c = Census(os.environ.get('CENSUS_KEY'), year=2016)
STATE_CODES = states.mapping("fips", "name")

BLOCKGROUPS = []
for state in STATE_CODES:
    temp = c.acs5.state_county("NAME", state, Census.ALL)
    BLOCKGROUPS.append({state : temp})

# pulling down variables from Census 
VAR_LOOKUP = requests.get("https://api.census.gov/data/2016/acs/acs5/variables.json").json()['variables']


def retrieve_block_group_data(group_name):
    """
    Retrieves data on the Census Block group level by the group name the variable is under.
    
    The purpose of this is that pulling all the variables takes too long, while
    pulling down only one variable at a time will result in too many tables to join on.
    This seems to be the approach that the Enigma group took when creating their tables. 

    Args:
        group_name: group name of the Census variables to query for
        type: string
        
    Returns:
        Pandas Dataframe

    Notes:
        This function takes two variables that are outside its scope:
            - VAR_LOOKUP is a dictionary of all the variables, and is used to subset the variables by group
            as well as changing the column name from its Census ID form to a more readable label
            - BLOCKGROUPS is a dictionary that contains the geographical metadata from each block group
            that is used for querying
        So be sure to have these variables in their global scope prior to running the function. This design
        is to make it parallelizable via Dask.
    """
    variables = {k: v for k,v in VAR_LOOKUP.items() if v['group'] == group_name}
    variables = list(variables.keys())
    BLKGROUPDATA = []
    for state in BLOCKGROUPS:
        for key, value in state.items():
            for element in value:
                results = None
                while results is None:
                    try:
                        results = c.acs5.state_county_blockgroup(
                            variables, 
                            element["state"], 
                            element["county"], 
                            Census.ALL
                        )
                        BLKGROUPDATA.append(results)
                    except Exception as e:
                        print(
                            'error calling api for {state}, {county}, sleeping...'.format(
                                state=element['state'],
                                county=element['county']
                            )
                        )
                        print(e)
                        time.sleep(3)
                        continue
    BLKGROUPDATA = [item for sublist in BLKGROUPDATA for item in sublist]
    data = pd.DataFrame.from_dict(BLKGROUPDATA)
    new_cols = [
        slugify(
            VAR_LOOKUP[col]['label']
        ) 
       for col in data.columns if col not in ['block group','county','state','tract']
    ] 
    data.columns = new_cols + ['block group','county','state','tract']
    return data


def main():
    from dask import delayed
    from dask.distributed import Client

    client = Client()
    
    var_df = pd.DataFrame.from_dict(VAR_LOOKUP).T
    # parallel execution using Dask workers, would probably change up the function to push to a database
    inputs = var_df.group.value_counts().index.values.tolist()
    values = [delayed(retrieve_block_group_data)(x) for x in inputs]


if __name__ == '__main__':
    main()