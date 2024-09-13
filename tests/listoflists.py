import tomllib
from sdbx.config import config_source_location
import os
from pathlib import Path
import tomllib
import ast

filename = Path(os.path.join(config_source_location, "classify.toml"))


def get_unique_lists(filename:str):
    d = {}
    count = 0
    keycount = 0
    f = {}
    print(filename)
    with open(filename, "rb") as block:
        data = tomllib.load(block)  # Load the TOML contents into `data`
        for key, value in data.items():  # Iterate over each item in the `data` dictionary
            keycount += 1
            d[count]=value.splitlines()
            for x in d.items():
                print(d[x])
                print(f[x])
            count +=1

    print(keycount)
    print(f)
            #print(value)



    # unique_lists = {}
    # for _, value in lists:
    #     if len(set(value)) == len(value):
    #         unique_list = set()
    #         for item in value:
    #             unique_list.add(item)
    #         unique_lists[key] = list(unique_list)
    #     else:
    #         return None
    
    # for key, value in unique_lists.items():
    #     print(key,value)
    # return 

get_unique_lists(filename)