'''
    This script updates the _attenList.json file according to current atten files.

    https://github.com/z0gSh1u/crip
'''

__all__ = []

import os
from os import path
import json

if __name__ == '__main__':
    Folders = ['compound', 'mixture', 'simple']
    _AttenPath = path.dirname(path.abspath(__file__))
    Dict = {}

    for folder in Folders:
        for file in os.listdir(path.join(_AttenPath, folder)):
            Dict[file.replace('.txt', '')] = folder

    JSONPath = path.join(_AttenPath, './_attenList.json')
    with open(JSONPath, 'w') as fp:
        json.dump(Dict, fp, indent=2)