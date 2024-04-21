#!/usr/bin/python    
import os
from jproperties import Properties



env = os.environ.get('ENV', 'local')
# Read the properties file

configs = Properties()

with open('src/configs/' + env + '-config.properties', 'rb') as config_file:
    configs.load(config_file, "utf-8")

items_view = configs.items()
list_keys = []

for item in items_view:
    list_keys.append(item[0])

print('props: ', list_keys)  

# get the value of a property
def get(key):
    return configs[key].data


