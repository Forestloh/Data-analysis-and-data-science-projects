import json
import os

curr_py_path = os.getcwd()
path = '\\DI_CONNECT\\'
path_addon = 'DI-Connect-Fitness\\'
file_name = 'loh_hong_sen@hotmail.com_personalRecord.json'

path = curr_py_path + path + path_addon + file_name
# make path to raw string
raw_string = r'{}'.format(path)

# read file
with open(raw_string, 'r', encoding="utf8") as f_handle:
  data = json.load(f_handle)
  #data.reverse()
  print(json.dumps(data, indent=4, ensure_ascii=False))
