from datasets import load_dataset
from tensorflow import keras
import time
import json
import signal
import sys
import pdb
import pandas as pd
import re
import subprocess

method = 'naive'

# first wait till temperature is below 50C
while True:
    # wait till the temperature is below 50C
    cmd = 'nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    temp, err = p.communicate()
    if int(temp) < 50:
        break
    else:
        time.sleep(5)

# Now launch inference service
#cmd = 'sudo nvidia-smi -i 0 -pm 1'
#subprocess.Popen([cmd], shell=True).communicate(input='456852@Kb\n')

cmd = 'python bert_lat.py {method}'
pid = subprocess.Popen([cmd], shell=True).pid

pdb.set_trace()

# Start continuous temperature monitor (no needed for naive)
Tstart = time.time()
temp_list = []
time_limit = 1800 # 30min
while True:
    cmd = 'nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    temp, err = p.communicate()
    temp_list.append(temp)
    if time.time() - Tstart >= time_limit:
        break
    else:
        time.sleep(5)

cmd = f'kill -2 {pid}'
subprocess.Popen([cmd], shell=True)

with open(f'logs/{method}_temp_bert.json', 'w') as f:
    json.dump(temp_list, f, indent=4)

