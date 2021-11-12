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
print('Waiting for temperature to drop...')
while True:
    # wait till the temperature is below 45C
    cmd = 'nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    temp, err = p.communicate()
    if int(temp) < 45:
        break
    else:
        time.sleep(5)

print('Start running inference')

cmd = f'python bert_lat.py {method}'
pid = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).pid

# Start continuous temperature monitor (no needed for naive)
Tstart = time.time()
temp_list = []
clk_list = []
time_limit = 1800 # 30min
while True:
    cmd = 'nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader,nounits'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    out, err = p.communicate()
    out = re.findall('\d+', str(out))
    temp = out[0]
    clk = out[1]
    temp_list.append(int(temp))
    clk_list.append(int(clk))
    if time.time() - Tstart >= time_limit:
        break
    else:
        time.sleep(5)

cmd = f'pkill -2 -P {pid}'
subprocess.Popen([cmd], shell=True)

with open(f'logs/{method}_temp_bert.json', 'w') as f:
    json.dump(temp_list, f, indent=4)
with open(f'logs/{method}_clk_bert.json', 'w') as f:
    json.dump(clk_list, f, indent=4)

