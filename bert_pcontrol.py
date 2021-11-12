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

gpu_type = sys.argv[1]

# first wait till temperature is below 40C


# read support clock
clocks = pd.read_csv('supported_clock.csv')
clocks = clocks[clocks['memory [MHz]']=='5001 MHz']
app_clks = clocks[' graphics [MHz]'].tolist()
app_clks = [int(re.findall(r'\d+', k)[0]) for k in app_clks][::-1]
app_clks = [k for k in app_clks if k > 400 and k < 1300]

# enable pm on nvidia-smi
cmd = 'sudo nvidia-smi -i 0 -pm 1'
subprocess.Popen([cmd], shell=True).communicate(input='456852@Kb\n')

for clk in app_clks:
    while True:
        # wait till the temperature is below 50C
        cmd = 'nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader'
        p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
        temp, err = p.communicate()
        if int(temp) < 50:
            break
        else:
            cmd = f'sudo nvidia-smi -i 0 -rac'
            subprocess.Popen([cmd], shell=True).communicate(input='456852@Kb\n')
            # reset application clock to cool down
            time.sleep(20)
    batch_time[clk] = []
    cmd = f'sudo nvidia-smi -i 0 -ac 5001,{clk}'
    subprocess.Popen([cmd], shell=True).communicate(input='456852@Kb\n')
    print(clk)
    for data in eval_tf_dataset:
        start_time = time.time()
        model.predict_on_batch(data)
        duration = round(time.time() - start_time,3)
        batch_time[clk].append(duration)

with open(f'logs/clk_vs_lat/{gpu_type}_duration_bert_lat8.json', 'w') as f:
    json.dump(batch_time, f, indent=4)
with open(f'logs/clk_vs_lat/{gpu_type}_timestamp_bert_lat8.json', 'w') as f:
    json.dump(iter_time, f, indent=4)

