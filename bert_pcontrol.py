import time
import json
import signal
import sys
import pdb
import pandas as pd
import re
import subprocess

method = 'pctrl'

# read support clock
clocks = pd.read_csv('supported_clock.csv')
clocks = clocks[clocks['memory [MHz]']=='5001 MHz']
app_clks = clocks[' graphics [MHz]'].tolist()
app_clks = [int(re.findall(r'\d+', k)[0]) for k in app_clks][::-1]
app_clks = [k for k in app_clks if k >= 585 and k <= 1005]

def adjust_clk(curr_clk, curr_temp, set_point, app_clks):
    curr_ind = app_clks.index(curr_clk)
    diff = curr_temp - set_point
    new_ind = curr_ind - diff # if diff > 0, temp too high, set to lower index to reduce clock
    if new_ind > len(app_clks) - 1:
        return app_clks[-1]
    elif new_ind < 0:
        return app_clks[0]
    else:
        return app_clks[new_ind]

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

# Now launch inference service
cmd = 'sudo nvidia-smi -i 0 -pm 1'
subprocess.Popen([cmd], shell=True).communicate(input='456852@Kb\n')
cmd = 'sudo nvidia-smi -i 0 -ac 5001,1005' # starting clock by default
subprocess.Popen([cmd], shell=True).communicate(input='456852@Kb\n')

print('Start running inference')

cmd = f'python bert_lat.py {method}'
pid = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).pid

Tstart = time.time()
temp_list = []
clk_list = []
time_limit = 1800 # 30min
curr_clk = 1005
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
        # control clock
        curr_clk = adjust_clk(curr_clk, temp, 65, app_clks)
        cmd = f'sudo nvidia-smi -i 0 -ac 5001,{curr_clk}' # starting clock by default
        subprocess.Popen([cmd], shell=True).communicate(input='456852@Kb\n')
        time.sleep(5)

cmd = f'pkill -2 -P {pid}'
subprocess.Popen([cmd], shell=True)

with open(f'logs/{method}_temp_bert.json', 'w') as f:
    json.dump(temp_list, f, indent=4)
with open(f'logs/{method}_clk_bert.json', 'w') as f:
    json.dump(clk_list, f, indent=4)

