import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#=================
# 1. Load EIS data
#=================
scripts_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(scripts_dir)
data_path = os.path.join(base_dir, "data", "battery_dataset.csv")

df = pd.read_csv(data_path, header=0)

#=================
# 2. BCt vs cycle
#=================

for name, group in df.groupby('battery_id'):
    plt.plot(group['cycle'], group['BCt'], marker='o', linestyle='-', label=f'Battery {name}')

plt.title('Capacity Fade (BCt) vs. Cycle Number for Li-ion Cells')
plt.xlabel('Cycle Number')
plt.ylabel('Discharge Capacity (BCt, Ah)')
plt.legend(title='Battery ID')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()

#=================
# 2. Internal resistance (R_int) vs cycle    
#=================

#R_int = (chV - disV) / (chI + disI)
df['DeltaV'] = df['chV'] - df['disV']
df['DeltaI'] = df['chI'] + df['disI']
df['R_int'] = df['DeltaV'] / df['DeltaI']

plt.figure(figsize=(10, 6))

for name, group in df.groupby('battery_id'):
    plt.plot(group['cycle'], group['R_int'], marker='o', linestyle='-', label=f'Battery {name}')

plt.title('Internal Resistance ($R_{int}$) vs. Cycle Number')
plt.xlabel('Cycle Number')
plt.ylabel('Internal Resistance ($\\Omega$)')
plt.legend(title='Battery ID')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()