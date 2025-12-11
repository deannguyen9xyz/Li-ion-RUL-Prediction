import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#=================
# 1. Load EIS data
#=================
scripts_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(scripts_dir)
data_path = os.path.join(base_dir, "data", "battery_dataset.csv")

df = pd.read_csv(data_path, header=0)

#=======================
# 2. Correlation Analysis
#=======================

# 1. Define the features to analyze (the operational parameters)
feature_cols = ['cycle', 'chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'BCt', 'RUL']

# 2. Calculate the Pearson correlation matrix
correlation_matrix = df[feature_cols].corr(method='pearson')

# 3. Extract the correlation coefficients against RUL
rul_correlations = correlation_matrix['RUL'].drop('RUL').sort_values(ascending=False)

print("\n--- Correlation Coefficients with RUL ---")
print(rul_correlations)

# 4. Visualization (Heatmap)
os.makedirs('figures', exist_ok=True)

plt.figure(figsize=(8, 6))
# We plot the full correlation matrix for a comprehensive view
sns.heatmap(correlation_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={'label': 'Pearson Correlation Coefficient'}
           )

plt.title('Pearson Correlation Matrix of Operational Features and RUL')
plt.show()