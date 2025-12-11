import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

#=================
# 1. Load EIS data
#=================
scripts_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(scripts_dir)
data_path = os.path.join(base_dir, "data", "battery_dataset.csv")

df = pd.read_csv(data_path, header=0)

#=====================
# 2. Feature selection
#=====================

# X: Input Features (Predictors)
features = ['cycle', 'BCt', 'chT', 'disT', 'disV']

# Y: Target Variable
target = 'RUL'

X = df[features]
Y = df[target]

#=====================
# 2. Train/Test Split
#=====================

# Define Training Cells (Source Domain)
train_cells = ['B5', 'B6']
# Define Testing Cell (Target Domain)
test_cell = 'B7'

# Create the training set (B5 and B6)
X_train = df[df['battery_id'].isin(train_cells)][features]
Y_train = df[df['battery_id'].isin(train_cells)][target]

# Create the testing set (B7)
X_test = df[df['battery_id'] == test_cell][features]
Y_test = df[df['battery_id'] == test_cell][target]

print(f"Data split: Training on {train_cells} ({len(X_train)} samples).")
print(f"Predicting on unseen cell {test_cell} ({len(X_test)} samples).")

#=====================================
# 3. Model Initialization and Training
#    (Random Forest Regressor)
#=====================================

rfr_model = RandomForestRegressor(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1 
)

print("\nStarting Random Forest Model Training on B5 & B6...")
rfr_model.fit(X_train, Y_train)
print("Training complete.")

# --- Make Predictions on B7 ---
Y_pred = rfr_model.predict(X_test)

# --- Evaluation Metrics for B7 ---
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("\n--- Model Evaluation (Prediction on B7) ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} cycles")
print(f"R-squared (R2 Score): {r2:.4f}")

# --- Feature Importance (What the model relied on) ---
feature_importances = pd.Series(rfr_model.feature_importances_, index=features).sort_values(ascending=False)

print("\n--- Feature Importance (Most Useful Predictors) ---")
print(feature_importances)

#=============================================
# 4. Visualization: Raw vs. Predicted RUL (B7)
#=============================================

os.makedirs('figures', exist_ok=True)

# Create a DataFrame for plotting the comparison
comparison_df = X_test.copy()
comparison_df['True_RUL'] = Y_test
comparison_df['Predicted_RUL'] = Y_pred

plt.figure(figsize=(10, 6))

# Plot the True RUL Curve (Raw B7 data)
plt.plot(comparison_df['cycle'], comparison_df['True_RUL'], 
         label='True RUL (B7)', 
         color='darkblue', 
         linewidth=3)

# Plot the Predicted RUL Curve (Model's forecast for B7)
plt.plot(comparison_df['cycle'], comparison_df['Predicted_RUL'], 
         label='Predicted RUL (Model B5/B6)', 
         color='red', 
         linestyle='--', 
         linewidth=2)

# 1. Filter the original DataFrame 'df' to include only 'B5' and 'B6'
target_batteries = ['B5', 'B6']
df_plot = df[df['battery_id'].isin(target_batteries)]

# 2. Group the filtered DataFrame (df_plot) by battery_id and plot each group separately
for name, group in df_plot.groupby('battery_id'):
    
    # Use different markers/styles for clarity
    plt.plot(group['cycle'],  
             group['RUL'],  
             label=f'True RUL ({name})',
             linewidth=2)

# Customization
plt.title(f'RUL Prediction: Model Trained on B5/B6 vs. Unseen B7')
plt.xlabel('Cycle Number')
plt.ylabel('Remaining Useful Life (Cycles)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()

# --- Final Prediction Difference ---
print("\nPrediction analysis complete.")