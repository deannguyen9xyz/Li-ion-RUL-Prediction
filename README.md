# Li-ion-RUL-Prediction
A case study of Lithium-ion battery degradation using NASA datasets. Features data visualization, correlation, and Machine Learning models to predict Remaining Useful Life (RUL).

## 🌟 Key Features

Three Python script that transitions from basic capacity fade modeling to advanced Random Forest RUL prediction. Its main technical contribution is the cross-cell validation strategy which rigorously tests the model's ability to extrapolate to an unseen battery (B7), explicitly highlighting the challenge of data distribution shift in real-world scenarios.

---

## ⚙️ Requirements

Place the input CSV data file named **"battery_dataset.csv"** in the **"data"** directory alongside the Python script(s).

Install required libraries using:

```
pip install numpy scipy pandas matplotlib sklearn seaborn
```

---

## 📁 Project Files

| File Name | Location | Description |
| :--- | :--- | :--- |
| `battery_dataset.csv` | `data/` | **Raw cyclic voltammetry data** for input. |
| `01_visualization.py` | `scripts/` | Visualize the raw data to understand physical degradation. |
| `02_correlation.py` | `scripts/` | Mathematically prove which physical parameters drive the aging process. |
| `03_random_forest_regression.py` | `scripts/` | Train an ML model to predict RUL for an unseen battery. |

---

## 📊 Input Data

This dataset is modeled after the NASA Ames Prognostics Center of Excellence lithium-ion battery degradation dataset. It simulates the charge-discharge behavior and aging process of lithium-ion batteries across multiple cycles, capturing realistic trends in battery health over time.

The dataset features three virtual battery cells B0005 (B5), B0006 (B6) and B0007 (B7) and includes average values per cycle for key parameters such as:

*Charging/Discharging Current

*Charging/Discharging Voltage

*Charging/Discharging Temperature

*Battery Capacity (BCt)

*State of Health (SOH)

*Remaining Useful Life (RUL)

https://www.kaggle.com/datasets/programmer3/lithium-ion-battery-degradation-dataset?resource=download

---
## ▶️ How to Run

Run each script separately.

---

## 📈 Output

Each script generates figures in the *results/* directory.

---
## 📊 Conclusion

Script 1 (01_visualization) : All cells (B5, B6, B7) demonstrate expected capacity fade over their cycle life, confirming the degradation phenomenon.

Script 2 (02_correlation) : BCt and cycle are better raw predictors than chT, disT, disV, chV, chI, disI.

Script 3 (03_random_forest_regression): Taining on only two cells (B5 and B6) was insufficient to capture the variability in degradation paths. True RUL of B7 is significantly different (longer lifetime) than the B5/B6 training data, confirming that the model was forced to extrapolate outside its learned domain, leading to the high MAE.

---

## 🎯 Purpose of This Project

This project is designed for:

* Traditional Physics-Based Modeling: Establishing a baseline RUL prediction using linear capacity fade analysis.
* An simple example of using ML to predict battery RUL.
* GitHub portfolio demonstration.

---

## 📌 Future Improvements

* The noisy prediction curve suggests that future work should focus on data preprocessing to smooth the features and reduce model instability.
* Try other prediction models.

--- 

## 🧑‍💻 Author

Developed by: Vu Bao Chau Nguyen, Ph.D.

Keywords: Li-ion battery, Remaining Useful Life (RUL), Random Forest Regressor (RFR).

---
