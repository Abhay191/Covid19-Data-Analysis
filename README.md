
# COVID-19 Daily Cases Autoregression Model

This repository contains a project that builds an autoregression (AR) model to analyze and predict the daily COVID-19 cases in India from January 30, 2020, to October 2, 2021. The dataset consists of the rolling 7-day average of newly confirmed cases.

## Project Overview

### 1. Autocorrelation Analysis

1. **Line Plot of Daily COVID-19 Cases**
    - Create a line plot with the date on the x-axis and the number of COVID-19 cases on the y-axis.
    - Observe the first wave (around August 2020) and the second wave (May 2021) of COVID-19 in India.

2. **Autocorrelation with Lagged Values**
    - Generate a one-day lagged time sequence and compute the Pearson correlation coefficient.
    - Create a scatter plot between the original and one-day lagged sequences to visualize the correlation.
    - Generate multiple time sequences with different lag values (1 to 6 days) and compute the Pearson correlation coefficients. Create a line plot of these coefficients against the lag values.
    - Plot a correlogram or Auto-Correlation Function (ACF) using the `plot_acf` function from the `statsmodels` library.

### 2. Autoregression (AR) Model

1. **Data Split**
    - Split the data into training (65%) and test (35%) sets, ensuring the order of the sequence is maintained. The test set approximately covers the second wave of COVID-19.
    - Plot the training and test sets.

2. **AR Model Training and Prediction**
    - Use the `AutoReg` function from the `statsmodels` library to generate an AR model with 5 lagged values.
    - Train the model and obtain the coefficients.
    - Predict the test data using the trained model and compute metrics such as RMSE (%) and MAPE.
    - Create scatter and line plots to visualize the actual vs. predicted values.

### 3. AR Model with Different Lags

1. **Multiple AR Models**
    - Generate five AR models with lag values of 1, 5, 10, 15, and 25 days.
    - Compute RMSE (%) and MAPE for each model.
    - Create bar charts to compare the RMSE (%) and MAPE values across different lag values.

### 4. Optimal Lag Calculation

1. **Heuristic Optimal Lag Calculation**
    - Compute the optimal number of lags based on the condition `abs(AutoCorrelation) > 2/sqrt(T)`, where `T` is the number of observations in the training data.
    - Use this optimal lag value in the `AutoReg` function to predict the new COVID-19 cases.
    - Compare the RMSE (%) and MAPE values with those from the previous AR models.

## Files

- `daily_covid_cases.csv`: The dataset containing daily COVID-19 cases.
- `covid_ar_model.ipynb`: Jupyter notebook containing the full implementation of the analysis and AR model.
- `README.md`: This file.

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- statsmodels
- math

## Results

The analysis and AR model provide insights into the trends and patterns in COVID-19 cases in India. The model's performance is evaluated using metrics such as RMSE and MAPE, and the results are visualized using various plots.

## Conclusion

This project demonstrates the application of autoregression models in time series analysis and forecasting. The insights gained from this analysis can be used to understand the spread of COVID-19 and inform public health strategies.

---
