# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 17:05:16 2024

@author: cavus
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout
from xgboost import XGBRegressor

# Paths to the data files and output directory
data_file_path = r'C:\Users\cavus\Desktop\Dilum_Paper_2\Data_2019_EV_infrastructure.csv'
output_dir = r'C:\Users\cavus\Desktop\Dilum_Paper_2\EDA_Results_2'

os.makedirs(output_dir, exist_ok=True)

# Load the data
data_df = pd.read_csv(data_file_path)

# Set a color palette
sns.set_palette("husl")

# 1. Basic Descriptive Statistics
def save_descriptive_statistics(data, output_dir):
    desc_stats = data.describe()
    desc_stats.to_csv(os.path.join(output_dir, 'basic_descriptive_statistics.csv'))

save_descriptive_statistics(data_df, output_dir)

# 2. Distributions of Numeric Variables
def plot_numeric_distributions(data, numeric_cols, output_dir):
    for start in range(0, len(numeric_cols), 20):
        plt.figure(figsize=(15, 10))
        end = min(start + 20, len(numeric_cols))
        for i, col in enumerate(numeric_cols[start:end], 1):
            plt.subplot(5, 4, i)
            sns.histplot(data[col].dropna(), kde=True, color=sns.color_palette()[i % len(sns.color_palette())])
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'numeric_distributions_part_{start // 20 + 1}.png'), dpi=600)
        plt.close()

numeric_cols = data_df.select_dtypes(include=['int64', 'float64']).columns
numeric_cols = numeric_cols.drop(['subject', 'scenario', 'freq'])
plot_numeric_distributions(data_df, numeric_cols, output_dir)

# 3. Pair Plots for Charging Hours and Important Variables
def plot_pairplot(data, output_dir):
    pairplot = sns.pairplot(data, vars=['home_chg', 'work_chg', 'fasttime', 'hsincome', 'dmileage', 'long_dist'], hue='gender', palette="husl")
    pairplot._legend.set_bbox_to_anchor((1, 0.5))
    pairplot._legend.set_title("Gender")
    for text in pairplot._legend.texts:
        text.set_fontsize(16)
    pairplot.savefig(os.path.join(output_dir, 'pairplot_charging_behavior.png'), dpi=600)
    plt.close()

plot_pairplot(data_df, output_dir)

# 4. Boxplots of Charging Hours by Demographic Variables
def plot_boxplots_by_demographic(data, demographic_columns, charging_columns, output_dir):
    for dem_col in demographic_columns:
        plt.figure(figsize=(15, 5))
        for i, charge_col in enumerate(charging_columns, 1):
            plt.subplot(1, 3, i)
            sns.boxplot(x=dem_col, y=charge_col, data=data, palette="husl")
            plt.title(f'{charge_col} by {dem_col}', fontsize=16)
            plt.xlabel(dem_col, fontsize=16)
            plt.ylabel(charge_col, fontsize=16)
            plt.xticks(rotation=45, fontsize=16)
            plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'boxplot_charging_by_{dem_col}.png'), dpi=600)
        plt.close()

demographic_columns = ['gender', 'race', 'edu', 'employment', 'hsincome', 'hhsize', 'housit', 'residence']
charging_columns = ['home_chg', 'work_chg', 'fasttime']
plot_boxplots_by_demographic(data_df, demographic_columns, charging_columns, output_dir)

# 5. ANOVA Analysis with Post-Hoc Tests
def perform_anova_with_posthoc(data, dep_var, group_var):
    groups = [group[dep_var].dropna() for name, group in data.groupby(group_var)]
    f_val, p_val = f_oneway(*groups)
    posthoc_results = pairwise_tukeyhsd(endog=data[dep_var].dropna(), groups=data[group_var].dropna(), alpha=0.05)
    return f_val, p_val, posthoc_results

anova_results = {}

for dem_col in demographic_columns:
    for charge_col in charging_columns:
        f_val, p_val, posthoc_results = perform_anova_with_posthoc(data_df, charge_col, dem_col)
        anova_results[(charge_col, dem_col)] = (f_val, p_val, posthoc_results)

# Save ANOVA results to CSV and Tukey HSD to text files
anova_results_df = pd.DataFrame.from_dict({k: v[:2] for k, v in anova_results.items()}, orient='index', columns=['F-value', 'p-value'])
anova_results_df.to_csv(os.path.join(output_dir, 'anova_results.csv'))
anova_results_df['Significant'] = anova_results_df['p-value'] < 0.05

# Save Tukey HSD results
for key, (f_val, p_val, posthoc_results) in anova_results.items():
    with open(os.path.join(output_dir, f'tukey_hsd_{key[0]}_by_{key[1]}.txt'), 'w') as file:
        file.write(str(posthoc_results))

# 6. Scatter Plot with Regression Line
def plot_scatter_with_regression(data, x_var, y_var, output_dir):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=x_var, y=y_var, data=data, scatter_kws={'s': 10}, line_kws={"color": "red"})
    plt.title(f'Regression: {y_var} vs {x_var}', fontsize=16)
    plt.xlabel(x_var, fontsize=16)
    plt.ylabel(y_var, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'scatter_regression_{x_var}_vs_{y_var}.png'), dpi=600)
    plt.close()

# Example scatter plots with regression lines
plot_scatter_with_regression(data_df, 'hsincome', 'home_chg', output_dir)
plot_scatter_with_regression(data_df, 'dmileage', 'work_chg', output_dir)

# 7. Fasttime Descriptive Statistics
def fasttime_statistics(data, output_dir):
    fasttime_stats = data['fasttime'].describe()
    fasttime_stats.to_csv(os.path.join(output_dir, 'fasttime_statistics.csv'))

fasttime_statistics(data_df, output_dir)

# 8. Fasttime Distribution Plot
def plot_fasttime_distribution(data, output_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['fasttime'].dropna(), kde=True, color="purple")
    plt.title('Distribution of Fast Charging Time (fasttime)')
    plt.xlabel('Fast Charging Time (minutes)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fasttime_distribution.png'), dpi=600)
    plt.close()

plot_fasttime_distribution(data_df, output_dir)

# 9. Boxplot of Fasttime by Demographic Variables
def plot_fasttime_by_demographics(data, demographic_columns, output_dir):
    plt.figure(figsize=(15, 8))
    for i, dem_col in enumerate(demographic_columns, 1):
        plt.subplot(2, 4, i)
        sns.boxplot(x=dem_col, y='fasttime', data=data, palette="husl")
        plt.xlabel(dem_col, fontsize=16)
        plt.ylabel('Fast Time Charging (minutes)', fontsize=16)
        plt.xticks(rotation=45, fontsize=16)
        plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fasttime_by_demographics.png'), dpi=600)
    plt.close()

plot_fasttime_by_demographics(data_df, demographic_columns, output_dir)

# 10. Regression Analysis: Predicting Fasttime
def regression_analysis_fasttime(data, output_dir):
    # Preparing the data for regression analysis
    independent_vars = ['hsincome', 'dmileage', 'home_chg', 'work_chg', 'long_dist', 'PopDensity']
    
    # Drop NA values from independent and dependent variables
    data_subset = data.dropna(subset=independent_vars + ['fasttime'])
    
    X = data_subset[independent_vars]
    y = data_subset['fasttime']

    X = sm.add_constant(X) 
    # Add a constant term to the predictors
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Save the regression results
    with open(os.path.join(output_dir, 'fasttime_regression_results.txt'), 'w') as file:
        file.write(model.summary().as_text())

    # Plot the regression residuals
    plt.figure(figsize=(10, 6))
    sns.residplot(x=model.predict(X), y=model.resid, lowess=True, color="g")
    plt.xlabel('Fitted values', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fasttime_regression_residuals.png'), dpi=600)
    plt.close()

regression_analysis_fasttime(data_df, output_dir)

# 11. Post-Hoc Tukey Test for Fasttime by Demographics
def posthoc_tukey_fasttime(data, demographic_columns, output_dir):
    for dem_col in demographic_columns:
        posthoc = pairwise_tukeyhsd(data['fasttime'].dropna(), data[dem_col].dropna(), alpha=0.05)
        with open(os.path.join(output_dir, f'posthoc_tukey_fasttime_by_{dem_col}.txt'), 'w') as file:
            file.write(str(posthoc))

posthoc_tukey_fasttime(data_df, demographic_columns, output_dir)

# ==============================
# Machine Learning and Deep Learning Models
# ==============================

# Handling non-numeric data by encoding categorical features
features = pd.get_dummies(data_df.drop(columns=['home_chg', 'work_chg', 'fasttime']), drop_first=True)
targets = data_df[['home_chg', 'work_chg', 'fasttime']]

# Normalize the features and target values using MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(targets)

# Split data into training and testing sets (last 24 records for testing)
train_size = len(data_df) - 24
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

# Reshape data for RNN and CNN models
X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Function to create deep learning models
def create_model(model_type):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(50, activation='relu', input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
    elif model_type == 'GRU':
        model.add(GRU(50, activation='relu', input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
    elif model_type == 'CNN':
        model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())
    elif model_type == 'LSTM+CNN':
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
        model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())
    elif model_type == 'GRU+CNN':
        model.add(GRU(50, activation='relu', return_sequences=True, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
        model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())
    elif model_type == 'CNN+LSTM':
        model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
        model.add(MaxPooling1D(pool_size=1))
        model.add(LSTM(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to train and predict with model
def train_and_predict(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1, shuffle=False)
    predictions = model.predict(X_test)
    if len(predictions.shape) == 3:
        predictions = predictions.reshape(predictions.shape[0], predictions.shape[2])
    return predictions

# Function to create and train XGBoost models
def train_xgboost(X_train, y_train, X_test):
    xgb_predictions = {}
    for i, target in enumerate(['home_chg', 'work_chg', 'fasttime']):
        xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train, y_train[:, i])
        xgb_predictions[target] = xgb_model.predict(X_test)
    return np.column_stack([xgb_predictions['home_chg'], xgb_predictions['work_chg'], xgb_predictions['fasttime']])

# Combined CNN+LSTM+XGBoost Implementation
def train_cnn_lstm_xgboost(X_train, y_train, X_test, y_test):
    cnn_lstm_model = create_model('CNN+LSTM')
    cnn_lstm_predictions_train = train_and_predict(cnn_lstm_model, X_train, y_train, X_train, y_train)
    cnn_lstm_predictions_test = train_and_predict(cnn_lstm_model, X_train, y_train, X_test, y_test)
    
    xgb_predictions = {}
    for i, target in enumerate(['home_chg', 'work_chg', 'fasttime']):
        xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        xgb_model.fit(cnn_lstm_predictions_train, y_train[:, i])
        xgb_predictions[target] = xgb_model.predict(cnn_lstm_predictions_test)
    return np.column_stack([xgb_predictions['home_chg'], xgb_predictions['work_chg'], xgb_predictions['fasttime']])

# Function to plot predictions for each model
def plot_predictions_side_by_side(predictions, model_name, output_dir):
    charging_types = ['home_chg', 'work_chg', 'fasttime']
    y_labels = ['Home charging [min]', 'Work charging [min]', 'Fast-time charging [min]']
    
    # Define colors for each model
    color_map = {
        'LSTM': 'red',
        'GRU': 'green',
        'CNN': 'blue',
        'XGBoost': 'purple',
        'LSTM+CNN': 'orange',
        'GRU+CNN': 'cyan',
        'Deep Charge-Fusion Model': 'magenta'
    }
    
    # Adjust model name if necessary
    if model_name == 'CNN+LSTM+XGBoost':
        model_name = 'Deep Charge-Fusion Model'
    
    plt.figure(figsize=(20, 8))  # Increased figure size
    
    for i, (charge_type, y_label) in enumerate(zip(charging_types, y_labels), 1):
        ax = plt.subplot(1, 3, i)
        plt.stairs(y_test_actual[:, i-1], label='Actual', color='black', linewidth=2)
        plt.stairs(predictions[:, i-1], label=f'{model_name}', color=color_map[model_name], linestyle='--', linewidth=2)
        plt.xlabel('Participants', fontsize=18)
        plt.ylabel(y_label, fontsize=18)
        
        # Remove the legend from individual subplots
        if i == 1:
            handles, labels = ax.get_legend_handles_labels()

    plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=0.1)  # Adjusting padding to remove excess space

    # Add a single, centered legend below the subplots
    plt.figlegend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=18)
    
    output_path = os.path.join(output_dir, f'{model_name}_predictions.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

# Example usage with the predictions and model names
output_dir = r'C:\Users\cavus\OneDrive\Masaüstü\Dilum_Paper_2\Model_Comparison_Results_5'
# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Train and predict using different models
model_names = ['LSTM', 'GRU', 'CNN', 'XGBoost', 'LSTM+CNN', 'GRU+CNN', 'CNN+LSTM+XGBoost']
models = [create_model(name) for name in model_names[:-1]]
predictions = [train_and_predict(model, X_train_rnn, y_train, X_test_rnn, y_test) for model in models]

# Add CNN+LSTM+XGBoost predictions
predictions.append(train_cnn_lstm_xgboost(X_train_rnn, y_train, X_test_rnn, y_test))

# Evaluate models
y_test_actual = scaler_y.inverse_transform(y_test)
predictions_rescaled = [scaler_y.inverse_transform(pred.reshape(pred.shape[0], -1)) for pred in predictions]

# Plot predictions for each model side by side
for pred_rescaled, model_name in zip(predictions_rescaled, model_names):
    plot_predictions_side_by_side(pred_rescaled, model_name, output_dir)

# Function to compare R² and MAE results
def plot_metrics_comparison(r2_scores, mae_scores, model_names, output_dir):
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.bar(x - width/2, r2_scores, width, label='R² Score', color='blue')
    ax1.set_ylabel('R² Score', fontsize=14)
    ax1.set_xlabel('Models', fontsize=14)
    ax1.set_title('R² and MAE Comparison Across Models', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)
    ax1.legend(loc='upper left', fontsize=12)

    ax2 = ax1.twinx()
    
    # Adjust the MAE scores by multiplying each element by 0.1
    adjusted_mae_scores = [score * 0.1 for score in mae_scores]
    
    ax2.bar(x + width/2, adjusted_mae_scores, width, label='MAE', color='red')
    ax2.set_ylabel('MAE', fontsize=14)
    ax2.legend(loc='upper right', fontsize=12)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_mae_comparison.png'), dpi=600)
    plt.close()

# Calculate R² and MAE for each model
r2_scores = []
mae_scores = []
for pred_rescaled in predictions_rescaled:
    r2_scores.append(r2_score(y_test_actual, pred_rescaled))
    mae_scores.append(mean_absolute_error(y_test_actual, pred_rescaled))

# Plot R² and MAE comparison with the adjusted MAE scores
plot_metrics_comparison(r2_scores, mae_scores, model_names, output_dir)

# Output R² and MAE scores
print("Model Performance Metrics:")
for i, model_name in enumerate(model_names):
    print(f"{model_name}: R² Score = {r2_scores[i]:.4f}, MAE = {mae_scores[i]:.4f}")
