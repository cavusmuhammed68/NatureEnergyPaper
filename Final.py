# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:55:34 2024

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

# Paths to the data files and output directory
data_file_path = r'C:\Users\cavus\Desktop\Dilum_Paper_2\Data_2019_EV_infrastructure.csv'
output_dir = r'C:\Users\cavus\Desktop\Dilum_Paper_2\EDA_Results'

os.makedirs(output_dir, exist_ok=True)

# Load the data
data_df = pd.read_csv(data_file_path)

# Verify column names
print(data_df.columns)

# Rename columns for better plot labels
data_df = data_df.rename(columns={
    'home_chg': 'Home Charging',
    'work_chg': 'Work Charging',
    'fasttime': 'Fast-time Charging',
    'edu': 'Education',
    'employment': 'Employment',
    'gender': 'Gender',
    'hhsize': 'Household size',
    'hsincome': 'House income',
    'housit': 'House situation',
    'residence': 'Residence',
    'race': 'Race'  # Ensure this matches the actual column name in your data
})

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
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'numeric_distributions_part_{start // 20 + 1}.png'), dpi=600)
        plt.close()

numeric_cols = data_df.select_dtypes(include=['int64', 'float64']).columns
numeric_cols = numeric_cols.drop(['subject', 'scenario', 'freq'], errors='ignore')  # Adjust if these columns are not present
plot_numeric_distributions(data_df, numeric_cols, output_dir)

# 3. Pair Plots for Charging Hours and Important Variables
def plot_pairplot(data, output_dir):
    pairplot = sns.pairplot(data, 
                            vars=['Home Charging', 'Work Charging', 'Fast-time Charging', 'Education', 'Employment', 'House income', 'Residence'], 
                            hue='Gender', 
                            palette="husl",
                            height=2.5,  # Adjust height
                            aspect=1.2,  # Adjust aspect ratio
                            plot_kws={'s': 10},  # Scatter plot point size
                            diag_kws={'shade': True})  # Density plot shading

    # Update labels and titles with fontsize
    for ax in pairplot.axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), fontsize=17)
        ax.set_ylabel(ax.get_ylabel(), fontsize=17)

    pairplot._legend.set_bbox_to_anchor((1, 0.5))
    pairplot._legend.set_title("Gender")
    
    # Increase legend font size
    for text in pairplot._legend.texts:
        text.set_fontsize(17)

    # Save the plot
    pairplot.savefig(os.path.join(output_dir, 'pairplot_charging_behavior.png'), dpi=600)
    plt.close()

plot_pairplot(data_df, output_dir)



# 4. Boxplots of Charging Hours by Demographic Variables
def sanitize_filename(filename):
    return "".join([c if c.isalnum() or c in (' ', '.', '_') else '_' for c in filename])

def plot_boxplots_by_demographic(data, demographic_columns, charging_columns, output_dir):
    for dem_col in demographic_columns:
        sanitized_dem_col = sanitize_filename(dem_col)
        plt.figure(figsize=(15, 5))
        for i, charge_col in enumerate(charging_columns, 1):
            plt.subplot(1, 3, i)
            sns.boxplot(x=dem_col, y=charge_col, data=data, palette="husl")
            plt.xlabel(dem_col, fontsize=16)
            plt.ylabel(f'{charge_col} [min]', fontsize=16)  # Added [min] to the y-axis label
            plt.xticks(rotation=45, fontsize=16)
            plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'boxplot_charging_by_{sanitized_dem_col}.png'), dpi=600)
        plt.close()


demographic_columns = ['Gender', 'Race', 'Education', 'Employment', 'House income', 'Household size', 'House situation', 'Residence']
charging_columns = ['Home Charging', 'Work Charging', 'Fast-time Charging']
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
    sanitized_dem_col = sanitize_filename(key[1])
    with open(os.path.join(output_dir, f'tukey_hsd_{key[0]}_by_{sanitized_dem_col}.txt'), 'w') as file:
        file.write(str(posthoc_results))

# 6. Scatter Plot with Regression Line
def plot_scatter_with_regression(data, x_var, y_var, output_dir):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=x_var, y=y_var, data=data, scatter_kws={'s': 10}, line_kws={"color": "red"})
    plt.xlabel(x_var, fontsize=14)
    plt.ylabel(f'{y_var} [min]', fontsize=14)  # Added [min] to the y-axis label
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'scatter_regression_{x_var}_vs_{y_var}.png'), dpi=600)
    plt.close()

# Example scatter plots with regression lines
plot_scatter_with_regression(data_df, 'House income', 'Home Charging', output_dir)
plot_scatter_with_regression(data_df, 'dmileage', 'Work Charging', output_dir)

# 7. Fasttime Descriptive Statistics
def fasttime_statistics(data, output_dir):
    fasttime_stats = data['Fast-time Charging'].describe()
    fasttime_stats.to_csv(os.path.join(output_dir, 'fasttime_statistics.csv'))

fasttime_statistics(data_df, output_dir)

# 8. Fasttime Distribution Plot
def plot_fasttime_distribution(data, output_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Fast-time Charging'].dropna(), kde=True, color="purple")
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
        sanitized_dem_col = sanitize_filename(dem_col)
        plt.subplot(2, 4, i)
        sns.boxplot(x=dem_col, y='Fast-time Charging', data=data, palette="husl")
        plt.xlabel(dem_col, fontsize=12)
        plt.ylabel('Fast-time Charging', fontsize=16)
        plt.xticks(rotation=45, fontsize=16)
        plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fasttime_by_demographics_{sanitized_dem_col}.png'), dpi=600)
    plt.close()

plot_fasttime_by_demographics(data_df, demographic_columns, output_dir)

# 10. Regression Analysis: Predicting Fasttime
def regression_analysis_fasttime(data, output_dir):
    # Preparing the data for regression analysis
    independent_vars = ['House income', 'dmileage', 'Home Charging', 'Work Charging', 'long_dist', 'PopDensity']
    
    # Drop NA values from independent and dependent variables
    data_subset = data.dropna(subset=independent_vars + ['Fast-time Charging'])
    
    # Extract independent variables and dependent variable
    X = data_subset[independent_vars]
    y = data_subset['Fast-time Charging']

    X = sm.add_constant(X)  # Adds a constant term to the predictors
    model = sm.OLS(y, X).fit()

    # Saving the regression results
    with open(os.path.join(output_dir, 'fasttime_regression_results.txt'), 'w') as file:
        file.write(model.summary().as_text())

    # Plotting the regression residuals
    plt.figure(figsize=(10, 6))
    sns.residplot(x=model.predict(X), y=model.resid, lowess=True, color="g")
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fasttime_regression_residuals.png'), dpi=600)
    plt.close()

regression_analysis_fasttime(data_df, output_dir)

# 11. Post-Hoc Tukey Test for Fasttime by Demographics
def posthoc_tukey_fasttime(data, demographic_columns, output_dir):
    for dem_col in demographic_columns:
        sanitized_dem_col = sanitize_filename(dem_col)
        posthoc = pairwise_tukeyhsd(data['Fast-time Charging'].dropna(), data[dem_col].dropna(), alpha=0.05)
        with open(os.path.join(output_dir, f'posthoc_tukey_fasttime_by_{sanitized_dem_col}.txt'), 'w') as file:
            file.write(str(posthoc))

posthoc_tukey_fasttime(data_df, demographic_columns, output_dir)
