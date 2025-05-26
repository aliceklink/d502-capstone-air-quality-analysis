#!/usr/bin/env python3
"""
D502 Capstone: PM2.5 vs Asthma Correlation Analysis - Main Statistical Analysis
Complete statistical analysis pipeline for air quality and asthma correlation

Author: Alice Klink
Date: May 2025

Research Question: Is there a statistically significant correlation between 
PM2.5 air pollution concentrations and asthma prevalence rates in US counties?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, ttest_ind, normaltest, levene, shapiro
import warnings
from pathlib import Path
from datetime import datetime
import os

# Configure plotting
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

def print_separator(title):
    """Print a formatted separator with title"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def load_data():
    """
    Load the processed and merged dataset
    
    Returns:
    DataFrame: Merged PM2.5 and asthma data
    """
    print("Loading processed data...")
    
    data_path = 'data/processed/merged_pm25_asthma.csv'
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("   Please run the data collection script first: python src/data_collection.py")
        return None
    
    try:
        data = pd.read_csv(data_path)
        print(f"Data loaded successfully: {len(data):,} counties")
        
        # Basic data info
        print(f"   PM2.5 range: {data['PM25_Annual_Mean'].min():.1f} - {data['PM25_Annual_Mean'].max():.1f} μg/m³")
        print(f"   Asthma range: {data['Asthma_Prevalence'].min():.1f} - {data['Asthma_Prevalence'].max():.1f}%")
        print(f"   States represented: {data['State Name'].nunique() if 'State Name' in data.columns else 'N/A'}")
        
        return data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def data_quality_check(data):
    """
    Perform comprehensive data quality assessment
    
    Parameters:
    data (DataFrame): The merged dataset
    
    Returns:
    DataFrame: Clean data ready for analysis
    """
    print_separator("DATA QUALITY ASSESSMENT")
    
    print("Checking data quality...")
    
    initial_count = len(data)
    
    # Check for missing values
    missing_pm25 = data['PM25_Annual_Mean'].isna().sum()
    missing_asthma = data['Asthma_Prevalence'].isna().sum()
    
    print(f"Missing data:")
    print(f"   PM2.5: {missing_pm25} ({missing_pm25/len(data)*100:.1f}%)")
    print(f"   Asthma: {missing_asthma} ({missing_asthma/len(data)*100:.1f}%)")
    
    # Remove missing values
    clean_data = data.dropna(subset=['PM25_Annual_Mean', 'Asthma_Prevalence']).copy()
    
    # Remove extreme outliers using IQR method
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return clean, outliers
    
    # Check PM2.5 outliers
    clean_data, pm25_outliers = remove_outliers(clean_data, 'PM25_Annual_Mean')
    print(f"PM2.5 outliers removed: {len(pm25_outliers)}")
    
    # Check asthma outliers
    clean_data, asthma_outliers = remove_outliers(clean_data, 'Asthma_Prevalence')
    print(f"Asthma outliers removed: {len(asthma_outliers)}")
    
    print(f"\nFinal clean dataset: {len(clean_data):,} counties ({len(clean_data)/initial_count*100:.1f}% of original)")
    
    return clean_data

def exploratory_analysis(data):
    """
    Comprehensive exploratory data analysis with visualizations
    
    Parameters:
    data (DataFrame): Clean dataset for analysis
    """
    print_separator("EXPLORATORY DATA ANALYSIS")
    
    print("Generating descriptive statistics and visualizations...")
    
    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print("\nPM2.5 Annual Mean (μg/m³):")
    pm25_stats = data['PM25_Annual_Mean'].describe()
    print(pm25_stats)
    
    print("\nAsthma Prevalence (%):")
    asthma_stats = data['Asthma_Prevalence'].describe()
    print(asthma_stats)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Distribution plots
    plt.subplot(2, 4, 1)
    plt.hist(data['PM25_Annual_Mean'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of PM2.5 Annual Mean', fontweight='bold')
    plt.xlabel('PM2.5 (μg/m³)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 2)
    plt.hist(data['Asthma_Prevalence'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Distribution of Asthma Prevalence', fontweight='bold')
    plt.xlabel('Asthma Prevalence (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Box plots
    plt.subplot(2, 4, 3)
    plt.boxplot(data['PM25_Annual_Mean'], patch_artist=True, boxprops=dict(facecolor='skyblue'))
    plt.title('PM2.5 Box Plot', fontweight='bold')
    plt.ylabel('PM2.5 (μg/m³)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 4)
    plt.boxplot(data['Asthma_Prevalence'], patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    plt.title('Asthma Prevalence Box Plot', fontweight='bold')
    plt.ylabel('Asthma Prevalence (%)')
    plt.grid(True, alpha=0.3)
    
    # Scatter plot with regression line
    plt.subplot(2, 4, 5)
    plt.scatter(data['PM25_Annual_Mean'], data['Asthma_Prevalence'], 
               alpha=0.6, color='darkgreen', s=20)
    
    # Add regression line
    z = np.polyfit(data['PM25_Annual_Mean'], data['Asthma_Prevalence'], 1)
    p = np.poly1d(z)
    plt.plot(data['PM25_Annual_Mean'], p(data['PM25_Annual_Mean']), 
             "r--", alpha=0.8, linewidth=2)
    
    plt.title('PM2.5 vs Asthma Prevalence', fontweight='bold')
    plt.xlabel('PM2.5 Annual Mean (μg/m³)')
    plt.ylabel('Asthma Prevalence (%)')
    plt.grid(True, alpha=0.3)
    
    # Q-Q plots for normality assessment
    plt.subplot(2, 4, 6)
    stats.probplot(data['PM25_Annual_Mean'], dist="norm", plot=plt)
    plt.title('PM2.5 Q-Q Plot', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 7)
    stats.probplot(data['Asthma_Prevalence'], dist="norm", plot=plt)
    plt.title('Asthma Q-Q Plot', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Correlation heatmap (if additional variables available)
    plt.subplot(2, 4, 8)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    
    # Focus on key variables for readability
    key_vars = ['PM25_Annual_Mean', 'Asthma_Prevalence']
    if 'PM25_Sample_Count' in corr_matrix.columns:
        key_vars.append('PM25_Sample_Count')
    
    sns.heatmap(corr_matrix.loc[key_vars, key_vars], 
                annot=True, cmap='coolwarm', center=0, 
                square=True, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix', fontweight='bold')
    
    plt.suptitle('Exploratory Data Analysis: PM2.5 and Asthma Prevalence', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Ensure results directory exists
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('results/figures/exploratory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional summary statistics
    print(f"\nData Distribution Summary:")
    print(f"PM2.5 skewness: {stats.skew(data['PM25_Annual_Mean']):.3f}")
    print(f"PM2.5 kurtosis: {stats.kurtosis(data['PM25_Annual_Mean']):.3f}")
    print(f"Asthma skewness: {stats.skew(data['Asthma_Prevalence']):.3f}")
    print(f"Asthma kurtosis: {stats.kurtosis(data['Asthma_Prevalence']):.3f}")

def correlation_analysis(data):
    """
    Comprehensive Pearson correlation analysis
    
    Parameters:
    data (DataFrame): Clean dataset
    
    Returns:
    tuple: correlation coefficient and p-value
    """
    print_separator("CORRELATION ANALYSIS")
    
    print("Performing Pearson correlation analysis...")
    
    # Calculate Pearson correlation
    correlation_coef, p_value = pearsonr(data['PM25_Annual_Mean'], data['Asthma_Prevalence'])
    n_samples = len(data)
    
    print(f"\nPearson Correlation Results:")
    print(f"  Correlation coefficient (r): {correlation_coef:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Sample size (n): {n_samples:,}")
    
    # Interpret correlation strength
    abs_corr = abs(correlation_coef)
    if abs_corr < 0.1:
        strength = "negligible"
    elif abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.5:
        strength = "moderate"
    elif abs_corr < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if correlation_coef > 0 else "negative"
    
    print(f"  Interpretation: {strength} {direction} correlation")
    
    # Statistical significance
    alpha = 0.05
    if p_value < alpha:
        print(f"  Result: Statistically significant (p < {alpha})")
        print("  We reject the null hypothesis.")
        print("  There IS a significant correlation between PM2.5 and asthma prevalence.")
    else:
        print(f"  Result:  Not statistically significant (p ≥ {alpha})")
        print("  We fail to reject the null hypothesis.")
        print("  No significant correlation detected.")
    
    # Confidence interval for correlation (Fisher's z-transformation)
    z = np.arctanh(correlation_coef)
    se = 1/np.sqrt(n_samples-3)
    ci_lower = np.tanh(z - 1.96*se)
    ci_upper = np.tanh(z + 1.96*se)
    
    print(f"  95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Effect size interpretation (Cohen's conventions for correlation)
    print(f"\nEffect Size Assessment:")
    if abs_corr < 0.1:
        effect_size = "trivial effect"
    elif abs_corr < 0.3:
        effect_size = "small effect"
    elif abs_corr < 0.5:
        effect_size = "medium effect"
    else:
        effect_size = "large effect"
    
    print(f"  Cohen's interpretation: {effect_size}")
    print(f"  Coefficient of determination (r²): {correlation_coef**2:.4f}")
    print(f"  {correlation_coef**2*100:.1f}% of variance in asthma is associated with PM2.5")
    
    # Create detailed correlation visualization
    plt.figure(figsize=(12, 8))
    
    # Main scatter plot
    plt.subplot(2, 2, (1, 2))
    plt.scatter(data['PM25_Annual_Mean'], data['Asthma_Prevalence'], 
               alpha=0.6, color='darkblue', s=40, edgecolors='white', linewidth=0.5)
    
    # Add regression line with confidence interval
    x = data['PM25_Annual_Mean']
    y = data['Asthma_Prevalence']
    
    # Regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r-", alpha=0.8, linewidth=2, 
             label=f'r = {correlation_coef:.3f}, p = {p_value:.4f}')
    
    plt.xlabel('PM2.5 Annual Mean (μg/m³)', fontsize=12)
    plt.ylabel('Asthma Prevalence (%)', fontsize=12)
    plt.title(f'Correlation Analysis: PM2.5 vs Asthma Prevalence\n'
              f'r = {correlation_coef:.3f}, p = {p_value:.4f}, n = {n_samples:,}', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals plot
    plt.subplot(2, 2, 3)
    predicted = p(x)
    residuals = y - predicted
    plt.scatter(predicted, residuals, alpha=0.6, color='green', s=30)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Predicted Asthma Prevalence (%)')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    # Histogram of residuals
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_coef, p_value

def hypothesis_testing(data):
    """
    Comprehensive hypothesis testing using two-sample t-test
    
    Parameters:
    data (DataFrame): Clean dataset
    
    Returns:
    tuple: t-statistic, p-value, and effect size
    """
    print_separator("HYPOTHESIS TESTING")
    
    print("Performing two-sample t-test analysis...")
    
    # Split data into high and low PM2.5 groups based on median
    pm25_median = data['PM25_Annual_Mean'].median()
    high_pm25 = data[data['PM25_Annual_Mean'] > pm25_median]['Asthma_Prevalence']
    low_pm25 = data[data['PM25_Annual_Mean'] <= pm25_median]['Asthma_Prevalence']
    
    print(f"\nGroup Definitions:")
    print(f"  High PM2.5 group (> {pm25_median:.2f} μg/m³): {len(high_pm25):,} counties")
    print(f"  Low PM2.5 group (≤ {pm25_median:.2f} μg/m³): {len(low_pm25):,} counties")
    
    # Descriptive statistics for each group
    print(f"\nDescriptive Statistics:")
    print(f"  High PM2.5 group:")
    print(f"    Mean asthma prevalence: {high_pm25.mean():.3f}%")
    print(f"    Standard deviation: {high_pm25.std():.3f}%")
    print(f"    Median: {high_pm25.median():.3f}%")
    
    print(f"  Low PM2.5 group:")
    print(f"    Mean asthma prevalence: {low_pm25.mean():.3f}%")
    print(f"    Standard deviation: {low_pm25.std():.3f}%")
    print(f"    Median: {low_pm25.median():.3f}%")
    
    mean_difference = high_pm25.mean() - low_pm25.mean()
    print(f"  Mean difference: {mean_difference:.3f}%")
    
    # Test assumptions
    print(f"\nTesting Statistical Assumptions:")
    
    # Normality tests
    _, p_norm_high = normaltest(high_pm25)
    _, p_norm_low = normaltest(low_pm25)
    
    print(f"  Normality tests (D'Agostino-Pearson):")
    print(f"    High PM2.5 group: p = {p_norm_high:.4f}")
    print(f"    Low PM2.5 group: p = {p_norm_low:.4f}")
    
    normality_met = (p_norm_high > 0.05) and (p_norm_low > 0.05)
    print(f"    Normality assumption: {'Met' if normality_met else '⚠️ Violated'}")
    
    # Equal variances test (Levene's test)
    _, p_levene = levene(high_pm25, low_pm25)
    print(f"  Equal variances test (Levene): p = {p_levene:.4f}")
    
    equal_var = p_levene > 0.05
    print(f"    Equal variances assumption: {'Met' if equal_var else '⚠️ Violated'}")
    
    # Perform appropriate t-test
    t_stat, p_value = ttest_ind(high_pm25, low_pm25, equal_var=equal_var)
    
    print(f"\nTwo-Sample T-Test Results:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    
    degrees_freedom = len(high_pm25) + len(low_pm25) - 2
    print(f"  Degrees of freedom: {degrees_freedom:,}")
    
    # Effect size (Cohen's d)
    if equal_var:
        # Pooled standard deviation
        pooled_std = np.sqrt(((len(high_pm25)-1)*high_pm25.var() + (len(low_pm25)-1)*low_pm25.var()) / 
                            (len(high_pm25) + len(low_pm25) - 2))
    else:
        # Separate variances
        pooled_std = np.sqrt((high_pm25.var() + low_pm25.var()) / 2)
    
    cohens_d = mean_difference / pooled_std
    print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
    
    # Interpret effect size
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect_interpretation = "negligible effect"
    elif abs_d < 0.5:
        effect_interpretation = "small effect"
    elif abs_d < 0.8:
        effect_interpretation = "medium effect"
    else:
        effect_interpretation = "large effect"
    
    print(f"  Effect size interpretation: {effect_interpretation}")
    
    # Statistical significance
    alpha = 0.05
    print(f"\nHypothesis Test Conclusion:")
    if p_value < alpha:
        print(f"  Result:  Statistically significant difference (p < {alpha})")
        print(f"  ➤ Counties with higher PM2.5 have significantly different asthma rates.")
        if mean_difference > 0:
            print(f"  ➤ Higher PM2.5 counties have {mean_difference:.2f}% higher asthma prevalence on average.")
        else:
            print(f"  ➤ Higher PM2.5 counties have {-mean_difference:.2f}% lower asthma prevalence on average.")
    else:
        print(f"  Result: No statistically significant difference (p ≥ {alpha})")
        print(f"  ➤ No significant difference in asthma rates between high and low PM2.5 counties.")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Box plot comparison
    plt.subplot(2, 3, 1)
    box_data = [low_pm25, high_pm25]
    bp = plt.boxplot(box_data, labels=['Low PM2.5', 'High PM2.5'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    plt.ylabel('Asthma Prevalence (%)')
    plt.title('Asthma Prevalence by PM2.5 Level')
    plt.grid(True, alpha=0.3)
    
    # Violin plot
    plt.subplot(2, 3, 2)
    plt.violinplot([low_pm25, high_pm25], positions=[1, 2], showmeans=True)
    plt.xticks([1, 2], ['Low PM2.5', 'High PM2.5'])
    plt.ylabel('Asthma Prevalence (%)')
    plt.title('Distribution Comparison (Violin Plot)')
    plt.grid(True, alpha=0.3)
    
    # Histogram comparison
    plt.subplot(2, 3, 3)
    plt.hist(low_pm25, alpha=0.7, label='Low PM2.5', bins=20, color='lightblue', density=True)
    plt.hist(high_pm25, alpha=0.7, label='High PM2.5', bins=20, color='lightcoral', density=True)
    plt.xlabel('Asthma Prevalence (%)')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Q-Q plots for normality
    plt.subplot(2, 3, 4)
    stats.probplot(low_pm25, dist="norm", plot=plt)
    plt.title('Q-Q Plot: Low PM2.5 Group')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    stats.probplot(high_pm25, dist="norm", plot=plt)
    plt.title('Q-Q Plot: High PM2.5 Group')
    plt.grid(True, alpha=0.3)
    
    # Mean comparison with error bars
    plt.subplot(2, 3, 6)
    means = [low_pm25.mean(), high_pm25.mean()]
    stds = [low_pm25.std(), high_pm25.std()]
    plt.bar(['Low PM2.5', 'High PM2.5'], means, yerr=stds, 
            color=['lightblue', 'lightcoral'], alpha=0.7, capsize=5)
    plt.ylabel('Mean Asthma Prevalence (%)')
    plt.title('Mean Comparison with Standard Deviation')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Hypothesis Testing: Two-Sample T-Test Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/hypothesis_testing.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return t_stat, p_value, cohens_d

def generate_final_report(data, correlation_results, hypothesis_results):
    """
    Generate comprehensive final analysis report
    
    Parameters:
    data (DataFrame): Clean dataset
    correlation_results (tuple): correlation coefficient and p-value
    hypothesis_results (tuple): t-statistic, p-value, and effect size
    """
    print_separator("FINAL ANALYSIS REPORT")
    
    correlation_coef, correlation_p = correlation_results
    t_stat, t_test_p, effect_size = hypothesis_results
    
    print("D502 CAPSTONE: PM2.5 vs ASTHMA CORRELATION ANALYSIS")
    print("    Statistical Analysis Report")
    print("    Author: Alice Klink")
    print(f"    Date: {datetime.now().strftime('%B %d, %Y')}")
    
    print(f"\nRESEARCH QUESTION:")
    print(f"Is there a statistically significant correlation between PM2.5 air")
    print(f"pollution concentrations and asthma prevalence rates in US counties?")
    
    print(f"\nDATASET SUMMARY:")
    print(f"  Total counties analyzed: {len(data):,}")
    print(f"  PM2.5 concentration range: {data['PM25_Annual_Mean'].min():.2f} - {data['PM25_Annual_Mean'].max():.2f} μg/m³")
    print(f"  PM2.5 mean: {data['PM25_Annual_Mean'].mean():.2f} μg/m³ (SD: {data['PM25_Annual_Mean'].std():.2f})")
    print(f"  Asthma prevalence range: {data['Asthma_Prevalence'].min():.2f} - {data['Asthma_Prevalence'].max():.2f}%")
    print(f"  Asthma mean: {data['Asthma_Prevalence'].mean():.2f}% (SD: {data['Asthma_Prevalence'].std():.2f})")
    
    if 'State Name' in data.columns:
        print(f"  Geographic coverage: {data['State Name'].nunique()} states")
    
    print(f"\nKEY FINDINGS:")
    
    print(f"\n1. CORRELATION ANALYSIS:")
    print(f"   • Pearson correlation coefficient: r = {correlation_coef:.4f}")
    print(f"   • Statistical significance: p = {correlation_p:.6f}")
    print(f"   • Coefficient of determination: r² = {correlation_coef**2:.4f}")
    print(f"   • Variance explained: {correlation_coef**2*100:.1f}%")
    
    if correlation_p < 0.05:
        significance_result = "STATISTICALLY SIGNIFICANT"
        correlation_conclusion = "There IS a significant correlation"
    else:
        significance_result = "NOT STATISTICALLY SIGNIFICANT"
        correlation_conclusion = "There is NO significant correlation"
    
    print(f"   • Result: {significance_result}")
    print(f"   • Interpretation: {correlation_conclusion} between PM2.5 and asthma prevalence")
    
    print(f"\n2. HYPOTHESIS TESTING (Two-Sample T-Test):")
    pm25_median = data['PM25_Annual_Mean'].median()
    high_pm25_mean = data[data['PM25_Annual_Mean'] > pm25_median]['Asthma_Prevalence'].mean()
    low_pm25_mean = data[data['PM25_Annual_Mean'] <= pm25_median]['Asthma_Prevalence'].mean()
    mean_diff = high_pm25_mean - low_pm25_mean
    
    print(f"   • Group comparison (median split at {pm25_median:.2f} μg/m³):")
    print(f"     - High PM2.5 counties: {high_pm25_mean:.2f}% asthma prevalence")
    print(f"     - Low PM2.5 counties: {low_pm25_mean:.2f}% asthma prevalence")
    print(f"     - Mean difference: {mean_diff:.2f}%")
    print(f"   • t-statistic: {t_stat:.4f}")
    print(f"   • Statistical significance: p = {t_test_p:.6f}")
    print(f"   • Effect size (Cohen's d): {effect_size:.4f}")
    
    if t_test_p < 0.05:
        t_test_result = "STATISTICALLY SIGNIFICANT DIFFERENCE"
    else:
        t_test_result = "NO STATISTICALLY SIGNIFICANT DIFFERENCE"
    
    print(f"   • Result: {t_test_result}")
    
    print(f"\nOVERALL CONCLUSION:")
    
    if correlation_p < 0.05 and t_test_p < 0.05:
        print(f"STRONG EVIDENCE of association between PM2.5 and asthma prevalence:")
        print(f"   • Both correlation analysis and group comparison show significant results")
        print(f"   • Counties with higher PM2.5 levels tend to have different asthma rates")
        if correlation_coef > 0:
            print(f"   • The relationship is POSITIVE: higher PM2.5 associated with higher asthma")
        else:
            print(f"   • The relationship is NEGATIVE: higher PM2.5 associated with lower asthma")
    elif correlation_p < 0.05 or t_test_p < 0.05:
        print(f"MIXED EVIDENCE of association:")
        print(f"   • One test shows significance, the other does not")
        print(f"   • Results suggest a possible but not definitive association")
    else:
        print(f"NO SIGNIFICANT EVIDENCE of association:")
        print(f"   • Neither correlation nor group comparison shows significance")
        print(f"   • Cannot conclude there is a meaningful relationship")
    
    print(f"\nLIMITATIONS:")
    print(f"   • Ecological study design - county-level associations may not apply to individuals")
    print(f"   • Cross-sectional analysis - cannot establish causation")
    print(f"   • Potential confounding variables not controlled (socioeconomic, healthcare access)")
    print(f"   • Data from different collection periods may introduce temporal bias")
    
    print(f"\nIMPLICATIONS:")
    print(f"   • Results contribute to understanding air quality-health relationships")
    print(f"   • Findings relevant for public health policy and environmental regulation")
    print(f"   • Geographic patterns may inform targeted interventions")
    
    print(f"\nRECOMMENDATIONS FOR FUTURE RESEARCH:")
    print(f"   • Individual-level studies to confirm ecological associations")
    print(f"   • Longitudinal analysis to establish temporal relationships")
    print(f"   • Control for socioeconomic and healthcare access variables")
    print(f"   • Investigate other air pollutants and health outcomes")
    
    # Save comprehensive results
    results_summary = pd.DataFrame({
        'Analysis_Type': ['Dataset Summary', 'Correlation Analysis', 'Correlation Analysis',
                         'Correlation Analysis', 'Hypothesis Testing', 'Hypothesis Testing', 
                         'Hypothesis Testing'],
        'Metric': ['Sample Size', 'Correlation Coefficient', 'Correlation P-value', 
                  'R-squared', 'T-statistic', 'T-test P-value', 'Effect Size (Cohen\'s d)'],
        'Value': [len(data), correlation_coef, correlation_p, correlation_coef**2,
                 t_stat, t_test_p, effect_size],
        'Interpretation': [f'{len(data):,} counties', 
                          'Moderate positive' if 0.3 <= correlation_coef < 0.5 else 'Weak positive' if 0.1 <= correlation_coef < 0.3 else 'Strong positive' if correlation_coef >= 0.5 else 'Negative',
                          'Significant' if correlation_p < 0.05 else 'Not significant',
                          f'{correlation_coef**2*100:.1f}% variance explained',
                          'Positive difference' if t_stat > 0 else 'Negative difference',
                          'Significant' if t_test_p < 0.05 else 'Not significant',
                          'Large effect' if abs(effect_size) >= 0.8 else 'Medium effect' if abs(effect_size) >= 0.5 else 'Small effect']
    })
    
    # Ensure results directory exists
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    
    results_summary.to_csv('results/tables/analysis_results_summary.csv', index=False)
    data.to_csv('results/tables/final_analysis_dataset.csv', index=False)
    
    print(f"\nFILES SAVED:")
    print(f"   • results/tables/analysis_results_summary.csv")
    print(f"   • results/tables/final_analysis_dataset.csv")
    print(f"   • results/figures/exploratory_analysis.png")
    print(f"   • results/figures/correlation_analysis.png") 
    print(f"   • results/figures/hypothesis_testing.png")

def main():
    """
    Execute the complete statistical analysis pipeline
    """
    print("="*70)
    print("D502 CAPSTONE: PM2.5 vs ASTHMA CORRELATION ANALYSIS")
    print("Statistical Analysis Pipeline")
    print("="*70)
    
    try:
        # Step 1: Load data
        data = load_data()
        if data is None:
            return False
        
        # Step 2: Data quality assessment
        clean_data = data_quality_check(data)
        
        # Step 3: Exploratory data analysis
        exploratory_analysis(clean_data)
        
        # Step 4: Correlation analysis
        correlation_results = correlation_analysis(clean_data)
        
        # Step 5: Hypothesis testing
        hypothesis_results = hypothesis_testing(clean_data)
        
        # Step 6: Final report
        generate_final_report(clean_data, correlation_results, hypothesis_results)
        
        print_separator("ANALYSIS COMPLETE")
        print("Statistical analysis completed successfully!")
        print(f"   Dataset: {len(clean_data):,} counties analyzed")
        print(f"   Results saved in results/ directory")
        print(f"   Ready for capstone report writing!")
        
        return True
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\nAnalysis pipeline completed successfully!")
        print(f"Next steps:")
        print(f"1. Review the generated figures and tables")
        print(f"2. Use results for your capstone report")
        print(f"3. Consider additional analyses if needed")
    else:
        print(f"\nAnalysis pipeline failed.")
        print(f"Please check error messages and data files.")