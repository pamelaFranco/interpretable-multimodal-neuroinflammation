###############################################################################
# Statistical Analysis Pipeline for Multimodal Neuroinflammatory Data
# This script performs comprehensive descriptive statistics, correlation analysis,
# and group comparisons on clinical, cognitive, frailty, and inflammatory biomarkers.

#   Author:      Dr. Pamela Franco
#   Time-stamp:  2025-12-22
#   E-mail:      pamela.franco@unab.cl / pafranco@uc.cl
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data from Excel file
df = pd.read_excel("anestesia_ML.xlsx", sheet_name=0)

# Define variable categories
cognitive = ['MoCA', 'DgMoCA']  # Cognitive assessment variables
frailty = ['CFS', 'DgCFS', 'FRIED', 'DgFRIED']  # Frailty scale variables
inflammatory = ['IL6P', 'IL10P', 'VCAM']  # Inflammatory biomarker variables

# Extract neuroimaging variables (columns from MiddleCerebellarPeduncle to Tapetum_left_)
neuro_cols = df.columns[9:-1]  # From column 9 to second-to-last column

def summary_stats(df, cols):
    """
    Generate descriptive statistics for specified columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    cols : list
        List of column names to analyze
        
    Returns:
    --------
    pandas.DataFrame with count, mean, std, min, quartiles, max, and missing values
    """
    stats_df = df[cols].describe().transpose()
    stats_df['missing'] = df[cols].isnull().sum().values
    return stats_df

print("=== DESCRIPTIVE STATISTICS ===")

print("\n1. Cognitive variables:")
cognitive_stats = summary_stats(df, cognitive)
print(cognitive_stats)

print("\n2. Frailty variables:")
frailty_stats = summary_stats(df, frailty)
print(frailty_stats)

print("\n3. Inflammatory biomarkers:")
inflammatory_stats = summary_stats(df, inflammatory)
print(inflammatory_stats)

# Combine key variables for correlation analysis
key_vars = cognitive + frailty + inflammatory
corr_matrix = df[key_vars].corr()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=0.5)
plt.title("Correlation Matrix of Key Clinical and Biomarker Variables")
plt.tight_layout()
plt.show()

# Separate data by group (Labels: 0 vs 1)
group_0 = df[df['Labels'] == 0]  # Typically control/preoperative group
group_1 = df[df['Labels'] == 1]  # Typically case/postoperative group

def compare_groups(var):
    """
    Perform independent t-test between two groups for a given variable.
    
    Parameters:
    -----------
    var : str
        Variable name to compare
        
    Returns:
    --------
    tuple: (t-statistic, p-value)
    """
    t_stat, p_val = stats.ttest_ind(group_0[var].dropna(), 
                                     group_1[var].dropna(),
                                     equal_var=False)  # Welch's t-test (unequal variances)
    return t_stat, p_val

print("\n=== GROUP COMPARISONS (Label 0 vs Label 1) ===")
print("Variable\t\t\t t-statistic\t p-value\t Significant (p < 0.05)")
print("-" * 70)

for var in key_vars:
    t, p = compare_groups(var)
    significance = '*' if p < 0.05 else ''
    print(f"{var:20}\t {t:8.3f}\t {p:8.4f}\t\t{significance}")

# Optional: Export results to CSV for reporting
def export_results():
    """Export statistical results to CSV files for documentation."""
    
    # Export descriptive statistics
    all_stats = pd.concat([cognitive_stats, frailty_stats, inflammatory_stats])
    all_stats.to_csv('descriptive_statistics.csv')
    
    # Export correlation matrix
    corr_matrix.to_csv('correlation_matrix.csv')
    
    # Export group comparison results
    comparison_results = []
    for var in key_vars:
        t, p = compare_groups(var)
        comparison_results.append({
            'Variable': var,
            't_statistic': t,
            'p_value': p,
            'significant': p < 0.05
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv('group_comparisons.csv', index=False)
    
    print("\nResults exported to CSV files:")
    print(" - descriptive_statistics.csv")
    print(" - correlation_matrix.csv")
    print(" - group_comparisons.csv")

# Uncomment to export results
# export_results()

# Optional: Generate summary figure
def create_summary_figure():
    """Create a multi-panel summary figure of key findings."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: MoCA distribution by group
    axes[0, 0].boxplot([group_0['MoCA'].dropna(), group_1['MoCA'].dropna()])
    axes[0, 0].set_xticklabels(['Group 0', 'Group 1'])
    axes[0, 0].set_ylabel('MoCA Score')
    axes[0, 0].set_title('Cognitive Performance by Group')
    
    # Panel 2: Inflammatory biomarkers
    biomarker_data = [group_0['IL6P'].dropna(), group_1['IL6P'].dropna(),
                      group_0['IL10P'].dropna(), group_1['IL10P'].dropna()]
    axes[0, 1].boxplot(biomarker_data)
    axes[0, 1].set_xticklabels(['IL6P G0', 'IL6P G1', 'IL10P G0', 'IL10P G1'], rotation=45)
    axes[0, 1].set_ylabel('Concentration (pg/mL)')
    axes[0, 1].set_title('Inflammatory Biomarkers by Group')
    
    # Panel 3: Correlation heatmap (simplified)
    sns.heatmap(corr_matrix.iloc[:6, :6], annot=True, cmap='coolwarm', 
                center=0, ax=axes[1, 0])
    axes[1, 0].set_title('Top Variable Correlations')
    
    # Panel 4: Significance bar plot
    p_values = [compare_groups(var)[1] for var in key_vars]
    axes[1, 1].bar(range(len(key_vars)), -np.log10(p_values))
    axes[1, 1].axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
    axes[1, 1].set_xticks(range(len(key_vars)))
    axes[1, 1].set_xticklabels(key_vars, rotation=45, ha='right')
    axes[1, 1].set_ylabel('-log10(p-value)')
    axes[1, 1].set_title('Group Comparison Significance')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('statistical_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Summary figure saved as 'statistical_summary.png'")

# Uncomment to generate summary figure
# create_summary_figure()

print("\n=== ANALYSIS COMPLETE ===")
print(f"Total samples: {len(df)}")
print(f"Preoperative (Label=0): {len(group_0)} samples")
print(f"Postoperative (Label=1): {len(group_1)} samples")
print(f"Total variables analyzed: {len(key_vars)}")