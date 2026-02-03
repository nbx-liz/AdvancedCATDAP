import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_importance(results, top_k=20, figsize=(10, 6)):
    """
    Plots the variable importance (Delta Score).
    """
    if results is None or results.empty:
        print("No results to plot.")
        return

    df_plot = results.sort_values('Delta_Score', ascending=False).head(top_k)
    
    fig = plt.figure(figsize=figsize)
    sns.barplot(data=df_plot, x='Delta_Score', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title(f'Top {top_k} Variable Importance (Delta AIC)')
    plt.xlabel('Delta Score (Higher is better)')
    plt.ylabel('Feature')
    plt.tight_layout()
    return fig

def plot_interaction_heatmap(combo_results, top_k=20, figsize=(10, 8)):
    """
    Plots the interaction gain as a heatmap for the top pairs.
    Note: This is a sparse representation visualization.
    """
    if combo_results is None or combo_results.empty:
        print("No combination results to plot.")
        return

    df_plot = combo_results.head(top_k)
    
    # Create a pivot table for the heatmap
    features = list(set(df_plot['Feature_1']).union(set(df_plot['Feature_2'])))
    matrix = pd.DataFrame(index=features, columns=features, dtype=float)
    
    for _, row in df_plot.iterrows():
        f1, f2, gain = row['Feature_1'], row['Feature_2'], row['Gain']
        matrix.loc[f1, f2] = gain
        matrix.loc[f2, f1] = gain
        
    fig = plt.figure(figsize=figsize)
    sns.heatmap(matrix, cmap='coolwarm', annot=True, fmt=".2f", center=0)
    plt.title(f'Top Interaction Gains')
    plt.tight_layout()
    return fig
