import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import glob
import argparse

# Set Matplotlib style
plt.style.use('default')

def read_experiment_data(model, dataset, method, seed):
    """Read CSV data for a specific experiment"""
    folder_pattern = f"{model}-{dataset}-{method}-{seed}"
    
    # Normalize dataset name
    normalized_dataset = dataset
    if 'ml-100k' in dataset:
        normalized_dataset = 'ml100k'
    elif 'ml100k' in dataset:
        normalized_dataset = 'ml100k'
        
    # Try different path combinations
    possible_paths = [
        os.path.join("metric_records", normalized_dataset, folder_pattern),
        os.path.join("metric_records", dataset, folder_pattern)
    ]
    
    # If it contains ml100k, try two possible paths
    if 'ml100k' in dataset or 'ml-100k' in dataset:
        alt_dataset = 'ml-100k' if 'ml100k' in dataset else 'ml100k'
        alt_pattern = f"{model}-{alt_dataset}-{method}-{seed}"
        possible_paths.extend([
            os.path.join("metric_records", normalized_dataset, alt_pattern),
            os.path.join("metric_records", alt_dataset, alt_pattern)
        ])
    
    # Try all possible paths
    for folder_path in possible_paths:
        if os.path.exists(folder_path):
            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
            if csv_files:
                try:
                    df = pd.read_csv(csv_files[0])
                    return df
                except Exception as e:
                    print(f"Error reading file {csv_files[0]}: {e}")
                    continue
    
    print(f"Could not find valid data folder: {folder_pattern}")
    return None

def find_min_loss_epochs(model, dataset, method, seeds):
    """Find the epoch with minimum loss for each seed, handle different loss column names, and calculate average"""
    min_loss_epochs = []
    for seed in seeds:
        df = read_experiment_data(model, dataset, method, seed)
        if df is not None:
            # Check whether to use epoch_loss or eval_loss
            loss_column = None
            if 'epoch_loss' in df.columns:
                loss_column = 'epoch_loss'
            elif 'eval_loss' in df.columns:
                loss_column = 'eval_loss'
            
            if loss_column:
                min_loss_idx = df[loss_column].idxmin()
                min_loss_epochs.append(df.loc[min_loss_idx, 'epoch'])
    
    # If multiple epochs were collected, calculate average
    if min_loss_epochs:
        avg_epoch = int(round(np.mean(min_loss_epochs)))
        print(f"{model}-{dataset}-{method} average epoch for minimum loss: {avg_epoch}, original values: {min_loss_epochs}")
        return avg_epoch
    return None

def aggregate_metrics(model, dataset, method, seeds, metric='NDCG@20'):
    """Aggregate results from multiple seeds, calculate mean and variance"""
    dfs = []
    for seed in seeds:
        df = read_experiment_data(model, dataset, method, seed)
        if df is not None and metric in df.columns:
            dfs.append(df)
    
    if not dfs:
        print(f"No valid data found for {model}-{dataset}-{method} with metric {metric}")
        return None, None
    
    min_epochs = min([len(df) for df in dfs])
    dfs = [df.iloc[:min_epochs] for df in dfs]
    
    all_data = np.stack([df.values for df in dfs], axis=0)
    mean_data = np.mean(all_data, axis=0)
    std_data = np.std(all_data, axis=0)
    
    mean_df = pd.DataFrame(mean_data, columns=dfs[0].columns)
    std_df = pd.DataFrame(std_data, columns=dfs[0].columns)
    
    return mean_df, std_df

def get_dataset_display_name(dataset):
    """Get display name for dataset"""
    mapping = {
        'ml100k': 'MovieLens-100K',
        'ml-100k': 'MovieLens-100K',
        'adressa': 'Adressa',
        'yelp': 'Yelp'
    }
    return mapping.get(dataset, dataset)

def get_method_display_name(method):
    """Get display name for method"""
    mapping = {
        'RCE': 'RCE',
        'TCE': 'TCE',
        'DCF': 'DCF',
        'UDT': 'UDT',
        'DeCA': 'DeCA',
        'DeCAp': 'DeCAp',
        'ERM': 'ERM',
        'ERMrevised': 'ERMrevised'
    }
    return mapping.get(method, method)

def find_nearest_epoch(epochs, target_epoch):
    """Find the closest epoch value to target epoch"""
    return epochs[np.abs(epochs - target_epoch).argmin()]

def plot_combined_metrics(models, datasets, methods, seeds, metric='NDCG@20', figsize=(20, 16), fontsize=14):
    """Plot combined charts"""
    # Create a 4x3 subplot layout and adjust bottom space to accommodate legend
    fig, axes = plt.subplots(4, 3, figsize=figsize)
    
    # Adjust vertical spacing between subplots
    plt.subplots_adjust(
        bottom=0.25,  # Bottom space
        hspace=14.5,   # Increase vertical spacing to a larger value, equivalent to vspace{15em}
        wspace=0.3    # Horizontal spacing
    )
    
    # Color settings - fixed colors for each method, ERM and ERMrevised use the same color
    method_colors = {
        'RCE': '#ff7f0e',     # RCE
        'TCE': '#2ca02c',     # TCE
        'DCF': '#d62728',     # DCF
        'UDT': '#9467bd',     # UDT
        'DeCA': '#8c564b',    # DeCA
        'DeCAp': '#e377c2',   # DeCAp - changed to a different color
        'ERM': '#17becf',      # ERM
        'ERMrevised': '#17becf'   # ERMrevised - same color as ERM
    }
    
    # For collecting all available methods
    all_available_methods = set()
    
    # Iterate through each model and dataset combination
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            
            # Get available methods for this combination
            available_methods = []
            for method in methods:
                mean_df, std_df = aggregate_metrics(model, dataset, method, seeds[dataset], metric)
                if mean_df is not None and metric in mean_df.columns:
                    available_methods.append(method)
                    all_available_methods.add(method)
            
            if not available_methods:
                ax.text(0.5, 0.5, 'No Data Available', 
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax.transAxes,
                       fontsize=fontsize)
                continue
            
            # Plot results for each method
            for method in available_methods:
                mean_df, std_df = aggregate_metrics(model, dataset, method, seeds[dataset], metric)
                if mean_df is None or metric not in mean_df.columns:
                    continue
                
                epochs = mean_df['epoch'].values
                metric_values = mean_df[metric].values
                std_values = std_df[metric].values
                
                method_display = get_method_display_name(method)
                
                # Use fixed colors
                color = method_colors.get(method, '#1f77b4')  # Default color
                
                # Plot mean line and standard deviation area
                line = ax.plot(epochs, metric_values,
                       label=method_display,
                       color=color,
                       linewidth=2)[0]
                
                ax.fill_between(epochs,
                              metric_values - std_values,
                              metric_values + std_values,
                              color=color,
                              alpha=0.15)
                
                # Find average epoch with minimum loss
                avg_min_loss_epoch = find_min_loss_epochs(model, dataset, method, seeds[dataset])
                
                # Ensure each line has a marker point
                if avg_min_loss_epoch is not None:
                    # If average epoch is not in data range, find closest epoch
                    if avg_min_loss_epoch not in epochs:
                        closest_epoch = find_nearest_epoch(epochs, avg_min_loss_epoch)
                        print(f"{model}-{dataset}-{method} average epoch {avg_min_loss_epoch} not in range, using closest {closest_epoch}")
                        avg_min_loss_epoch = closest_epoch
                    
                    idx = np.where(epochs == avg_min_loss_epoch)[0][0]
                    
                    # ERMrevised uses triangle marker, other methods use circle marker
                    if method == 'ERMrevised':
                        ax.plot(avg_min_loss_epoch, metric_values[idx], '^',  # Triangle marker
                              color=color, markersize=9, 
                              markerfacecolor='white',
                              markeredgewidth=2)
                    else:
                        ax.plot(avg_min_loss_epoch, metric_values[idx], 'o',  # Circle marker
                              color=color, markersize=8, 
                              markerfacecolor='white',
                              markeredgewidth=2)
                else:
                    # If no minimum loss epoch found, use middle position
                    mid_idx = len(epochs) // 2
                    print(f"{model}-{dataset}-{method} minimum loss epoch not found, using middle position {epochs[mid_idx]}")
                    
                    # ERMrevised uses triangle marker, other methods use circle marker
                    if method == 'ERMrevised':
                        ax.plot(epochs[mid_idx], metric_values[mid_idx], '^',  # Triangle marker
                              color=color, markersize=9, 
                              markerfacecolor='white',
                              markeredgewidth=2)
                    else:
                        ax.plot(epochs[mid_idx], metric_values[mid_idx], 'o',  # Circle marker
                              color=color, markersize=8, 
                              markerfacecolor='white',
                              markeredgewidth=2)
            
            # Set subplot properties
            ax.set_xlabel('Epoch', fontsize=fontsize-2)
            ax.set_ylabel(metric, fontsize=fontsize-2)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
            
            # Set title at the bottom, add numbering
            dataset_display = get_dataset_display_name(dataset)
            subplot_num = i * len(datasets) + j
            subplot_letter = chr(ord('a') + subplot_num)
            ax.set_title(f'({subplot_letter}) {model} on {dataset_display}', 
                        fontsize=fontsize,
                        fontweight='bold',
                        pad=20,  # Increase distance between title and chart
                        y=-0.3)  # Move title below the chart, increase distance
            
            # Remove top and right borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    # Create legend and place at the bottom
    # Set method legend in specified order: ERM, RCE, TCE, DeCA, DeCAp, DCF, UDT, ERMrevised
    ordered_methods = ['ERM', 'RCE', 'TCE', 'DeCA', 'DeCAp', 'DCF', 'UDT', 'ERMrevised']
    all_handles = []
    all_labels = []
    
    # First add ERM
    if 'ERM' in all_available_methods:
        color = method_colors.get('ERM', '#17becf')
        erm_line = plt.Line2D([0], [0], color=color, linewidth=2, marker='o',
                            markerfacecolor='white', markersize=8)
        all_handles.append(erm_line)
        all_labels.append('ERM')
    
    # Add middle methods (RCE, TCE, DeCA, DeCAp, DCF, UDT)
    for method in ['RCE', 'TCE', 'DeCA', 'DeCAp', 'DCF', 'UDT']:
        if method in all_available_methods:
            color = method_colors.get(method, '#1f77b4')
            line = plt.Line2D([0], [0], color=color, linewidth=2, marker='o',
                            markerfacecolor='white', markersize=8)
            all_handles.append(line)
            all_labels.append(get_method_display_name(method))
    
    # Finally add ERMrevised
    if 'ERMrevised' in all_available_methods:
        color = method_colors.get('ERMrevised', '#17becf')
        ermrevised_line = plt.Line2D([0], [0], color=color, linewidth=2, marker='^',
                                markerfacecolor='white', markersize=9)
        all_handles.append(ermrevised_line)
        all_labels.append('ERMrevised')
    
    if all_handles:  # Only create legend when there is data
        # Create legend, set to horizontal layout
        legend = fig.legend(all_handles, all_labels, 
                          loc='upper center',  # Position in center
                          bbox_to_anchor=(0.5, -0.05),  # Adjust to bottom of chart, increase distance
                          ncol=len(all_handles),  # Arrange all items horizontally
                          fontsize=fontsize+6,  # Increase legend font size
                          frameon=True,
                          fancybox=True,
                          framealpha=0.9)
        
        # Adjust legend style
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('gray')
    
    # Adjust layout to ensure legend is not clipped
    plt.tight_layout()
    
    # Create save directory
    os.makedirs("figures", exist_ok=True)
    
    # Save chart, ensure legend is not clipped
    save_path = os.path.join("figures", "main.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    print(f"Chart saved to: {save_path}")
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot performance metrics across models and datasets')
    parser.add_argument('--metric', type=str, default='NDCG@20', 
                        help='Metric to plot (e.g., NDCG@20, Recall@50)')
    args = parser.parse_args()
    
    # Set fixed models and datasets
    models = ['GMF', 'NeuMF', 'CDAE', 'LightGCN']
    datasets = ['ml100k', 'adressa', 'yelp']
    # Add ERMrevised method
    methods = ['RCE', 'TCE', 'DCF', 'UDT', 'DeCA', 'DeCAp', 'ERM', 'ERMrevised']
    
    # Set seeds
    default_seeds = ["2023", "2024", "2025"]
    yelp_seeds = ["2024", "2025"]
    
    # Choose appropriate seeds for each dataset
    seeds = {
        'ml100k': default_seeds,
        'adressa': default_seeds,
        'yelp': yelp_seeds
    }
    
    # Plot combined chart with selected metric
    plot_combined_metrics(models, datasets, methods, seeds, metric=args.metric)

if __name__ == "__main__":
    main() 