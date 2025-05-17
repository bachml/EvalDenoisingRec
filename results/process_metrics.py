import os
import glob
import pandas as pd
import argparse
import numpy as np
from pathlib import Path


def get_best_metrics(csv_file, loss_column='eval_loss'):
    """Get metrics corresponding to the epoch with the smallest eval_loss"""
    df = pd.read_csv(csv_file)
    
    # Find the row with the minimum value in the specified loss column
    min_loss_idx = df[loss_column].idxmin()
    best_metrics = df.loc[min_loss_idx]
    
    return best_metrics


def get_best_epoch_from_new_metric(csv_file, loss_column='eval_loss'):
    """Get the epoch with the smallest eval_loss from new_metric_records"""
    try:
        df = pd.read_csv(csv_file)
        # Find the row with the minimum value in the specified loss column
        min_loss_idx = df[loss_column].idxmin()
        best_epoch = df.loc[min_loss_idx, 'epoch']
        return best_epoch
    except Exception as e:
        print(f"Error reading file {csv_file}: {str(e)}")
        return None


def get_metrics_at_epoch(csv_file, target_epoch):
    """Get metrics for the specified epoch"""
    df = pd.read_csv(csv_file)
    
    # Find the row for the target epoch
    epoch_row = df[df['epoch'] == target_epoch]
    
    if epoch_row.empty:
        print(f"Epoch {target_epoch} not found in file {csv_file}")
        return None
    
    return epoch_row.iloc[0]


def process_seed_group(model, dataset, method, select_new_eval=False):
    """Process all seeds for the specified group"""
    # Set base directory to current working directory
    base_dir = os.getcwd()
    
    pattern = f"{model}-{dataset}-{method}"
    actual_method = method  # Original method name
    
    # For DeCA method, use epoch_loss instead of eval_loss
    # Check which columns actually exist in the CSV file, prefer eval_loss, use epoch_loss if not available
    loss_column = 'eval_loss'  # Default to eval_loss
    
    print(f"Processing pattern: {pattern}")
    
    # If using new_metric_records to select the best epoch, modify the method name in the results
    if select_new_eval:
        actual_method = f"{method}-new_select"
        print(f"Using new_metric_records to select the best epoch, result method name will be: {actual_method}")
    
    all_results = []
    seed_folders = []
    
    # Find metric_records folder
    metric_folder = os.path.join(base_dir, "metric_records", dataset)
    if not os.path.exists(metric_folder):
        print(f"Metric_records folder not found: {metric_folder}")
        return
    
    # If need to select the best epoch from new_metric_records, check if the folder exists
    if select_new_eval:
        new_metric_folder = os.path.join(base_dir, "new_metric_records", dataset)
        if not os.path.exists(new_metric_folder):
            print(f"New_metric_records folder not found: {new_metric_folder}")
            return
    
    # Find matching seed folders (e.g., LightGCN-ml100k-DeCA-2023)
    search_pattern = os.path.join(metric_folder, f"{model}-{dataset}-{method}-*")
    seed_folders = glob.glob(search_pattern)
    
    if not seed_folders:
        print(f"No folders matching {pattern} found")
        return
    
    print(f"Found {len(seed_folders)} folders matching {pattern}")
    
    # Process each seed folder
    for folder in seed_folders:
        folder_name = os.path.basename(folder)
        parts = folder_name.split('-')
        if len(parts) >= 4:  # Ensure there are enough parts to extract the seed
            seed = parts[-1]  # The last part is the seed
            print(f"Processing seed: {seed}")
        else:
            print(f"Cannot extract seed from folder name {folder_name}")
            continue
            
        # Find CSV files
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in folder {folder}")
            continue
        
        # If there are multiple CSV files, use the most recent one
        if len(csv_files) > 1:
            csv_files.sort(key=os.path.getmtime, reverse=True)
        
        csv_file = csv_files[0]
        print(f"Using CSV file: {csv_file}")
        
        # Check columns in the CSV file to determine which loss column to use
        try:
            temp_df = pd.read_csv(csv_file, nrows=1)
            if 'eval_loss' in temp_df.columns:
                loss_column = 'eval_loss'
            elif 'epoch_loss' in temp_df.columns:
                loss_column = 'epoch_loss'
            else:
                print(f"Warning: eval_loss or epoch_loss column not found in file {csv_file}")
                # Try to use other columns containing 'loss'
                loss_cols = [col for col in temp_df.columns if 'loss' in col.lower()]
                if loss_cols:
                    loss_column = loss_cols[0]
                    print(f"Using alternative loss column: {loss_column}")
                else:
                    print(f"Error: No loss column found in file {csv_file}, skipping")
                    continue
            
            print(f"For method {method}, using loss column: {loss_column}")
        except Exception as e:
            print(f"Error checking CSV file columns: {str(e)}")
            continue
        
        try:
            if select_new_eval:
                # Construct the corresponding folder and CSV path in new_metric_records
                new_folder_name = folder_name
                new_folder = os.path.join(base_dir, "new_metric_records", dataset, new_folder_name)
                
                if not os.path.exists(new_folder):
                    print(f"Corresponding folder not found in new_metric_records: {new_folder}")
                    continue
                
                new_csv_files = glob.glob(os.path.join(new_folder, "*.csv"))
                if not new_csv_files:
                    print(f"No CSV files found in folder {new_folder}")
                    continue
                
                # If there are multiple CSV files, use the most recent one
                if len(new_csv_files) > 1:
                    new_csv_files.sort(key=os.path.getmtime, reverse=True)
                
                new_csv_file = new_csv_files[0]
                print(f"Reading file from new_metric_records: {new_csv_file}")
                
                # Get the best epoch from new_metric_records
                best_epoch = get_best_epoch_from_new_metric(new_csv_file, loss_column)
                if best_epoch is None:
                    continue
                
                print(f"Using best epoch from new_metric_records: {best_epoch}")
                
                # Get metrics for the specified epoch from metric_records
                print(f"Reading file from metric_records: {csv_file}")
                metrics = get_metrics_at_epoch(csv_file, best_epoch)
                
                if metrics is None:
                    continue
                
                metrics['seed'] = seed
                all_results.append(metrics)
            else:
                # Use original logic
                print(f"Processing file: {csv_file}")
                best_metrics = get_best_metrics(csv_file, loss_column)
                best_metrics['seed'] = seed
                all_results.append(best_metrics)
                
        except Exception as e:
            print(f"Error processing file: {str(e)}")
    
    if not all_results:
        print("No valid results found")
        return
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Only keep the specified 4 metrics: R@5, R@20, N@5, N@20
    target_metrics = ['Recall@5', 'Recall@20', 'NDCG@5', 'NDCG@20']
    
    # Ensure target metrics exist
    available_metrics = [col for col in results_df.columns if col in target_metrics]
    if not available_metrics:
        print(f"Warning: None of the target metrics {target_metrics} found. Available columns: {list(results_df.columns)}")
        # Try to find alternative metrics
        recall_metrics = [col for col in results_df.columns if 'recall' in col.lower()]
        ndcg_metrics = [col for col in results_df.columns if 'ndcg' in col.lower()]
        available_metrics = recall_metrics + ndcg_metrics
        if available_metrics:
            print(f"Using alternative metrics: {available_metrics}")
        else:
            print("Error: Cannot find any usable metrics")
            return
    
    # Calculate mean and standard deviation
    metrics_columns = available_metrics
    avg_results = results_df[metrics_columns].mean()
    std_results = results_df[metrics_columns].std()
    
    # Create folder to save results
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results
    output_file = os.path.join(results_dir, f"{dataset}-{model}.csv")
    
    # Prepare the new row to write, keep only the mean with 4 decimal places
    mean_std_dict = {}
    for col in metrics_columns:
        mean_std_dict[col] = [f"{avg_results[col]:.4f}"]
    
    new_row = pd.DataFrame({'method': [actual_method], **mean_std_dict})
    
    # Check if the file exists
    if os.path.exists(output_file):
        # Read existing data
        existing_df = pd.read_csv(output_file)
        
        # Directly append the new row, without checking if it already exists
        existing_df = pd.concat([existing_df, new_row], ignore_index=True)
        
        # Save the updated file
        existing_df.to_csv(output_file, index=False)
        print(f"Appended results to {output_file}")
    else:
        # Create a new file
        new_row.to_csv(output_file, index=False)
        print(f"Saved results to {output_file}")
    
    # Return results, including mean and standard deviation (internal keeps complete information)
    result_with_std = {}
    for col in metrics_columns:
        result_with_std[col] = {
            'mean': avg_results[col],
            'std': std_results[col],
            'formatted': f"{avg_results[col]:.4f}"
        }
    
    return result_with_std


def main():
    parser = argparse.ArgumentParser(description='Process experiment metric data')
    
    # Add positional arguments (optional), for compatibility with old calling method
    parser.add_argument('pattern', nargs='?', help='Experiment group pattern, format: model-dataset-method')
    
    # Add named arguments, for compatibility with proc_all.sh script
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--dataset', help='Dataset name')
    parser.add_argument('--method', help='Method name')
    
    # Add new parameter to select the best epoch from new_metric_records
    parser.add_argument('--select_new_eval', action='store_true', 
                        help='If set, read the best epoch from new_metric_records and apply it in metric_records')
    
    args = parser.parse_args()
    
    # Parse arguments
    model = None
    dataset = None
    method = None
    
    # If pattern positional argument is provided
    if args.pattern:
        parts = args.pattern.split('-')
        if len(parts) != 3:
            print(f"Invalid pattern: {args.pattern}, should be in 'model-dataset-method' format")
            return
        model, dataset, method = parts
    # If named arguments are provided
    elif args.model and args.dataset:
        model = args.model
        dataset = args.dataset
        method = args.method
    else:
        print("Error: Please provide pattern parameter (model-dataset-method) or --model, --dataset, --method parameters")
        return
    
    print(f"Processing combination: model={model}, dataset={dataset}, method={method}")
    
    if args.select_new_eval:
        print("Will use new_metric_records to select the best epoch")
    
    results = process_seed_group(model, dataset, method, args.select_new_eval)
    if results is not None:
        print("\nAverage metric results:")
        for metric, data in results.items():
            print(f"{metric}: {data['formatted']}")


if __name__ == "__main__":
    main() 