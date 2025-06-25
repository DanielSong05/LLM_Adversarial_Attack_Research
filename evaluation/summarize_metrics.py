import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

def load_metrics_file(file_path):
    """Load and parse metrics from a JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_metrics_info(filename):
    """Extract attack type and model name from filename"""
    base_name = os.path.basename(filename).replace('_metrics.json', '')
    parts = base_name.split('_')
    attack_type = parts[0]  # mhj, cosafe
    model_name = '_'.join(parts[1:])  # handle multi-part model names
    return attack_type, model_name

def create_table_image(df, output_path):
    """Create and save a table as a PNG image"""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = Table(ax, bbox=[0, 0, 1, 1])
    
    # Add cells
    nrows, ncols = len(df) + 1, len(df.columns)
    width, height = 1.0 / ncols, 1.0 / nrows
    
    # Find rows with highest success rate for each attack type
    max_asr_rows = df.groupby('Attack Type')['Success Rate (%)'].idxmax()
    
    # Add header
    for j, col in enumerate(df.columns):
        table.add_cell(0, j, width, height, text=col, loc='center', facecolor='#f0f0f0')
    
    # Add data
    for i, (idx, row) in enumerate(df.iterrows()):
        for j, val in enumerate(row):
            cell = table.add_cell(i + 1, j, width, height, text=str(val), loc='center')
            if idx in max_asr_rows.values:  # Make all text bold for highest success rate rows
                cell.get_text().set_weight('bold')
    
    ax.add_table(table)
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Define metrics directory and files directly
    metrics_dir = "../results/metrics"
    metrics_files = [
        os.path.join(metrics_dir, "mhj_gpt-4_metrics.json"),
        os.path.join(metrics_dir, "mhj_claude-sonnet_metrics.json"),
        os.path.join(metrics_dir, "mhj_llama-70b_metrics.json"),
        os.path.join(metrics_dir, "mhj_llama-8b_metrics.json"),
        os.path.join(metrics_dir, "cosafe_gpt-4_metrics.json"),
        os.path.join(metrics_dir, "cosafe_claude-sonnet_metrics.json"),
        os.path.join(metrics_dir, "cosafe_llama-70b_metrics.json"),
        os.path.join(metrics_dir, "cosafe_llama-8b_metrics.json"),
    ]
    
    # Initialize list to store data
    data = []
    
    # Process each metrics file
    for file_path in metrics_files:
        try:
            # Load metrics
            metrics = load_metrics_file(file_path)
            
            # Extract attack type and model name
            attack_type, model_name = extract_metrics_info(file_path)
            
            # Extract relevant metrics
            attack_success_rate = metrics.get('attack_success_rate', 0)
            adversarial_costs = metrics.get('adversarial_costs', {})
            avg_turns = adversarial_costs.get('avg_turns_to_success', 0)
            avg_tokens = adversarial_costs.get('avg_tokens_to_success', 0)
            
            # Add to data list
            data.append({
                'Attack Type': attack_type,
                'Model': model_name,
                'Success Rate (%)': round(attack_success_rate, 1),
                'Avg Turns to Success': round(avg_turns, 1),
                'Avg Tokens to Success': round(avg_tokens, 0)
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by Attack Type and Model
    df = df.sort_values(['Attack Type', 'Model'])
    
    # Save to PNG
    create_table_image(df, "../results/metrics_summary.png")

if __name__ == "__main__":
    main() 