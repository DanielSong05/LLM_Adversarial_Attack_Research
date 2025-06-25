import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_cumulative_success(metrics):
    max_turns = max(conv['total_turns'] for conv in metrics['conversation_metrics'])
    success_by_turn = {i: [] for i in range(1, max_turns + 1)}
    
    for conv in metrics['conversation_metrics']:
        if conv['success']:
            turn_success = conv['turns_to_success']
            for turn in range(1, max_turns + 1):
                if turn >= turn_success:
                    success_by_turn[turn].append(1)
                else:
                    success_by_turn[turn].append(0)
    
    cumulative_success = []
    total_convs = len(metrics['conversation_metrics'])
    for turn in range(1, max_turns + 1):
        success_rate = sum(success_by_turn[turn]) / total_convs * 100
        cumulative_success.append(success_rate)
    
    return cumulative_success

def plot_attack_success(attack_type):
    models = ['llama-70b', 'llama-8b', 'gpt-4', 'claude-sonnet']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    plt.figure(figsize=(10, 6))
    
    for model, color in zip(models, colors):
        file_path = f'../results/metrics/{attack_type}_{model}_metrics.json'
        metrics = load_metrics(file_path)
        success_rates = calculate_cumulative_success(metrics)
        turns = range(1, len(success_rates) + 1)
        plt.plot(turns, success_rates, marker='o', label=model, color=color)
    
    if attack_type == 'cosafe':
        attack_type = 'coreference'

    plt.xlabel('Number of Turns')
    plt.ylabel('Cumulative Attack Success Rate (%)')
    plt.title(f'Cumulative Attack Success Rate by Turn - {attack_type.upper()}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(range(1, max(turns) + 1))
    
    # Set different y-axis limits based on attack type
    if attack_type == 'coreference':
        plt.ylim(0, 20)
    else:  # mhj
        plt.ylim(0, 100)
    
    # Save plot
    plt.savefig(f'../results/plots/{attack_type}_success_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create plots directory if it doesn't exist
    Path('../results/plots').mkdir(parents=True, exist_ok=True)
    
    # Generate plots only for mhj and cosafe
    attack_types = ['mhj', 'cosafe']
    for attack_type in attack_types:
        plot_attack_success(attack_type)

if __name__ == '__main__':
    main() 