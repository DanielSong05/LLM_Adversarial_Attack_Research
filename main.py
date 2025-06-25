#!/usr/bin/env python3
"""
Main orchestration script for LLM Robustness Evaluation Pipeline

This script runs the complete pipeline in the following order:
1. Preprocess data files (CoSafe and MHJ)
2. Run attack scripts against different models
3. Calculate metrics for each model's results
4. Summarize metrics across all models
5. Generate attack success plots

Usage:
    python main.py [--models MODEL1 MODEL2 ...] [--max_conversations N] [--skip_preprocess]
"""

import os
import sys
import subprocess
import argparse
import asyncio
from pathlib import Path
from typing import List, Optional

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_command(cmd: List[str], description: str, cwd: Optional[str] = None) -> bool:
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or os.getcwd(),
            check=True,
            capture_output=False,
            text=True
        )
        print(f"SUCCESS: {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {description} failed with exit code {e.returncode}")
        return False

def run_preprocessing():
    """Run preprocessing scripts for both CoSafe and MHJ datasets"""
    print("\nStarting preprocessing phase...")
    
    # Run CoSafe preprocessing
    cosafe_success = run_command(
        ["python", "preprocess/preprocess_cosafe.py"],
        "CoSafe data preprocessing"
    )
    
    # Run MHJ preprocessing
    mhj_success = run_command(
        ["python", "preprocess/preprocess_mhj.py"],
        "MHJ data preprocessing"
    )
    
    return cosafe_success and mhj_success

async def run_attacks(models: List[str], max_conversations: int):
    """Run attack scripts for different attack types and models"""
    print("\nStarting attack phase...")
    
    attack_types = ["cosafe", "mhj"]
    
    for attack_type in attack_types:
        print(f"\nRunning {attack_type.upper()} attacks...")
        
        # Import and run the attack script
        from attacks.attack import run_attack
        try:
            await run_attack(
                attack_type=attack_type,
                models=models,
                max_conversations=max_conversations
            )
            print(f"SUCCESS: {attack_type.upper()} attacks completed successfully")
        except Exception as e:
            print(f"FAILED: {attack_type.upper()} attacks failed: {str(e)}")
            return False
    
    return True

def run_metrics_calculation():
    """Calculate metrics for all result files"""
    print("\nStarting metrics calculation phase...")
    
    # Define expected result files
    models = ["gpt-4", "claude-sonnet", "llama-70b", "llama-8b"]
    attack_types = ["cosafe", "mhj"]
    
    results_dir = Path("results/raw")
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return False
    
    success_count = 0
    total_count = 0
    
    for attack_type in attack_types:
        for model in models:
            result_file = f"{attack_type}_{model}_results.json"
            result_path = results_dir / result_file
            
            if result_path.exists():
                total_count += 1
                success = run_command(
                    ["python", "evaluation/metrics.py", "--input_file", result_file],
                    f"Metrics calculation for {attack_type}_{model}"
                )
                if success:
                    success_count += 1
            else:
                print(f"Result file not found: {result_path}")
    
    print(f"\nMetrics calculation completed: {success_count}/{total_count} successful")
    return success_count > 0

def run_summarization():
    """Run metrics summarization and plotting"""
    print("\nStarting summarization and plotting phase...")
    
    # Create plots directory
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Run metrics summarization
    summary_success = run_command(
        ["python", "evaluation/summarize_metrics.py"],
        "Metrics summarization"
    )
    
    # Run attack success plotting
    plotting_success = run_command(
        ["python", "evaluation/plot_attack_success.py"],
        "Attack success plotting"
    )
    
    return summary_success and plotting_success

def check_prerequisites():
    """Check if required directories and files exist"""
    print("\nChecking prerequisites...")
    
    required_dirs = [
        "data",
        "preprocess",
        "attacks", 
        "evaluation",
        "models"
    ]
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"Required directory not found: {dir_name}")
            return False
    
    # Check for .env file (required for API keys)
    if not os.path.exists(".env"):
        print(".env file not found. Make sure you have set up your API keys.")
    
    print("Prerequisites check completed")
    return True

def print_summary():
    """Print a summary of the pipeline execution"""
    print("\n" + "="*60)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*60)
    
    # Check for output files
    output_files = [
        "results/metrics_summary.png",
        "results/plots/mhj_success_rate.png",
        "results/plots/cosafe_success_rate.png"
    ]
    
    print("\nGenerated Output Files:")
    for file_path in output_files:
        if os.path.exists(file_path):
            print(f"  SUCCESS: {file_path}")
        else:
            print(f"  MISSING: {file_path}")
    
    print("\nResults Directory Structure:")
    results_dir = Path("results")
    if results_dir.exists():
        for item in results_dir.rglob("*"):
            if item.is_file():
                print(f"  {item.relative_to(results_dir)}")
    
    print("\nPipeline execution completed!")
    print("Check the results/ directory for all generated files and visualizations.")

async def main():
    """Main orchestration function"""
    parser = argparse.ArgumentParser(
        description="LLM Robustness Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run with default settings
  python main.py --models gpt-4 claude-sonnet      # Run only specific models
  python main.py --max_conversations 25            # Limit conversations per attack
  python main.py --skip_preprocess                 # Skip preprocessing phase
        """
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4", "claude-sonnet", "llama-70b", "llama-8b"],
        help="Models to test (default: all supported models)"
    )
    
    parser.add_argument(
        "--max_conversations",
        type=int,
        default=50,
        help="Maximum number of conversations per attack type (default: 50)"
    )
    
    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="Skip the preprocessing phase"
    )
    
    args = parser.parse_args()
    
    print("LLM Robustness Evaluation Pipeline")
    print("="*50)
    print(f"Models: {', '.join(args.models)}")
    print(f"Max conversations per attack: {args.max_conversations}")
    print(f"Skip preprocessing: {args.skip_preprocess}")
    
    # Check prerequisites
    if not check_prerequisites():
        print("Prerequisites check failed. Please fix the issues and try again.")
        return 1
    
    # Phase 1: Preprocessing
    if not args.skip_preprocess:
        if not run_preprocessing():
            print("Preprocessing phase failed. Stopping pipeline.")
            return 1
    else:
        print("\nSkipping preprocessing phase as requested")
    
    # Phase 2: Attacks
    if not await run_attacks(args.models, args.max_conversations):
        print("Attack phase failed. Stopping pipeline.")
        return 1
    
    # Phase 3: Metrics Calculation
    if not run_metrics_calculation():
        print("Metrics calculation phase failed. Stopping pipeline.")
        return 1
    
    # Phase 4: Summarization and Plotting
    if not run_summarization():
        print("Summarization phase failed. Stopping pipeline.")
        return 1
    
    # Print summary
    print_summary()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        sys.exit(1) 