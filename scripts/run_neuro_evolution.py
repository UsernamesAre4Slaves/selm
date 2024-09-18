"""
Script to trigger the neuro-evolutionary process within the SELM architecture.
This script is responsible for executing the neuro-evolutionary algorithm
and managing configuration, logging, and error handling.
"""

import os
import subprocess
import logging
import yaml

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
def load_config(config_path):
    """
    Loads the YAML configuration file for neuro-evolution.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    logging.info(f"Loaded configuration from {config_path}")
    return config

# Execute neuro-evolution process
def run_neuro_evolution(config):
    """
    Runs the neuro-evolution process by executing the neuro_evolution.py script.
    
    Args:
        config (dict): Configuration parameters for neuro-evolution.
    """
    command = f"python3 {config['neuro_evolution_script']}"
    
    # Add any additional command-line arguments from the config
    for arg, value in config.get('arguments', {}).items():
        command += f" --{arg} {value}"
    
    try:
        logging.info("Starting neuro-evolution process...")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logging.info("Neuro-evolution process completed successfully!")
        logging.info(f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Neuro-evolution process failed: {e}")
        logging.error(f"Error Output: {e.stderr}")
        raise

# Main function
def main():
    """
    Main function to load the configuration and trigger the neuro-evolutionary process.
    """
    config_path = 'config/neuro_evolution_config.yaml'
    
    # Load configuration
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        logging.error(f"Error loading configuration: {e}")
        return

    # Run neuro-evolution process
    try:
        run_neuro_evolution(config)
    except Exception as e:
        logging.error(f"An error occurred while running neuro-evolution: {e}")

if __name__ == "__main__":
    main()
