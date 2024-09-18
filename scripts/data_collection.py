import os
import logging
from src.scraper.scraper import WebScraper  # Ensure this module is correctly implemented
import argparse

def setup_logging(log_file):
    """
    Set up logging configuration.

    Parameters:
    - log_file (str): Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info('Logging setup complete.')

def collect_data(urls, output_file):
    """
    Collect data from a list of URLs and save to a CSV file.

    Parameters:
    - urls (list of str): List of URLs to scrape.
    - output_file (str): Path to save the scraped data.
    """
    try:
        # Initialize the scraper
        scraper = WebScraper(urls, output_file)
        logging.info(f'Starting data collection from: {urls}')
        
        # Run the scraper
        scraper.run()
        
        logging.info(f'Data collection completed. Data saved to: {output_file}')
        
    except Exception as e:
        logging.error(f'An error occurred during data collection: {e}')
        raise

def main():
    """
    Main function to handle argument parsing and data collection.
    """
    parser = argparse.ArgumentParser(description='Collect data from specified URLs.')
    parser.add_argument(
        '--urls',
        type=str,
        nargs='+',
        required=True,
        help='List of URLs to scrape.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to the file where scraped data will be saved.'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default='data_collection.log',
        help='Path to the log file.'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    
    # Collect data
    collect_data(args.urls, args.output_file)

if __name__ == '__main__':
    main()
