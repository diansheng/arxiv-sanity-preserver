"""
Script that iterates over all files in data/pdf/f.pdf and creates a file data/txt/f.pdf.txt
containing the raw text, extracted using the "pdftotext" command.
This version uses multiprocessing for parallel execution and tqdm for progress visualization.
"""

import os
import sys
import time
import shutil
import pickle
import multiprocessing as mp
import argparse
import logging
from tqdm import tqdm

from src.utils import Config, setup_logger

# Function to process a single PDF file
def process_pdf(args):
    i, f, total_files = args
    
    txt_basename = f + '.txt'
    pdf_path = os.path.join(Config.pdf_dir, f)
    txt_path = os.path.join(Config.txt_dir, txt_basename)
    
    # Skip if already processed
    if os.path.exists(txt_path):
        return f"Skipped {txt_basename}, already exists."
    
    # Convert PDF to text
    cmd = f"pdftotext \"{pdf_path}\" \"{txt_path}\""
    os.system(cmd)
    
    # Check if output was created
    if not os.path.isfile(txt_path):
        # There was an error with converting the pdf
        error_msg = f"Problem parsing {pdf_path} to text, creating empty file."
        # On Windows, 'touch' command doesn't exist, so create empty file this way
        with open(txt_path, 'w') as f:
            pass
        return error_msg
    
    return f"Processed {f}"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert PDF files to text using pdftotext')
    parser.add_argument('--log-level', type=str, default='INFO', 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger('pdf_converter', log_file='pdf_convert.log')
    
    # Set log level based on argument
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)
        for handler in logger.handlers:
            handler.setLevel(numeric_level)
    
    # Make sure pdftotext is installed
    if not shutil.which('pdftotext'):  # needs Python 3.3+
        logger.error('ERROR: you don\'t have pdftotext installed. Install it first before calling this script')
        sys.exit()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(Config.txt_dir):
        logger.info('Creating %s', Config.txt_dir)
        os.makedirs(Config.txt_dir)
    
    # Get list of PDF files to process
    files = os.listdir(Config.pdf_dir)
    total_files = len(files)
    logger.info("Found %d PDF files to process", total_files)
    
    # Prepare arguments for multiprocessing
    args_list = [(i, f, total_files) for i, f in enumerate(files)]
    
    # Determine number of processes to use (use 75% of available cores by default)
    num_processes = max(1, int(mp.cpu_count() * 0.75))
    logger.info("Using %d processes for parallel processing", num_processes)
    
    # Process files in parallel with progress bar
    with mp.Pool(processes=num_processes) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap(process_pdf, args_list),
            total=total_files,
            desc="Converting PDFs to text",
            unit="file"
        ))
    
    # Count successful conversions
    successful = sum(1 for result in results if not result.startswith("Skipped") and not "Problem" in result)
    skipped = sum(1 for result in results if result.startswith("Skipped"))
    failed = sum(1 for result in results if "Problem" in result)
    
    logger.info("\nProcessing complete!")
    logger.info("Successfully converted: %d files", successful)
    logger.info("Skipped (already exist): %d files", skipped)
    logger.info("Failed to convert: %d files", failed)

if __name__ == "__main__":
    main()

