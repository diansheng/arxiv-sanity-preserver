import os
import time
import pickle
import shutil
import random
import threading
import queue
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

from src.utils import Config, setup_logger

# Configuration
timeout_secs = 10  # after this many seconds we give up on a paper
max_workers = 20   # number of worker threads (can be higher than max_concurrent_downloads)
max_concurrent_downloads = 5  # maximum number of simultaneous downloads (QPS control)
delay_min = 0.05   # minimum delay between downloads
delay_max = 0.15   # maximum delay between downloads

# Create PDF directory if it doesn't exist
if not os.path.exists(Config.pdf_dir): os.makedirs(Config.pdf_dir)

# Get list of all PDFs we already have
have = set(os.listdir(Config.pdf_dir))

# Function to download a single PDF
def download_pdf(item, stats, db_size, download_semaphore, logger):
    pid, j = item
    
    # Extract PDF URL
    pdfs = [x['href'] for x in j['links'] if x['type'] == 'application/pdf']
    assert len(pdfs) == 1
    pdf_url = pdfs[0] + '.pdf'
    basename = pdf_url.split('/')[-1]
    fname = os.path.join(Config.pdf_dir, basename)
    
    # Increment total counter
    numtot = stats.increment_total()
    
    try:
        if basename not in have:
            # Acquire semaphore before downloading (limits concurrent downloads)
            with download_semaphore:
                logger.info('Fetching %s into %s', pdf_url, fname)
                req = urlopen(pdf_url, None, timeout_secs)
                with open(fname, 'wb') as fp:
                    shutil.copyfileobj(req, fp)
                # Add a small delay to avoid overwhelming the server
                time.sleep(delay_min + random.uniform(0, delay_max - delay_min))
        else:
            logger.info('%s exists, skipping', fname)
        
        # Increment success counter
        numok = stats.increment_success()
        logger.info('%d/%d of %d downloaded ok.', numok, numtot, db_size)
    except Exception as e:
        logger.error('Error downloading: %s', pdf_url)
        logger.error(str(e))
        logger.info('%d/%d of %d downloaded ok.', stats.numok, numtot, db_size)

# Stats tracking with thread safety
class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.numok = 0
        self.numtot = 0
        
    def increment_total(self):
        with self.lock:
            self.numtot += 1
            return self.numtot
            
    def increment_success(self):
        with self.lock:
            self.numok += 1
            return self.numok

# Main execution
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download PDFs for papers in the database')
    parser.add_argument('--timeout', type=int, default=10, 
                      help='Timeout in seconds for each download')
    parser.add_argument('--max-workers', type=int, default=20, 
                      help='Number of worker threads')
    parser.add_argument('--max-concurrent', type=int, default=5, 
                      help='Maximum number of concurrent downloads (QPS control)')
    parser.add_argument('--delay-min', type=float, default=0.05, 
                      help='Minimum delay between downloads')
    parser.add_argument('--delay-max', type=float, default=0.15, 
                      help='Maximum delay between downloads')
    parser.add_argument('--log-level', type=str, default='INFO', 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger('pdf_downloader', log_file='pdf_download.log')
    
    # Set log level based on argument
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)
        for handler in logger.handlers:
            handler.setLevel(numeric_level)
    
    # Configuration
    global timeout_secs, max_workers, max_concurrent_downloads, delay_min, delay_max
    timeout_secs = args.timeout
    max_workers = args.max_workers
    max_concurrent_downloads = args.max_concurrent
    delay_min = args.delay_min
    delay_max = args.delay_max
    
    # Create PDF directory if it doesn't exist
    if not os.path.exists(Config.pdf_dir): 
        os.makedirs(Config.pdf_dir)
        logger.info('Created directory %s', Config.pdf_dir)
    
    # Get list of all PDFs we already have
    global have
    have = set(os.listdir(Config.pdf_dir))
    logger.info('Found %d existing PDFs in %s', len(have), Config.pdf_dir)
    
    # Load database
    try:
        db = pickle.load(open(Config.db_path, 'rb'))
        db_size = len(db)
        logger.info('Loaded database with %d papers', db_size)
    except Exception as e:
        logger.error('Error loading database: %s', e, exc_info=True)
        logger.info('Exiting due to database error')
        return
    
    stats = Stats()
    
    # Create a semaphore to limit concurrent downloads
    download_semaphore = threading.Semaphore(max_concurrent_downloads)
    
    logger.info("Starting download of %d papers using %d threads", db_size, max_workers)
    logger.info("Maximum concurrent downloads limited to %d", max_concurrent_downloads)
    
    # Create a thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit download tasks
        futures = [executor.submit(download_pdf, item, stats, db_size, download_semaphore, logger) for item in db.items()]
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()
    
    logger.info('Final number of papers downloaded okay: %d/%d', stats.numok, db_size)

if __name__ == "__main__":
    main()

