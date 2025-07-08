"""
Queries arxiv API and downloads papers (the query is a parameter).
The script is intended to enrich an existing database pickle (by default db.p),
so this file will be loaded first, and then new results will be added to it.
"""

import os
import time
import pickle
import random
import argparse
import urllib.request
import feedparser
import logging
import copy

from src.utils import Config, safe_pickle_dump, setup_logger

# Remove the setup_logger function since it's now in utils.py

def encode_feedparser_dict(d):
    """ 
    helper function to get rid of feedparser bs with a deep copy. 
    I hate when libs wrap simple things in their own classes.
    """
    # Use Python's built-in copy module for safer deep copying
    if isinstance(d, feedparser.FeedParserDict) or isinstance(d, dict):
        j = {}
        for k in d.keys():
            j[k] = encode_feedparser_dict(d[k])
        return j
    elif isinstance(d, list):
        l = []
        for k in d:
            l.append(encode_feedparser_dict(k))
        return l
    elif isinstance(d, (str, int, float, bool, type(None))):
        # Explicitly handle primitive types
        return d
    else:
        # For other types, convert to string to avoid potential issues
        return str(d)

def parse_arxiv_url(url):
    """ 
    examples is http://arxiv.org/abs/1512.08756v2
    we want to extract the raw id and the version
    """
    ix = url.rfind('/')
    idversion = url[ix+1:] # extract just the id (and the version)
    parts = idversion.split('v')
    assert len(parts) == 2, 'error parsing url ' + url
    return parts[0], int(parts[1])

# At the end of the script, add this before exiting:
if __name__ == "__main__":
    # Set up logging
    logger = setup_logger('arxiv_fetcher', log_file='arxiv_fetch.log')
    
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--search-query', type=str,
                      default='(cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL+OR+cat:cs.NE+OR+cat:stat.ML)+AND+agent',
                      help='query used for arxiv API. See http://arxiv.org/help/api/user-manual#detailed_examples')
    # parser.add_argument('--search-query', type=str,
    #                   default='cat:cs.CV+OR+cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL+OR+cat:cs.NE+OR+cat:stat.ML',
    #                   help='query used for arxiv API. See http://arxiv.org/help/api/user-manual#detailed_examples')
    parser.add_argument('--start-index', type=int, default=0, help='0 = most recent API result')
    parser.add_argument('--max-index', type=int, default=10000, help='upper bound on paper index we will fetch')
    parser.add_argument('--results-per-iteration', type=int, default=100, help='passed to arxiv API')
    parser.add_argument('--wait-time', type=float, default=5.0, help='lets be gentle to arxiv API (in number of seconds)')
    parser.add_argument('--break-on-no-added', type=int, default=1, help='break out early if all returned query papers are already in db? 1=yes, 0=no')
    parser.add_argument('--log-level', type=str, default='INFO', 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level')
    args = parser.parse_args()
    
    # Set log level based on argument
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)
        for handler in logger.handlers:
            handler.setLevel(numeric_level)

    # misc hardcoded variables
    base_url = 'http://export.arxiv.org/api/query?' # base api query url
    logger.info('Searching arXiv for %s', args.search_query)

    # lets load the existing database to memory
    try:
        db = pickle.load(open(Config.db_path, 'rb'))
    except Exception as e:
        logger.error('Error loading existing database: %s', e, exc_info=True)
        logger.info('Starting from an empty database')
        db = {}

    # -----------------------------------------------------------------------------
    # main loop where we fetch the new results
    logger.info('Database has %d entries at start', len(db))
    num_added_total = 0
    for i in range(args.start_index, args.max_index, args.results_per_iteration):

        logger.info("Results %i - %i", i, i+args.results_per_iteration)
        query = 'search_query=%s&sortBy=lastUpdatedDate&start=%i&max_results=%i' % (args.search_query,
                                                         i, args.results_per_iteration)
        try:
            with urllib.request.urlopen(base_url+query) as url:
                response = url.read()
            parse = feedparser.parse(response)
            num_added = 0
            num_skipped = 0


            logger.info(f"Received {len(parse.entries)} results for query {query}")

            for e in parse.entries:
                try:
                    # Create a safe copy of the entry
                    j = encode_feedparser_dict(e)

                    # extract just the raw arxiv id and version for this paper
                    rawid, version = parse_arxiv_url(j['id'])
                    j['_rawid'] = rawid
                    j['_version'] = version

                    # add to our database if we didn't have it before, or if this is a new version
                    if not rawid in db or j['_version'] > db[rawid]['_version']:
                        # Make sure all string values are properly decoded
                        if 'updated' in j and isinstance(j['updated'], bytes):
                            j['updated'] = j['updated'].decode('utf-8')
                        if 'title' in j and isinstance(j['title'], bytes):
                            j['title'] = j['title'].decode('utf-8')
                            
                        db[rawid] = j
                        # logger.debug('Updated %s added %s', j['updated'], j['title'])
                        num_added += 1
                        num_added_total += 1
                    else:
                        num_skipped += 1
                except Exception as e:
                    logger.error('Error processing entry: %s', e, exc_info=True)

            # log some information
            logger.info(f'Added {num_added} papers, already had {num_added_total}.')

            if len(parse.entries) == 0:
                logger.warning('Received no results from arxiv. Rate limiting? Exiting. Restart later maybe.')
                logger.debug('Response: %s', response)
                break

            if num_added == 0 and args.break_on_no_added == 1:
                logger.info('No new papers were added. Assuming no new papers exist. Exiting.')
                break

            logger.info('Sleeping for %i seconds', args.wait_time)
            time.sleep(args.wait_time + random.uniform(0, 3))
            
        except Exception as e:
            logger.error('Error in fetch iteration %d-%d: %s', i, i+args.results_per_iteration, e, exc_info=True)

    # save the database before we quit, if we found anything new
    if num_added_total > 0:
        logger.info('Saving database with %d papers to %s', len(db), Config.db_path)
        try:
            # Create tmp directory if it doesn't exist
            os.makedirs(Config.tmp_dir, exist_ok=True)
            
            # Make sure we're not saving any problematic objects
            clean_db = {}
            for key, value in db.items():
                # Ensure all values are JSON serializable
                if isinstance(key, str) and isinstance(value, dict):
                    clean_db[key] = value
                    
            safe_pickle_dump(clean_db, Config.db_path)
        except Exception as e:
            logger.error('Failed to save database: %s', e, exc_info=True)
    else:
        logger.info('No new papers added. Database not updated.')
    