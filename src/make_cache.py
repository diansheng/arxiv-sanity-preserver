"""
computes various cache things on top of db.py so that the server
(running from serve.py) can start up and serve faster when restarted.

this script should be run whenever db.p is updated, and 
creates db2.p, which can be read by the server.
"""

import os
import json
import time
import pickle
import argparse
import logging
import dateutil.parser

from sqlite3 import dbapi2 as sqlite3
from src.utils import safe_pickle_dump, Config, setup_logger

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compute cache for faster server startup')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger('make_cache', log_file='make_cache.log')
    
    # Set log level based on argument
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)
        for handler in logger.handlers:
            handler.setLevel(numeric_level)
    
    sqldb = sqlite3.connect(Config.database_path)
    sqldb.row_factory = sqlite3.Row # to return dicts rather than tuples
    
    CACHE = {}
    
    logger.info('Loading the paper database %s', Config.db_path)
    db = pickle.load(open(Config.db_path, 'rb'))
    
    logger.info('Loading tfidf_meta %s', Config.meta_path)
    meta = pickle.load(open(Config.meta_path, "rb"))
    vocab = meta['vocab']
    idf = meta['idf']
    
    logger.info('Decorating the database with additional information...')
    for pid,p in db.items():
        timestruct = dateutil.parser.parse(p['updated'])
        p['time_updated'] = int(timestruct.strftime("%s")) # store in struct for future convenience
        timestruct = dateutil.parser.parse(p['published'])
        p['time_published'] = int(timestruct.strftime("%s")) # store in struct for future convenience
    
    logger.info('Computing min/max time for all papers...')
    tts = [time.mktime(dateutil.parser.parse(p['updated']).timetuple()) for pid,p in db.items()]
    ttmin = min(tts)*1.0
    ttmax = max(tts)*1.0
    for pid,p in db.items():
        tt = time.mktime(dateutil.parser.parse(p['updated']).timetuple())
        p['tscore'] = (tt-ttmin)/(ttmax-ttmin)
    
    logger.info('Precomputing papers date sorted...')
    scores = [(p['time_updated'], pid) for pid,p in db.items()]
    scores.sort(reverse=True, key=lambda x: x[0])
    CACHE['date_sorted_pids'] = [sp[1] for sp in scores]
    
    # compute top papers in peoples' libraries
    logger.info('Computing top papers...')
    libs = sqldb.execute('''select * from library''').fetchall()
    counts = {}
    for lib in libs:
        pid = lib['paper_id']
        counts[pid] = counts.get(pid, 0) + 1
    top_paper_counts = sorted([(v,k) for k,v in counts.items() if v > 0], reverse=True)
    CACHE['top_sorted_pids'] = [q[1] for q in top_paper_counts]
    
    # some utilities for creating a search index for faster search
    punc = "'!\"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'" # removed hyphen from string.punctuation
    trans_table = {ord(c): None for c in punc}
    def makedict(s, forceidf=None, scale=1.0):
        words = set(s.lower().translate(trans_table).strip().split())
        idfd = {}
        for w in words: # todo: if we're using bigrams in vocab then this won't search over them
            if forceidf is None:
                if w in vocab:
                    # we have idf for this
                    idfval = idf[vocab[w]]*scale
                else:
                    idfval = 1.0*scale # assume idf 1.0 (low)
            else:
                idfval = forceidf
            idfd[w] = idfval
        return idfd
    
    def merge_dicts(dlist):
        m = {}
        for d in dlist:
            for k,v in d.items():
                m[k] = m.get(k,0) + v
        return m
    
    logger.info('Building an index for faster search...')
    search_dict = {}
    for pid,p in db.items():
        dict_title = makedict(p['title'], forceidf=5, scale=3)
        dict_authors = makedict(' '.join(x['name'] for x in p['authors']), forceidf=5)
        dict_categories = {x['term'].lower():5 for x in p['tags']}
        if 'and' in dict_authors: 
            # special case for "and" handling in authors list
            del dict_authors['and']
        dict_summary = makedict(p['summary'])
        search_dict[pid] = merge_dicts([dict_title, dict_authors, dict_categories, dict_summary])
    CACHE['search_dict'] = search_dict
    
    # save the cache
    logger.info('Writing %s', Config.serve_cache_path)
    safe_pickle_dump(CACHE, Config.serve_cache_path)
    logger.info('Writing %s', Config.db_serve_path)
    safe_pickle_dump(db, Config.db_serve_path)
    logger.info('Cache creation complete!')

if __name__ == "__main__":
    main()
