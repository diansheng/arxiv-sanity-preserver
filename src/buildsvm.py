# standard imports
import os
import sys
import pickle
import argparse
import logging
# non-standard imports
import numpy as np
from sklearn import svm
from sqlite3 import dbapi2 as sqlite3
# local imports
from src.utils import safe_pickle_dump, strip_version, Config, setup_logger

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build SVM models for user recommendations')
    parser.add_argument('--num-recommendations', type=int, default=1000,
                      help='Number of papers to recommend per user')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger('buildsvm', log_file='buildsvm.log')
    
    # Set log level based on argument
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)
        for handler in logger.handlers:
            handler.setLevel(numeric_level)
    
    num_recommendations = args.num_recommendations
    
    if not os.path.isfile(Config.database_path):
        logger.error("The database file %s should exist. You can create an empty database with sqlite3 as.db < schema.sql", Config.database_path)
        sys.exit()
    
    sqldb = sqlite3.connect(Config.database_path)
    sqldb.row_factory = sqlite3.Row # to return dicts rather than tuples
    
    def query_db(query, args=(), one=False):
        """Queries the database and returns a list of dictionaries."""
        cur = sqldb.execute(query, args)
        rv = cur.fetchall()
        return (rv[0] if rv else None) if one else rv
    
    # fetch all users
    users = query_db('''select * from user''')
    logger.info('Number of users: %d', len(users))
    
    # load the tfidf matrix and meta
    logger.info('Loading TF-IDF metadata from %s', Config.meta_path)
    meta = pickle.load(open(Config.meta_path, 'rb'))
    
    logger.info('Loading TF-IDF matrix from %s', Config.tfidf_path)
    out = pickle.load(open(Config.tfidf_path, 'rb'))
    X = out['X']
    X = X.todense().astype(np.float32)
    
    xtoi = { strip_version(x):i for x,i in meta['ptoi'].items() }
    
    user_sim = {}
    for ii, u in enumerate(users):
        logger.info("%d/%d Building an SVM for %s", ii, len(users), u['username'])
        uid = u['user_id']
        lib = query_db('''select * from library where user_id = ?''', [uid])
        pids = [x['paper_id'] for x in lib] # raw pids without version
        posix = [xtoi[p] for p in pids if p in xtoi]
        
        if not posix:
            logger.warning("Skipping user %s - empty library", u['username'])
            continue # empty library for this user maybe?
    
        logger.debug("User papers: %s", pids)
        y = np.zeros(X.shape[0])
        for ix in posix: y[ix] = 1
    
        clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
        clf.fit(X,y)
        s = clf.decision_function(X)
    
        sortix = np.argsort(-s)
        sortix = sortix[:min(num_recommendations, len(sortix))] # crop paper recommendations to save space
        user_sim[uid] = [strip_version(meta['pids'][ix]) for ix in list(sortix)]
    
    logger.info('Writing %s', Config.user_sim_path)
    safe_pickle_dump(user_sim, Config.user_sim_path)
    logger.info('SVM building complete!')

if __name__ == "__main__":
    main()
