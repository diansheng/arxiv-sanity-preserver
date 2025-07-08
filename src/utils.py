from contextlib import contextmanager

import os
import re
import pickle
import tempfile
import logging

# global settings
# -----------------------------------------------------------------------------
class Config(object):
    # Folder structure
    src_dir = 'src'
    log_dir = 'log'
    output_dir = 'output'
    
    # main paper information repo file
    db_path = os.path.join(output_dir, 'db.p')
    # intermediate processing folders
    pdf_dir = os.path.join('data', 'pdf')
    txt_dir = os.path.join('data', 'txt')
    thumbs_dir = os.path.join('static', 'thumbs')
    # intermediate pickles
    tfidf_path = os.path.join(output_dir, 'tfidf.p')
    meta_path = os.path.join(output_dir, 'tfidf_meta.p')
    sim_path = os.path.join(output_dir, 'sim_dict.p')
    user_sim_path = os.path.join(output_dir, 'user_sim.p')
    # sql database file
    db_serve_path = os.path.join(output_dir, 'db2.p') # an enriched db.p with various preprocessing info
    database_path = os.path.join(output_dir, 'as.db')
    serve_cache_path = os.path.join(output_dir, 'serve_cache.p')
    
    beg_for_hosting_money = 1 # do we beg the active users randomly for money? 0 = no.
    banned_path = os.path.join(output_dir, 'banned.txt') # for twitter users who are banned
    tmp_dir = os.path.join(output_dir, 'tmp')

# Context managers for atomic writes courtesy of
# http://stackoverflow.com/questions/2333872/atomic-writing-to-file-with-python
@contextmanager
def _tempfile(*args, **kws):
    """ Context for temporary file.

    Will find a free temporary filename upon entering
    and will try to delete the file on leaving

    Parameters
    ----------
    suffix : string
        optional file suffix
    """

    fd, name = tempfile.mkstemp(*args, **kws)
    os.close(fd)
    try:
        yield name
    finally:
        try:
            os.remove(name)
        except OSError as e:
            if e.errno == 2:
                pass
            else:
                raise e


@contextmanager
def open_atomic(filepath, *args, **kwargs):
    """ Open temporary file object that atomically moves to destination upon
    exiting.

    Allows reading and writing to and from the same filename.

    Parameters
    ----------
    filepath : string
        the file path to be opened
    fsync : bool
        whether to force write the file to disk
    kwargs : mixed
        Any valid keyword arguments for :code:`open`
    """
    fsync = kwargs.pop('fsync', False)

    with _tempfile(dir=os.path.dirname(filepath)) as tmppath:
        with open(tmppath, *args, **kwargs) as f:
            yield f
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        # Handle Windows case where os.rename cannot replace existing files
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(tmppath, filepath)

def safe_pickle_dump(obj, fname):
    with open_atomic(fname, 'wb') as f:
        pickle.dump(obj, f, -1)


# arxiv utils
# -----------------------------------------------------------------------------

def strip_version(idstr):
    """ identity function if arxiv id has no version, otherwise strips it. """
    parts = idstr.split('v')
    return parts[0]

# "1511.08198v1" is an example of a valid arxiv id that we accept
def isvalidid(pid):
  return re.match(r'^\d+\.\d+(v\d+)?$', pid)


# Logger configuration
def setup_logger(name, log_level=logging.INFO, log_file=None):
    """Set up and configure logger"""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    # Create formatter with line numbers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    
    # Add formatter to ch
    ch.setFormatter(formatter)
    
    # Add ch to logger if not already added
    if not logger.handlers:
        logger.addHandler(ch)
        
        # Create file handler for persistent logs if specified
        if log_file:
            # Use the shared log directory
            if log_file == 'default':
                log_file = os.path.join(Config.log_dir, 'arxiv-sanity.log')
            elif not os.path.isabs(log_file):
                # If it's not an absolute path, put it in the log directory
                log_file = os.path.join(Config.log_dir, log_file)
                
            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            fh = logging.FileHandler(log_file)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    
    return logger

# Add a convenience function to get a logger with the shared log file
def get_shared_logger(name, log_level=logging.INFO):
    """Get a logger that writes to the shared log file"""
    return setup_logger(name, log_level=log_level, log_file='default')
