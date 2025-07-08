#!/usr/bin/env python

import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def main():
    parser = argparse.ArgumentParser(description='arXiv Sanity Preserver')
    parser.add_argument('command', choices=[
        'fetch', 'download', 'parse', 'analyze', 'buildsvm', 
        'make_cache', 'serve', 'twitter'
    ], help='Command to run')
    parser.add_argument('args', nargs=argparse.REMAINDER, help='Arguments for the command')
    
    args = parser.parse_args()
    
    # Map commands to modules
    command_map = {
        'fetch': 'fetch_papers',
        'download': 'download_pdfs',
        'parse': 'parse_pdf_to_text',
        'analyze': 'analyze',
        'buildsvm': 'buildsvm',
        'make_cache': 'make_cache',
        'serve': 'serve',
        'twitter': 'twitter_daemon'
    }
    
    # Import the appropriate module and run it
    module_name = command_map[args.command]
    module = __import__(module_name)
    
    # Run the module with the remaining arguments
    sys.argv = [module_name + '.py'] + args.args
    module.main()

if __name__ == '__main__':
    main()