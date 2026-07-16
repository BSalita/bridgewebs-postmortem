import pathlib
import requests
import sys
import polars as pl
from endplay.parsers import pbn
from urllib.parse import urlparse, parse_qs

# import endplay # for __version__
# from endplay.parsers import pbn, lin, json
# from endplay.types import Deal, Contract, Denom, Player, Penalty, Vul
# from endplay.dds import par, calc_all_tables
# from endplay.dealer import generate_deals

_APP_DIR = pathlib.Path(__file__).resolve().parent
for _p in (_APP_DIR, _APP_DIR / 'mlBridge', _APP_DIR / 'streamlitlib'):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))

from mlBridge import mlBridgeEndplayLib
from mlBridge.mlBridgeAugmentLib import (
    AllAugmentations,
)

def download_pbn_file(source_url=None):
    """
    Convert the curl command to Python using requests library
    Extracts pid, event, club from the source URL and downloads PBN data
    
    Args:
        source_url: URL in the form of https://www.bridgewebs.com/cgi-bin/bwoq/bw.cgi?pid=display_rank&event=20250526_1&club=irelandimps
    """
    # Default URL if none provided
    if source_url is None:
        source_url = "https://www.bridgewebs.com/cgi-bin/bwoq/bw.cgi?pid=display_rank&event=20250526_1&club=irelandimps"
    
    # Parse the source URL to extract parameters
    parsed_url = urlparse(source_url)
    query_params = parse_qs(parsed_url.query)
    
    # Extract required parameters
    try:
        event = query_params['event'][0]
        club = query_params['club'][0]
        print(f"Extracted parameters - event: {event}, club: {club}")
    except (KeyError, IndexError) as e:
        print(f"Error extracting parameters from URL: {e}")
        print(f"URL: {source_url}")
        sys.exit(1)
    
    # Base URL for the request
    url = "https://www.bridgewebs.com/cgi-bin/bwoq/bw.cgi"
    
    # Query parameters for PBN download
    params = {
        'pid': 'display_hands',
        'msec': '1',
        'event': event,
        'wd': '1',
        'club': club,
        'deal_format': 'pbn'
    }
    
    # Headers with the source URL as referer
    headers = {
        'Referer': source_url
    }
    
    print(f"Making request to: {url}")
    print(f"Parameters: {params}")
    print(f"Referer: {source_url}")
    
    try:
        # Make the GET request (equivalent to curl --silent --show-error --fail)
        response = requests.get(url, params=params, headers=headers)
        
        # Raise an exception for bad status codes (equivalent to --fail)
        response.raise_for_status()
        
        # Write to file (equivalent to -o x.pbn)
        with open('x.pbn', 'wb') as f:
            f.write(response.content)
        
        print("Successfully downloaded x.pbn")
        
        # Parse the PBN file using endplay
        print("Parsing PBN file using endplay...")
        file_content = response.content.decode('utf-8')
        boards = pbn.loads(file_content)
        
        print(f"Parsed {len(boards)} boards from PBN file")
        
        # Convert to Polars DataFrame using existing library
        print("Converting to Polars DataFrame...")
        df = mlBridgeEndplayLib.endplay_boards_to_df({'x.pbn': boards})
        
        # Convert to mlBridge format
        df = mlBridgeEndplayLib.convert_endplay_df_to_mlBridge_df(df)
        
        # Display the DataFrame
        print("\nPolars DataFrame:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Display some basic statistics
        print(f"\nBasic Statistics:")
        print(f"Number of boards: {len(df)}")
        if 'Board' in df.columns:
            print(f"Board numbers: {df['Board'].min()} - {df['Board'].max()}")
        if 'Contract' in df.columns:
            print(f"Unique contracts: {df['Contract'].n_unique()}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing PBN file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Example usage with the provided URL
    source_url = "https://www.bridgewebs.com/cgi-bin/bwoq/bw.cgi?pid=display_rank&event=20250526_1&club=irelandimps"
    df = download_pbn_file(source_url) 