import argparse
from mlBridgeWebsLib import process_bridgewebs_url
import polars as pl
import pprint

def main():
    """
    This script serves as a command-line wrapper to fetch, parse, augment, and filter
    bridge tournament data from a bridgewebs.com URL using the
    process_bridgewebs_url function.
    """
    parser = argparse.ArgumentParser(description="Fetch and process bridge tournament results from a Bridgewebs URL.")
    parser.add_argument(
        'url',
        nargs='?',
        default='https://www.bridgewebs.com/cgi-bin/bwoq/bw.cgi?pid=display_rank&event=20250526_1&club=irelandimps',
        help="The URL of the event's ranking page. Defaults to a hardcoded URL."
    )
    args = parser.parse_args()

    print(f"Processing URL: {args.url}")
    df, info = process_bridgewebs_url(url=args.url)

    if df is not None and info is not None:
        print("\n\n--- Processed DataFrame (filtered for top pair) ---")
        print(df)
        print("\n\n--- Player and Game Info ---")
        pprint.pprint(info)
    else:
        print("No results could be processed from the URL.")


if __name__ == "__main__":
    main() 