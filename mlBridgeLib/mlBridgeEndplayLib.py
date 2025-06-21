# Contains functions for:
# 1. reading endplay compatible files
# 2. creates endplay board classes
# 3. creates an endplay polars df from boards classes
# 4. converts the endplay df to a mlBridge df


import polars as pl
import pickle
from collections import defaultdict

import endplay.parsers.lin as lin


def lin_files_to_boards_dict(lin_files_l,boards_d,bbo_lin_files_cache_file=None):
    load_count = 0
    for i,lin_file in enumerate(lin_files_l):
        if i % 10000 == 0:
            print(f'{i}/{len(lin_files_l)} {load_count=} file:{lin_file}')
        if lin_file in boards_d:
            continue
        with open(lin_file, 'r', encoding='utf-8') as f:
            try:
                boards_d[lin_file] = lin.load(f)
            except Exception as e:
                print(f'error: {i}/{len(lin_files_l)} file:{lin_file} error:{e}')
                continue
        load_count += 1
        if load_count % 1000000 == 0:
            if bbo_lin_files_cache_file is not None:
                with open(bbo_lin_files_cache_file, 'wb') as f:
                    pickle.dump(boards_d,f)
                print(f"Saved {str(bbo_lin_files_cache_file)}: len:{len(boards_d)} size:{bbo_lin_files_cache_file.stat().st_size}")
    return boards_d


def endplay_boards_to_df(pbn_deal_d: dict[str, list]) -> pl.DataFrame:
    """
    Converts a dictionary of PBN deals into a Polars DataFrame, exploding the
    score table for each board to create one row per result. This function is
    designed to be robust against missing attributes in the parsed data.
    """
    all_rows = []
    for source_file, boards in pbn_deal_d.items():
        for board in boards:
            # --- Base information for the board ---
            base_info = {
                "source_file": str(source_file),
                "board_num": getattr(board, 'board_num', None),
                "PBN": getattr(board, 'deal', None).to_pbn() if hasattr(board, 'deal') else None,
                "vulnerability": str(getattr(board, '_vul', 'None')),
                "dealer": getattr(getattr(board, 'dealer', None), 'abbr', None)
            }

            # --- Score table contains the crucial per-pair results ---
            if hasattr(board, "score_table") and board.score_table:
                for score_row in board.score_table:
                    row_data = base_info.copy()
                    row_data.update({
                        "PairId_NS": getattr(score_row, 'pair_ns', None),
                        "PairId_EW": getattr(score_row, 'pair_ew', None),
                        "Contract_str": getattr(score_row, 'contract', None),
                        "Declarer_str": getattr(score_row, 'declarer', None),
                        "Result_str": getattr(score_row, 'result', None),
                        "Score_str": getattr(score_row, 'score', None),
                    })
                    all_rows.append(row_data)
            else:
                # If no score_table, we still need a row for the deal itself,
                # but pair/result info will be missing.
                all_rows.append(base_info)
    
    if not all_rows:
        return pl.DataFrame()

    # Create the DataFrame from the list of dictionaries.
    # Polars will handle schema inference.
    return pl.DataFrame(all_rows)


# convert lin file columns to conform to bidding table columns.

# make sure the dicts have same dtypes for keys and values. It's required for some polars operations.

# all these dicts have been copied to mlBridgeLib.py. todo: remove these but requires using import mlBridgeLib.
Direction_to_NESW_d = {
    0:'N',
    1:'E',
    2:'S',
    3:'W',
    '0':'N',
    '1':'E',
    '2':'S',
    '3':'W',
    'north':'N',
    'east':'E',
    'south':'S',
    'west':'W',
    'North':'N',
    'East':'E',
    'South':'S',
    'West':'W',
    'N':'N',
    'E':'E',
    'S':'S',
    'W':'W',
    'n':'N',
    'e':'E',
    's':'S',
    'w':'W',
    None:None, # PASS
    '':'' # PASS
}

Strain_to_CDHSN_d = {
    'spades':'S',
    'hearts':'H',
    'diamonds':'D',
    'clubs':'C',
    'Spades':'S',
    'Hearts':'H',
    'Diamonds':'D',
    'Clubs':'C',
    'nt':'N',
    '♠':'S',
    '♥':'H',
    '♦':'D',
    '♣':'C',
    'NT':'N',
    'p':'PASS',
    'Pass':'PASS',
    'PASS':'PASS'
}

# todo: use mlBridgeLib.Vulnerability_to_Vul_d instead?
Vulnerability_to_Vul_d = {
    0: 'None',
    1: 'N_S',
    2: 'E_W',
    3: 'Both',
    '0': 'None',
    '1': 'N_S',
    '2': 'E_W',
    '3': 'Both',
    'None': 'None',
    'N_S': 'N_S',
    'E_W': 'E_W',
    'N-S': 'N_S',
    'E-W': 'E_W',
    'Both': 'Both',
    'NS': 'N_S',
    'EW': 'E_W',
    'All': 'Both',
    'none': 'None',
    'ns': 'N_S',
    'ew': 'E_W',
    'both': 'Both',
}

EpiVul_to_Vul_NS_Bool_d = {
    0: False,
    1: True,
    2: False,
    3: True,
}

EpiVul_to_Vul_EW_Bool_d = {
    0: False,
    1: False,
    2: True,
    3: True,
}

Dbl_to_X_d = {
    'passed':'',
    'doubled':'X',
    'redoubled':'XX',
    'p':'',
    'd':'X',
    'r':'XX',
    'p':'',
    'x':'X',
    'xx':'XX'
}


def convert_endplay_df_to_mlBridge_df(df):
    """
    This function takes a DataFrame created from endplay objects and standardizes
    it for use in the mlBridge library, including type casting and column renaming.
    """
    # Create a copy to avoid modifying the original DataFrame in place
    df = df.clone()
    
    # Ensure all required columns exist, adding them with null values if not
    required_cols = {
        'PairId_NS': pl.Utf8, 'PairId_EW': pl.Utf8, 'Player_Name_N': pl.Utf8,
        'Player_Name_S': pl.Utf8, 'Player_Name_E': pl.Utf8, 'Player_Name_W': pl.Utf8,
        'Contract_str': pl.Utf8, 'Declarer_str': pl.Utf8, 'Result_str': pl.Int64,
        'Score_str': pl.Int64
    }
    for col, dtype in required_cols.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
            
    # Normalize contract string and parse it
    df = df.with_columns(
        pl.col('Contract_str').str.replace_all(r'[^0-9A-Z]', '').alias('Contract_norm')
    )
    
    df = df.with_columns([
        pl.col('Contract_norm').str.slice(0, 1).cast(pl.UInt8, strict=False).alias('level'),
        pl.col('Contract_norm').str.slice(1, 2).alias('denom'),
        pl.col('Contract_norm').str.slice(2).alias('penalty_str')
    ])
    
    # Map penalty string to a standard format
    df = df.with_columns(
        pl.when(pl.col('penalty_str') == 'X')
        .then(pl.lit('DOUBLED'))
        .when(pl.col('penalty_str') == 'XX')
        .then(pl.lit('REDOUBLED'))
        .otherwise(pl.lit('UNDOUBLED'))
        .alias('penalty')
    )

    # Standardize player and pair columns
    df = df.with_columns([
        pl.col('Player_Name_N').alias('Player_N'),
        pl.col('Player_Name_S').alias('Player_S'),
        pl.col('Player_Name_E').alias('Player_E'),
        pl.col('Player_Name_W').alias('Player_W'),
        pl.col('Result_str').alias('result'),
        pl.col('Score_str').alias('score_int')
    ])

    # Final column selection to create a clean, standardized DataFrame
    final_cols = [
        'source_file', 'board_num', 'PBN', 'vulnerability', 'dealer', 
        'PairId_NS', 'PairId_EW', 'Player_N', 'Player_S', 'Player_E', 'Player_W',
        'level', 'denom', 'penalty', 'Declarer_str', 'result', 'score_int'
    ]
    
    # Ensure all final columns exist in the DataFrame before selecting
    for col in final_cols:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))

    return df.select(final_cols)
