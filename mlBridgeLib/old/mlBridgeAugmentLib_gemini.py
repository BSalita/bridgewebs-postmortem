# Refactored mlBridgeAugmentLib.py
#
# This script contains classes to augment a Polars DataFrame with various metrics
# and features related to the game of bridge.
#
# ## Suggested Class Structure:
# - BridgeDataAugmenter: Main orchestrator.
# - DealAugmenter: Augmentations related to the deal as a whole (PBN, Dealer, Vul).
# - HandEvaluationAugmenter: Augmentations related to individual hand evaluations (HCP, Suit Length, LoTT, etc.).
# - DDSdAugmenter: Double Dummy (DD) and Single Dummy (SD) calculations, Par scores, and EV.
# - ContractAugmenter: Processing of the actual played contract (Declarer, Bid, etc.).
# - ScoreAugmenter: Calculations based on the contract's outcome (Result, Score, Differences).
# - MatchPointAugmenter: Matchpoint scoring calculations.
# - ImpAugmenter: Placeholder for IMP scoring calculations.
# - Utilities: Helper functions and constants.

import polars as pl
from collections import defaultdict
import sys
import pathlib
import time # For _time_operation and performance prints

# Assuming mlBridgeLib.mlBridgeLib is in the PYTHONPATH or same directory
# import mlBridgeLib.mlBridgeLib as mlBridgeLib # If it still exists and is needed directly
# For now, use constants directly or redefine minimally for this refactoring
# (Full dependency on mlBridgeLib.mlBridgeLib is assumed to be resolvable in the user's environment)

from mlBridgeLib.mlBridgeLib import (
    NESW, SHDC, SHDCN, NS_EW,
    PlayerDirectionToPairDirection,
    NextPosition,
    PairDirectionToOpponentPairDirection,
    score
)

# Endplay imports
from endplay.types import Deal, Contract, Denom, Player, Penalty, Vul
from endplay.dds import calc_all_tables, par
from endplay.dealer import generate_deals # For SD calculations

# ## Utility Functions
def _time_operation(operation_name, func, *args, **kwargs):
    """Helper function to time an operation."""
    t_start = time.time()
    result = func(*args, **kwargs)
    print(f"TIMER: {operation_name}: {time.time()-t_start:.4f} seconds")
    return result

def _assert_columns_exist(df, columns, context=""):
    """Checks if required columns exist in the DataFrame."""
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for {context}: {missing_cols}. Available: {df.columns}")

def _assert_new_columns_dont_exist(df, new_columns, context=""):
    """Checks if new columns to be created already exist."""
    existing_new_cols = [col for col in new_columns if col in df.columns]
    if existing_new_cols:
        # Consider if this should be a warning or an error based on desired idempotency
        print(f"WARNING: Columns intended for creation already exist for {context}: {existing_new_cols}. They will be overwritten.")


# Moved from original top-level
# This function calculates potential scores for contracts.
# It's used by DDSdAugmenter and potentially ScoreAugmenter.
def calculate_contract_score_tables():
    """
    Calculates dictionaries and a DataFrame of contract scores for various outcomes.
    Sets (down tricks) are assumed to be penalty doubled for the `scores_d` dictionary
    used in DD/SD expected value calculations. `all_scores_d` includes passed, doubled, redoubled.

    Outputs:
        all_scores_d (dict): {(level, suit_char, tricks_taken, is_vul, penalty_abbr): score}
        scores_d (dict): {(level, suit_char, tricks_taken, is_vul): score} (sets are doubled)
        scores_df (pl.DataFrame): DataFrame with columns 'Score_[Level][Suit]'
                                   each containing a list [non_vul_score, vul_score]
                                   for a range of tricks taken.
    """
    scores_d = {}
    all_scores_d = {(None, None, None, None, None): 0} # PASS score

    suit_to_denom = {'C': Denom.clubs, 'D': Denom.diamonds, 'H': Denom.hearts, 'S': Denom.spades, 'N': Denom.nt}

    for suit_char, denom in suit_to_denom.items():
        for level in range(1, 8): # contract level
            for tricks_taken in range(14): # 0 to 13 tricks
                result = tricks_taken - 6 - level

                # For scores_d: sets are always penalty doubled
                # Contract object needs a declarer, using North as arbitrary. Vulnerability is key.
                # Passed penalty if made, Doubled penalty if down
                contract_made_nv = Contract(level=level, denom=denom, declarer=Player.north, penalty=Penalty.passed, result=result)
                contract_down_doubled_nv = Contract(level=level, denom=denom, declarer=Player.north, penalty=Penalty.doubled, result=result)

                contract_made_v = Contract(level=level, denom=denom, declarer=Player.north, penalty=Penalty.passed, result=result)
                contract_down_doubled_v = Contract(level=level, denom=denom, declarer=Player.north, penalty=Penalty.doubled, result=result)

                scores_d[(level, suit_char, tricks_taken, False)] = contract_made_nv.score(Vul.none) if result >= 0 else contract_down_doubled_nv.score(Vul.none)
                scores_d[(level, suit_char, tricks_taken, True)] = contract_made_v.score(Vul.both) if result >= 0 else contract_down_doubled_v.score(Vul.both)

                # For all_scores_d: all penalty variations
                penalties = {'': Penalty.passed, 'X': Penalty.doubled, 'XX': Penalty.redoubled}
                for pen_abbr, pen_obj in penalties.items():
                    c_nv = Contract(level=level, denom=denom, declarer=Player.north, penalty=pen_obj, result=result)
                    c_v = Contract(level=level, denom=denom, declarer=Player.north, penalty=pen_obj, result=result)
                    all_scores_d[(level, suit_char, tricks_taken, False, pen_abbr)] = c_nv.score(Vul.none)
                    all_scores_d[(level, suit_char, tricks_taken, True, pen_abbr)] = c_v.score(Vul.both)

    # Create score DataFrame from scores_d (for EV calculations primarily)
    sd_polars = defaultdict(list)
    for suit_char in 'CDHSN': # Order for dataframe columns
        for level in range(1,8):
            col_name = f'Score_{str(level)}{suit_char}'
            trick_scores_for_contract = []
            for tricks_taken in range(14): # 0-13 tricks
                # Store as [non_vul_score, vul_score]
                trick_scores_for_contract.append([
                    scores_d[(level, suit_char, tricks_taken, False)],
                    scores_d[(level, suit_char, tricks_taken, True)]
                ])
            sd_polars[col_name] = trick_scores_for_contract

    # Ensuring all lists have the same length (14 for tricks 0-13)
    # Polars DataFrame construction from dict of lists requires equal length lists
    scores_pl_df = pl.DataFrame(dict(sd_polars)) # pl.DataFrame will have 14 rows
                                              # Each cell is a list [NV_score, V_score]

    return all_scores_d, scores_d, scores_pl_df

# ## Augmenter Classes

class DealAugmenter:
    """Augmentations related to the deal as a whole (PBN, Dealer, Vul)."""
    def __init__(self):
        pass

    # Inputs: df column ['PBN']
    # Outputs: df columns ['Hand_N', 'Hand_E', 'Hand_S', 'Hand_W']
    # Description: Extracts individual hand strings from the PBN string.
    def _add_hand_nesw_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['PBN'], "DealAugmenter._add_hand_nesw_columns")
        new_cols = [f'Hand_{d}' for d in NESW]
        # _assert_new_columns_dont_exist(df, new_cols, "DealAugmenter._add_hand_nesw_columns") # Optional check

        if 'Hand_N' not in df.columns: # Check if augmentation already done
            df = df.with_columns([
                pl.col('PBN')
                .str.slice(2) # Skip "N:"
                .str.split(' ')
                .list.get(i)
                .alias(f'Hand_{direction}')
                for i, direction in enumerate(NESW)
            ])
        return df

    # Inputs: df column ['PBN']
    # Outputs: df column ['Hands'] (List of lists of strings, e.g., [['AKQ', 'J10', ...], ...])
    # Description: Creates a nested list structure of hands and suits from PBN.
    def _add_hands_lists_column(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['PBN'], "DealAugmenter._add_hands_lists_column")
        if 'Hands' not in df.columns:
            df = df.with_columns([
                pl.col('PBN')
                .str.slice(2)
                .str.split(' ')
                .list.eval(pl.element().str.split('.'), parallel=True)
                .alias('Hands')
            ])
        return df

    # Inputs: df columns ['Hand_N', ..., 'Hand_W'] (from _add_hand_nesw_columns)
    # Outputs: df columns ['Suit_N_S', ..., 'Suit_W_C'] (individual suit holdings)
    # Description: Extracts suit strings for each hand.
    def _add_suit_nesw_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, [f'Hand_{d}' for d in NESW], "DealAugmenter._add_suit_nesw_columns")
        if 'Suit_N_C' not in df.columns: # Check one to see if augmentation is done
            for d in NESW:
                for i, s in enumerate(SHDC):
                    df = df.with_columns([
                        pl.col(f'Hand_{d}')
                        .str.split('.')
                        .list.get(i) # PBN typically S,H,D,C
                        .alias(f'Suit_{d}_{s}')
                    ])
        return df

    # Inputs: df column ['board_number'] (if 'Dealer' not present)
    # Outputs: df column ['Dealer'] ('N', 'E', 'S', or 'W')
    # Description: Determines the dealer based on board number if not already provided.
    def _add_dealer_column(self, df: pl.DataFrame) -> pl.DataFrame:
        if 'Dealer' not in df.columns:
            _assert_columns_exist(df, ['board_number'], "DealAugmenter._add_dealer_column (fallback)")
            def board_number_to_dealer(bn_series: pl.Series) -> pl.Series:
                return bn_series.map_elements(lambda bn: NESW[(bn - 1) % 4], return_dtype=pl.String)

            df = df.with_columns(
                board_number_to_dealer(pl.col('board_number')).alias('Dealer')
            )
        return df

    # Inputs: df columns ['Vul'] or ['board_number'] (if 'Vul' not present)
    # Outputs: df columns ['iVul', 'Vul', 'Vul_NS', 'Vul_EW']
    # Description: Standardizes vulnerability representation.
    def _add_vulnerability_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        # Ensure 'Vul' string column exists
        if 'Vul' not in df.columns:
            _assert_columns_exist(df, ['board_number'], "DealAugmenter._add_vulnerability_columns (Vul fallback)")
            # Vulnerability from board number: 1=None, 2=N/S, 3=E/W, 4=Both, then repeats for 5-8, 9-12, 13-16
            # Dealer: N E S W N E S W N E S W N E S W
            # Vul:    - NS EW B NS EW B - EW B - NS B - NS EW (incorrect sequence in original code)
            # Standard pattern: Board 1 (N deal, None vul), 2 (E deal, NS vul), 3 (S deal, EW vul), 4 (W deal, Both vul) etc.
            vul_map = {
                # (Board - 1) % 16 gives 0-15
                0: 'None', 1: 'N_S', 2: 'E_W', 3: 'Both',
                4: 'N_S', 5: 'E_W', 6: 'Both', 7: 'None',
                8: 'E_W', 9: 'Both', 10: 'None', 11: 'N_S',
                12: 'Both', 13: 'None', 14: 'N_S', 15: 'E_W'
            }
            df = df.with_columns(
                pl.col('board_number').map_elements(lambda bn: vul_map[(bn - 1) % 16], return_dtype=pl.String).alias('Vul')
            )

        # Ensure 'iVul' integer column exists
        if 'iVul' not in df.columns:
            _assert_columns_exist(df, ['Vul'], "DealAugmenter._add_vulnerability_columns (iVul creation)")
            vul_to_int = {'None': 0, 'N_S': 1, 'E_W': 2, 'Both': 3}
            df = df.with_columns(
                pl.col('Vul').map_elements(lambda v: vul_to_int.get(v), return_dtype=pl.UInt8).alias('iVul')
            )

        # Ensure 'Vul_NS' and 'Vul_EW' boolean columns exist
        if 'Vul_NS' not in df.columns:
            _assert_columns_exist(df, ['Vul'], "DealAugmenter._add_vulnerability_columns (Vul_NS/EW creation)")
            df = df.with_columns([
                pl.col('Vul').is_in(['N_S', 'Both']).alias('Vul_NS'),
                pl.col('Vul').is_in(['E_W', 'Both']).alias('Vul_EW')
            ])
        return df

    # Inputs: None (adds default columns if they don't exist)
    # Outputs: df columns ['group_id', 'session_id', 'section_name']
    # Description: Adds default/placeholder columns for grouping and session info.
    def _add_default_metadata_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        defaults = {
            'group_id': 0,
            'session_id': 0, # Matchpoint calculations are often grouped by session
            'section_name': ""
        }
        for col, val in defaults.items():
            if col not in df.columns:
                lit_val = pl.lit(val, dtype=pl.Int32 if isinstance(val, int) else pl.String)
                df = df.with_columns(lit_val.alias(col))
        return df

    def perform_augmentations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Applies all deal-related augmentations."""
        df = _time_operation("DealAugmenter: Add Default Metadata", self._add_default_metadata_columns, df)
        df = _time_operation("DealAugmenter: Add Hand NESW", self._add_hand_nesw_columns, df)
        df = _time_operation("DealAugmenter: Add Hands Lists", self._add_hands_lists_column, df)
        df = _time_operation("DealAugmenter: Add Suit NESW", self._add_suit_nesw_columns, df)
        df = _time_operation("DealAugmenter: Add Dealer", self._add_dealer_column, df)
        df = _time_operation("DealAugmenter: Add Vulnerability", self._add_vulnerability_columns, df)
        return df

class HandEvaluationAugmenter:
    """Augmentations related to individual hand evaluations (HCP, Suit Length, LoTT, etc.)."""
    def __init__(self):
        # Criteria from original ResultAugmenter
        self.suit_quality_criteria = {
            "Biddable": lambda sl, hcp: sl.ge(5) | (sl.eq(4) & hcp.ge(3)),
            "Rebiddable": lambda sl, hcp: sl.ge(6) | (sl.eq(5) & hcp.ge(3)),
            "Twice_Rebiddable": lambda sl, hcp: sl.ge(7) | (sl.eq(6) & hcp.ge(3)),
            "Strong_Rebiddable": lambda sl, hcp: sl.ge(6) & hcp.ge(9),
            "Solid": lambda sl, hcp: hcp.ge(9), # Original todo: 6 card requires ten
        }
        self.stopper_criteria = {
            "At_Best_Partial_Stop_In": lambda sl, hcp: (sl + hcp).lt(4),
            "Partial_Stop_In": lambda sl, hcp: (sl + hcp).ge(4),
            "Likely_Stop_In": lambda sl, hcp: (sl + hcp).ge(5),
            "Stop_In": lambda sl, hcp: hcp.ge(4) | (sl + hcp).ge(6),
            "At_Best_Stop_In": lambda sl, hcp: (sl + hcp).ge(7),
            "Two_Stops_In": lambda sl, hcp: (sl + hcp).ge(8),
        }

    # Inputs: df columns ['Suit_N_S', ..., 'Suit_W_C'] (from DealAugmenter)
    # Outputs: df columns ['C_NSA', ..., 'C_W2C'] (One-hot encoded cards)
    # Description: Converts suit holdings into one-hot encoded card presence.
    def _add_one_hot_encoded_card_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, [f'Suit_{d}_{s}' for d in NESW for s in SHDC], "HandEvaluationAugmenter._add_one_hot_encoded_card_columns")
        if 'C_NSA' not in df.columns: # Check one example
            lazy_df = df.lazy()
            lazy_cards_df = lazy_df.with_columns([
                pl.col(f'Suit_{direction}_{suit}').str.contains(rank).alias(f'C_{direction}{suit}{rank}')
                for direction in NESW
                for suit in SHDC
                for rank in 'AKQJT98765432'
            ])
            df = lazy_cards_df.collect()
        return df

    # Inputs: df columns ['C_NSA', ..., 'C_W2C'] (from _add_one_hot_encoded_card_columns)
    # Outputs: df columns ['HCP_N_S', ..., 'HCP_EW']
    # Description: Calculates High Card Points (HCP) for hands and partnerships.
    def _add_hcp_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        card_cols_exist = all(f'C_{d}{s}A' in df.columns for d in NESW for s in SHDC) # Check one rank 'A' for each suit/player
        if not card_cols_exist:
             raise ValueError("Missing one-hot encoded card columns for HCP calculation. Run _add_one_hot_encoded_card_columns first.")

        if 'HCP_N_S' not in df.columns:
            hcp_d = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
            hcp_suit_exprs = []
            for d in NESW:
                for s in SHDC:
                    suit_hcp_expr = pl.sum_horizontal([
                        pl.col(f'C_{d}{s}{r}').cast(pl.UInt8) * v for r, v in hcp_d.items()
                        if f'C_{d}{s}{r}' in df.columns # Ensure column exists
                    ]).alias(f'HCP_{d}_{s}')
                    hcp_suit_exprs.append(suit_hcp_expr)
            df = df.with_columns(hcp_suit_exprs)

            hcp_direction_expr = [
                pl.sum_horizontal([pl.col(f'HCP_{d}_{s}') for s in SHDC]).alias(f'HCP_{d}')
                for d in NESW
            ]
            df = df.with_columns(hcp_direction_expr)

            hcp_partnership_expr = [
                (pl.col('HCP_N') + pl.col('HCP_S')).alias('HCP_NS'),
                (pl.col('HCP_E') + pl.col('HCP_W')).alias('HCP_EW')
            ]
            df = df.with_columns(hcp_partnership_expr)
        return df

    # Inputs: df columns ['Suit_N_S', ..., 'Suit_W_C']
    # Outputs: df columns ['QT_N_S', ..., 'QT_EW']
    # Description: Calculates Quick Tricks for hands and partnerships.
    def _add_quick_tricks_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, [f'Suit_{d}_{s}' for d in NESW for s in SHDC], "HandEvaluationAugmenter._add_quick_tricks_columns")
        if 'QT_N_S' not in df.columns:
            qt_dict = {'AK': 2.0, 'AQ': 1.5, 'A': 1.0, 'KQ': 1.0, 'K': 0.5}
            qt_expr = []
            for d in NESW:
                for s in SHDC:
                    # Build expression for current suit
                    current_expr = pl.lit(0.0) # Start with 0.0
                    # Iterate in order of preference (AK before A, KQ before K)
                    if 'AK' in qt_dict:
                         current_expr = pl.when(pl.col(f'Suit_{d}_{s}').str.starts_with('AK')).then(pl.lit(qt_dict['AK'])).otherwise(current_expr)
                    if 'AQ' in qt_dict:
                         current_expr = pl.when(pl.col(f'Suit_{d}_{s}').str.starts_with('AQ')).then(pl.lit(qt_dict['AQ'])).otherwise(current_expr)
                    if 'A' in qt_dict and 'AK' not in qt_dict and 'AQ' not in qt_dict : # only if not covered by AK, AQ
                         current_expr = pl.when(pl.col(f'Suit_{d}_{s}').str.starts_with('A')).then(pl.lit(qt_dict['A'])).otherwise(current_expr)
                    elif 'A' in qt_dict : # A can be matched by AKQ, AK, AQ, A. Need to be careful.
                         current_expr = pl.when(pl.col(f'Suit_{d}_{s}').str.starts_with('A') &
                                               ~pl.col(f'Suit_{d}_{s}').str.starts_with('AK') &
                                               ~pl.col(f'Suit_{d}_{s}').str.starts_with('AQ')
                                               ).then(pl.lit(qt_dict['A'])).otherwise(current_expr)

                    if 'KQ' in qt_dict:
                         current_expr = pl.when(pl.col(f'Suit_{d}_{s}').str.starts_with('KQ')).then(pl.lit(qt_dict['KQ'])).otherwise(current_expr)
                    if 'K' in qt_dict and 'KQ' not in qt_dict : # only if not K covered by KQ
                         current_expr = pl.when(pl.col(f'Suit_{d}_{s}').str.starts_with('K')).then(pl.lit(qt_dict['K'])).otherwise(current_expr)
                    elif 'K' in qt_dict :
                         current_expr = pl.when(pl.col(f'Suit_{d}_{s}').str.starts_with('K') &
                                               ~pl.col(f'Suit_{d}_{s}').str.starts_with('KQ')
                                               ).then(pl.lit(qt_dict['K'])).otherwise(current_expr)
                    qt_expr.append(current_expr.alias(f'QT_{d}_{s}'))

            df = df.with_columns(qt_expr)

            direction_qt = [
                pl.sum_horizontal([pl.col(f'QT_{d}_{s}') for s in SHDC]).alias(f'QT_{d}')
                for d in NESW
            ]
            df = df.with_columns(direction_qt)

            partnership_qt = [
                (pl.col('QT_N') + pl.col('QT_S')).alias('QT_NS'),
                (pl.col('QT_E') + pl.col('QT_W')).alias('QT_EW')
            ]
            df = df.with_columns(partnership_qt)
        return df

    # Inputs: df columns ['Suit_N_S', ..., 'Suit_W_C']
    # Outputs: df columns ['SL_N_S', ..., 'SL_EW_C', 'SL_Max_NS', 'SL_Max_EW']
    # Description: Calculates suit lengths and identifies the partnership's longest suit.
    def _add_suit_length_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, [f'Suit_{d}_{s}' for d in NESW for s in SHDC], "HandEvaluationAugmenter._add_suit_length_columns")

        if 'SL_N_S' not in df.columns:
            sl_nesw_columns = [
                pl.col(f"Suit_{direction}_{suit}").map_elements(lambda x: len(x) if x else 0, return_dtype=pl.UInt8).alias(f"SL_{direction}_{suit}")
                for direction in NESW
                for suit in SHDC
            ]
            df = df.with_columns(sl_nesw_columns)

        if 'SL_NS_S' not in df.columns:
            sl_ns_ew_columns = [
                (pl.col(f"SL_{pair[0]}_{suit}") + pl.col(f"SL_{pair[1]}_{suit}")).alias(f"SL_{pair}_{suit}")
                for pair in NS_EW
                for suit in SHDC
            ]
            df = df.with_columns(sl_ns_ew_columns)

        if 'SL_Max_NS' not in df.columns: # Column containing the NAME of the column with max length for that pair
            for pair_short in NS_EW: # 'NS', 'EW'
                suit_cols_for_pair = [f'SL_{pair_short}_{s}' for s in SHDC]
                # Find the maximum length among the suit length columns for the pair
                max_len_expr = pl.max_horizontal(suit_cols_for_pair)
                # Create an expression to pick the column name corresponding to that max length
                # This is tricky in Polars. We want the *name* of the column.
                # One way: iterate and build a when/then chain
                when_expr = None
                for suit_col_name in suit_cols_for_pair:
                    condition = (pl.col(suit_col_name) == max_len_expr)
                    if when_expr is None:
                        when_expr = pl.when(condition).then(pl.lit(suit_col_name))
                    else:
                        when_expr = when_expr.when(condition).then(pl.lit(suit_col_name))
                df = df.with_columns(when_expr.otherwise(pl.lit(None)).alias(f'SL_Max_{pair_short}'))
        return df

    # Inputs: df columns ['SL_N_S', ..., 'SL_W_C']
    # Outputs: df columns like 'SL_N_CDHS', 'SL_N_ML_SJ', etc.
    # Description: Creates array and string representations of suit length distributions.
    def _add_suit_length_array_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, [f'SL_{d}_{s}' for d in NESW for s in SHDC], "HandEvaluationAugmenter._add_suit_length_array_columns")
        if 'SL_N_CDHS' not in df.columns: # Check one
            for d in NESW:
                # Polars equivalent for creating these arrays/strings from existing SL columns
                # Example for SL_[D]_CDHS_SJ (shape as string '5-4-2-2')
                df = df.with_columns(
                    pl.concat_str(
                        [pl.col(f"SL_{d}_{s}").cast(pl.String) for s in 'CDHS'], # Note suit order
                        separator="-"
                    ).alias(f"SL_{d}_CDHS_SJ")
                )
                # For SL_[D]_ML (sorted lengths) and SL_[D]_ML_SJ (sorted lengths as string)
                # This is more complex, requires UDF or multiple steps if pure Polars
                # For simplicity in this refactor, we'll assume SL_N_ML_SJ is a key target like in original.
                # A UDF that takes a row (struct of SL_d_C, SL_d_D, SL_d_H, SL_d_S) and returns sorted string:
                def get_ml_sj(row_struct: dict) -> str:
                    lengths = [row_struct[f'SL_{d}_C'], row_struct[f'SL_{d}_D'], row_struct[f'SL_{d}_H'], row_struct[f'SL_{d}_S']]
                    return "-".join(map(str, sorted(lengths, reverse=True)))

                df = df.with_columns(
                    pl.struct([f"SL_{d}_{s}" for s in 'CDHS']) # struct uses C,D,H,S order for consistency
                     .map_elements(get_ml_sj, return_dtype=pl.String)
                     .alias(f"SL_{d}_ML_SJ") # Most Longest Suit Joined
                )
        return df

    # Inputs: df columns ['SL_N_S', ..., 'SL_W_C']
    # Outputs: df columns ['DP_N_S', ..., 'DP_EW']
    # Description: Calculates distribution points.
    def _add_distribution_point_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, [f'SL_{d}_{s}' for d in NESW for s in SHDC], "HandEvaluationAugmenter._add_distribution_point_columns")
        if 'DP_N_S' not in df.columns:
            dp_columns = []
            for direction in NESW:
                for suit in SHDC:
                    dp_columns.append(
                        pl.when(pl.col(f"SL_{direction}_{suit}") == 0).then(3)
                        .when(pl.col(f"SL_{direction}_{suit}") == 1).then(2)
                        .when(pl.col(f"SL_{direction}_{suit}") == 2).then(1)
                        .otherwise(0)
                        .cast(pl.UInt8) # Ensure type
                        .alias(f"DP_{direction}_{suit}")
                    )
            df = df.with_columns(dp_columns)
            df = df.with_columns([
                pl.sum_horizontal([f'DP_{d}_{s}' for s in SHDC]).alias(f'DP_{d}')
                for d in NESW
            ])
            df = df.with_columns([
                (pl.col('DP_N') + pl.col('DP_S')).alias('DP_NS'),
                (pl.col('DP_E') + pl.col('DP_W')).alias('DP_EW'),
            ])
        return df

    # Inputs: df columns ['HCP_N_S', ..., 'HCP_W_C', 'DP_N_S', ..., 'DP_W_C']
    # Outputs: df columns ['Total_Points_N_S', ..., 'Total_Points_EW']
    # Description: Calculates total points (HCP + DP).
    def _add_total_point_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df,
                              [f'HCP_{d}_{s}' for d in NESW for s in SHDC] +
                              [f'DP_{d}_{s}' for d in NESW for s in SHDC],
                              "HandEvaluationAugmenter._add_total_point_columns")
        if 'Total_Points_N_S' not in df.columns:
            # Original had a todo: "Don't forget to adjust Total_Points for singleton king and doubleton queen."
            # This refactoring keeps current logic. Adjustment would be complex here.
            df = df.with_columns([
                (pl.col(f'HCP_{d}_{s}') + pl.col(f'DP_{d}_{s}')).alias(f'Total_Points_{d}_{s}')
                for d in NESW for s in SHDC
            ])
            df = df.with_columns([
                pl.sum_horizontal([f'Total_Points_{d}_{s}' for s in SHDC]).alias(f'Total_Points_{d}')
                for d in NESW
            ])
            df = df.with_columns([
                (pl.col('Total_Points_N') + pl.col('Total_Points_S')).alias('Total_Points_NS'),
                (pl.col('Total_Points_E') + pl.col('Total_Points_W')).alias('Total_Points_EW'),
            ])
        return df

    # Inputs: df columns ['SL_Max_NS', 'SL_Max_EW', 'SL_NS_S'...'SL_EW_C', 'DD_NS_S'...'DD_EW_C']
    # Outputs: df columns ['LoTT_SL', 'LoTT_DD', 'LoTT_Diff'] and intermediate LoTT columns.
    # Description: Calculates Law of Total Tricks (LoTT) related metrics.
    def _add_lott_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        # This function depends on DD (Double Dummy) columns which are created by DDSdAugmenter.
        # Ensure this method is called *after* DDSdAugmenter.
        required_cols = ['SL_Max_NS', 'SL_Max_EW'] + \
                        [f'SL_{p}_{s}' for p in NS_EW for s in SHDC] + \
                        [f'DD_{p}_{s}' for p in NS_EW for s in SHDC] # DD columns like DD_NS_S
        _assert_columns_exist(df, required_cols, "HandEvaluationAugmenter._add_lott_columns")

        if 'LoTT_SL' not in df.columns:
            for pair_max_col_name_holder in ['SL_Max_NS', 'SL_Max_EW']: # e.g. SL_Max_NS holds 'SL_NS_S'
                pair_short = pair_max_col_name_holder[-2:] # NS or EW

                # For each potential max suit ('SL_NS_S', 'SL_NS_H', etc.)
                sl_cols_for_lott = []
                dd_cols_for_lott = []

                for suit_char_for_lott in SHDC:
                    # Name of the actual column holding suit length, e.g. SL_NS_S
                    current_sl_col_name = f"SL_{pair_short}_{suit_char_for_lott}"
                    # Name of the corresponding DD column, e.g. DD_NS_S
                    current_dd_col_name = f"DD_{pair_short}_{suit_char_for_lott}"

                    # Create LoTT_SL_[Pair]_[Suit] (e.g. LoTT_SL_NS_S)
                    # Value is from SL_[Pair]_[Suit] if this suit is the max for the pair, else 0
                    sl_cols_for_lott.append(
                        pl.when(pl.col(pair_max_col_name_holder) == current_sl_col_name)
                        .then(pl.col(current_sl_col_name))
                        .otherwise(0).alias(f"LoTT_SL_{pair_short}_{suit_char_for_lott}")
                    )
                    # Create LoTT_DD_[Pair]_[Suit] (e.g. LoTT_DD_NS_S)
                    dd_cols_for_lott.append(
                        pl.when(pl.col(pair_max_col_name_holder) == current_sl_col_name)
                        .then(pl.col(current_dd_col_name))
                        .otherwise(0).alias(f"LoTT_DD_{pair_short}_{suit_char_for_lott}")
                    )
                df = df.with_columns(sl_cols_for_lott + dd_cols_for_lott)

                # Sum horizontally for the pair
                df = df.with_columns([
                    pl.sum_horizontal(pl.col(f'^LoTT_SL_{pair_short}_[SHDC]$')).alias(f'LoTT_SL_{pair_short}'),
                    pl.sum_horizontal(pl.col(f'^LoTT_DD_{pair_short}_[SHDC]$')).alias(f'LoTT_DD_{pair_short}'),
                ])

            df = df.with_columns([
                pl.sum_horizontal(pl.col(r'^LoTT_SL_(NS|EW)$')).alias('LoTT_SL'),
                pl.sum_horizontal(pl.col(r'^LoTT_DD_(NS|EW)$')).alias('LoTT_DD')
            ])
            df = df.with_columns((pl.col('LoTT_SL') - pl.col('LoTT_DD').cast(pl.Int8)).alias('LoTT_Diff'))
        return df

    # Inputs: df columns ['SL_N_S', ..., 'HCP_W_C']
    # Outputs: df columns like ['Biddable_N_S', 'Stop_In_N_S', ...]
    # Description: Adds boolean indicators for suit quality and stoppers.
    def _add_suit_quality_stopper_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        required_sl = [f'SL_{d}_{s}' for d in NESW for s in SHDC]
        required_hcp = [f'HCP_{d}_{s}' for d in NESW for s in SHDC]
        _assert_columns_exist(df, required_sl + required_hcp, "HandEvaluationAugmenter._add_suit_quality_stopper_columns")

        if 'Biddable_N_S' not in df.columns: # Check one
            series_expressions = []
            all_criteria = {**self.suit_quality_criteria, **self.stopper_criteria}
            for direction in NESW:
                for suit in SHDC:
                    for series_type, criteria_fn in all_criteria.items():
                        series_expressions.append(
                            criteria_fn(
                                df[f"SL_{direction}_{suit}"], # This passes Series to lambda
                                df[f"HCP_{direction}_{suit}"]
                            ).alias(f"{series_type}_{direction}_{suit}") # Result of lambda is a boolean Series
                        )
            # Add placeholder columns from original logic
            # These might need more sophisticated rules based on the criteria above or other factors.
            # For now, adding them as false literals as in the original.
            # placeholder_cols = [
            #     pl.lit(False).alias("Forcing_One_Round"),
            #     pl.lit(False).alias("Opponents_Cannot_Play_Undoubled_Below_2N"),
            #     pl.lit(False).alias("Forcing_To_2N"),
            #     pl.lit(False).alias("Forcing_To_3N"),
            # ]
            df = df.with_columns(series_expressions)# + placeholder_cols) # Hold off on placeholders unless logic is clear
        return df

    # Inputs: df columns ['SL_N_ML_SJ', ..., 'SL_W_ML_SJ'] (Sorted suit lengths as string)
    # Outputs: df columns ['Balanced_N', ..., 'Balanced_W']
    # Description: Adds boolean indicators for balanced hands.
    def _add_balanced_hand_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, [f'SL_{d}_ML_SJ' for d in NESW], "HandEvaluationAugmenter._add_balanced_hand_columns")
        if 'Balanced_N' not in df.columns: # Check one
            balanced_exprs = []
            for direction in NESW:
                # Original logic: Balanced if 4333, 4432, or 5m332 (5 card minor, 3-3-2 in others)
                # The SL_d_ML_SJ is 'MajorLen-NextMajorLen-Minor1-Minor2' or similar sorted.
                # Need SL_d_C, SL_d_D for the 5-card minor check.
                _assert_columns_exist(df, [f'SL_{direction}_C', f'SL_{direction}_D'], "HandEvaluationAugmenter._add_balanced_hand_columns (minor check)")

                is_pattern_4333 = pl.col(f"SL_{direction}_ML_SJ") == '4-3-3-3'
                is_pattern_4432 = pl.col(f"SL_{direction}_ML_SJ") == '4-4-3-2'

                # 5-card minor with 3-3-2 distribution (e.g., 5 Clubs, 3 Diamonds, 3 Hearts, 2 Spades or 5D,3C,3H,2S etc.)
                # This means the sorted lengths are 5-3-3-2
                is_pattern_5332 = pl.col(f"SL_{direction}_ML_SJ") == '5-3-3-2'
                is_5_card_minor = (pl.col(f"SL_{direction}_C") == 5) | (pl.col(f"SL_{direction}_D") == 5)

                balanced_condition = is_pattern_4333 | is_pattern_4432 | (is_pattern_5332 & is_5_card_minor)

                balanced_exprs.append(balanced_condition.alias(f"Balanced_{direction}"))
            df = df.with_columns(balanced_exprs)
        return df

    def perform_augmentations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Applies all hand evaluation augmentations."""
        df = _time_operation("HandEval: Add One-Hot Cards", self._add_one_hot_encoded_card_columns, df)
        df = _time_operation("HandEval: Add HCP", self._add_hcp_columns, df)
        df = _time_operation("HandEval: Add Quick Tricks", self._add_quick_tricks_columns, df)
        df = _time_operation("HandEval: Add Suit Lengths", self._add_suit_length_columns, df)
        df = _time_operation("HandEval: Add Suit Length Arrays", self._add_suit_length_array_columns, df)
        df = _time_operation("HandEval: Add Distribution Points", self._add_distribution_point_columns, df)
        df = _time_operation("HandEval: Add Total Points", self._add_total_point_columns, df)
        # LoTT depends on DD columns from DDSdAugmenter, so it's called later by the orchestrator if needed.
        df = _time_operation("HandEval: Add Suit Quality/Stoppers", self._add_suit_quality_stopper_columns, df)
        df = _time_operation("HandEval: Add Balanced Indicators", self._add_balanced_hand_columns, df)
        return df

class DDSdAugmenter:
    """Double Dummy (DD) and Single Dummy (SD) calculations, Par scores, and Expected Values (EV)."""
    def __init__(self, hrs_d=None, progress_callback=None):
        self.hrs_d = hrs_d if hrs_d is not None else {} # Hand Results Store (cache)
        self.progress_callback = progress_callback # For long operations like SD

        # Calculate score tables once
        self.all_scores_d, self.scores_d, self.scores_df = calculate_contract_score_tables()
        self.scores_df_vuls = self._create_scores_df_with_vul_columns(self.scores_df)


    def _update_progress(self, current, total, message):
        if self.progress_callback:
            self.progress_callback.progress(current / total, text=f"{message} - {current}/{total} ({current*100/total:.0f}%)")
        elif total > 0 : # Basic console progress
            if current % (total // 10 if total > 10 else 1) == 0 or current == total:
                 print(f"PROGRESS: {message} - {current}/{total} ({current*100/total:.0f}%)")

    def _create_scores_df_with_vul_columns(self, scores_df: pl.DataFrame) -> pl.DataFrame:
        """
        Expands the scores_df to have separate columns for non-vulnerable and vulnerable scores
        for each trick count.
        Input: scores_df (from calculate_contract_score_tables)
               Each cell in 'Score_[L][S]' is a list [NV_score, V_score]
        Output: DataFrame with columns like 'Score_[L][S]_NV' and 'Score_[L][S]_V'
        """
        if not scores_df.is_empty() and isinstance(scores_df.select(pl.first()).item(0,0), list): # check if it needs expansion
            exploded_columns = []
            for col_name in scores_df.columns:
                if col_name.startswith("Score_"):
                    # Each element in these columns is a list [non_vul_score, vul_score]
                    # We need to transform rows of lists into rows of individual scores for NV and V
                    # This requires accessing list elements. If scores_df has 14 rows (for tricks 0-13):
                    exploded_columns.append(pl.col(col_name).list.get(0).alias(f"{col_name}_NV"))
                    exploded_columns.append(pl.col(col_name).list.get(1).alias(f"{col_name}_V"))

            # Original scores_df has columns 'Score_1C', 'Score_1D', ...
            # Each of these columns has 14 elements (lists).
            # Applying with_columns directly will try to operate on these list-columns.
            # We need to ensure that the main df that this is later joined to has the same number of rows (14).
            # This function is more about preparing the scores_df_vuls for the sd_expected_values calculation,
            # which itself processes a main df row by row, and for each row, iterates 14 times (tricks).

            # The `scores_df` has 14 rows. The `df_scores.with_columns(exploded_columns)` will also have 14 rows.
            # Original code returns `df_scores.with_columns(exploded_columns).drop(df_scores.columns)`
            # This means it only returns the new _NV and _V columns.
            df_scores_expanded = scores_df.with_columns(exploded_columns)
            return df_scores_expanded.select([col for col in df_scores_expanded.columns if '_NV' in col or '_V' in col])
        return scores_df # Already expanded or empty

    # Inputs: df columns ['PBN', 'Dealer', 'Vul']
    # Outputs: df columns related to DD tricks, Par scores, and DD scores.
    #          ['DD_N_S', ..., 'DD_EW_N'], ['Par_NS', 'Par_EW', 'ParContract'],
    #          ['DD_Score_1C_N', ..., 'DD_Score_7N_W']
    # Description: Calculates double dummy results, par scores, and associated DD-based scores.
    def _add_dd_par_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['PBN', 'Dealer', 'Vul'], "DDSdAugmenter._add_dd_par_columns")

        unique_pbns_series = df.select('PBN').to_series().unique(maintain_order=True)

        # Filter PBNs that need calculation (not in cache self.hrs_d or missing 'DD' entry)
        pbns_to_calculate = []
        for pbn_val in unique_pbns_series:
            if pbn_val not in self.hrs_d or 'DD' not in self.hrs_d[pbn_val]:
                pbns_to_calculate.append(pbn_val)

        deals_to_calc = [Deal(pbn_str) for pbn_str in pbns_to_calculate]

        if deals_to_calc:
            #dds = _init_dds() # If DDS needs explicit init/close, manage here
            self._update_progress(0, len(deals_to_calc), "Calculating DD tables")
            #calc_all_tables can take a progress callback itself if library supports it.
            #Simulate batching if calc_all_tables is slow on very large inputs:
            batch_size = 40 # 40 is max for calc_all_tables
            all_calculated_dd_tables = []
            for i in range(0, len(deals_to_calc), batch_size):
                batch_deals = deals_to_calc[i:i+batch_size]
                # result_tables = calc_all_tables(batch_deals, progress=lambda curr,tot: self._update_progress(i+curr, len(deals_to_calc), "Calculating DD tables"))
                # Assuming calc_all_tables doesn't have a progress callback for now.
                result_tables = calc_all_tables(batch_deals)
                all_calculated_dd_tables.extend(result_tables)
                self._update_progress(min(i+batch_size, len(deals_to_calc)), len(deals_to_calc), "Calculating DD tables")

            for deal_obj, rt_obj in zip(deals_to_calc, all_calculated_dd_tables):
                pbn_str = deal_obj.to_pbn()
                if pbn_str not in self.hrs_d:
                    self.hrs_d[pbn_str] = {}
                self.hrs_d[pbn_str]['DD'] = rt_obj

        # Prepare data for DataFrame construction
        par_scores_ns_list = []
        par_scores_ew_list = []
        par_contracts_list = []
        flattened_dd_rows_list = []

        vul_map_ep = {'None': Vul.none, 'N_S': Vul.ns, 'E_W': Vul.ew, 'Both': Vul.both}
        player_map_ep = {'N': Player.north, 'E': Player.east, 'S': Player.south, 'W': Player.west}

        for row_tuple in df.select(['PBN', 'Dealer', 'Vul']).iter_rows():
            pbn, dealer_str, vul_str = row_tuple
            dd_table = self.hrs_d[pbn]['DD']

            # Par calculation
            # Cache par results per (pbn, dealer, vul)
            par_cache_key = (dealer_str, vul_str)
            if 'Par' not in self.hrs_d[pbn] or par_cache_key not in self.hrs_d[pbn]['Par']:
                par_result = par(dd_table, vul_map_ep[vul_str], player_map_ep[dealer_str])
                if 'Par' not in self.hrs_d[pbn]: self.hrs_d[pbn]['Par'] = {}
                self.hrs_d[pbn]['Par'][par_cache_key] = par_result
            else:
                par_result = self.hrs_d[pbn]['Par'][par_cache_key]

            par_scores_ns_list.append(par_result.score)
            par_scores_ew_list.append(-par_result.score)
            par_contracts_list.append([ # List of one string, as in original
                ', '.join([
                    str(c.level) + SHDCN[c.denom.value] + c.declarer.abbr + c.penalty.abbr +
                    ('' if c.result == 0 else f'+{c.result}' if c.result > 0 else str(c.result))
                    for c in par_result
                ])
            ])

            # DD Table flattening: N(S,H,D,C,N), E(S,H,D,C,N), S(...), W(...)
            # dd_table.to_list() gives [[N_S, N_H, ...], [E_S, E_H, ...], ...]
            # Original used zip(*rt.to_list()) to transpose, then flatten.
            # This results in N_S, E_S, S_S, W_S, N_H, E_H, S_H, W_H, ...
            # The schema then is DD_N_S, DD_E_S, DD_S_S, DD_W_S, DD_N_H, ...
            # It seems the schema order should be DD_N_S, DD_N_H, ... DD_E_S, ...
            # Let's match original: columns = {f'DD_{direction}_{suit}':pl.UInt8 for direction in 'NESW' for suit in 'SHDCN'}
            # This order is N_S, N_H, N_D, N_C, N_N, E_S, E_H, ...
            # So, rt.to_list() which is already [N_tricks_by_suit, E_tricks_by_suit, ...] then flatten.
            flat_row = [trick for hand_tricks in dd_table.to_list() for trick in hand_tricks]
            flattened_dd_rows_list.append(flat_row)

        # Create DD_Tricks_df
        dd_trick_cols_schema = {f'DD_{d}_{s}':pl.UInt8 for d in NESW for s in SHDCN}
        dd_tricks_df = pl.DataFrame(flattened_dd_rows_list, schema=dd_trick_cols_schema, orient='row')

        # Add partnership DD trick columns
        dd_pair_trick_exprs = []
        for pair_short in NS_EW: # 'NS', 'EW'
            p1, p2 = pair_short[0], pair_short[1]
            for s_char in SHDCN:
                dd_pair_trick_exprs.append(
                    pl.max_horizontal(f"DD_{p1}_{s_char}", f"DD_{p2}_{s_char}").alias(f"DD_{pair_short}_{s_char}")
                )
        dd_tricks_df = dd_tricks_df.with_columns(dd_pair_trick_exprs)

        # Create Par_df
        par_df = pl.DataFrame({
            'Par_NS': par_scores_ns_list,
            'Par_EW': par_scores_ew_list,
            'ParContract': par_contracts_list # This will be a list column
        })

        # Create DD_Score_df (scores for making 1 to 7 of a suit by each player, based on DD tricks)
        # This depends on self.scores_d and the main df's 'Vul' column
        # The original dd_score_cols structure was complex. It creates columns like 'DD_Score_1C_N', etc.
        # The value is the score if player N makes 1C with the DD-calculated tricks for N in Clubs.
        dd_score_cols_data = [] # Each element will be a list of scores for a new column
        dd_score_col_names = []

        # Iterate through each potential contract (level, suit, declarer)
        for level in range(1, 8):
            for s_char in SHDCN:
                for d_char in NESW: # Declarer
                    col_name = f'DD_Score_{level}{s_char}_{d_char}'
                    dd_score_col_names.append(col_name)

                    current_col_scores = []
                    # For each row in the main DataFrame (deal)
                    for i in range(df.height):
                        # Get the DD tricks for this declarer and suit for this deal
                        # dd_tricks_df[i, f'DD_{d_char}_{s_char}'] doesn't work directly like pandas
                        tricks_for_contract = dd_tricks_df.row(i, named=True)[f'DD_{d_char}_{s_char}']

                        # Determine vulnerability of declarer d_char for this deal
                        vul_str_for_deal = df.row(i, named=True)['Vul']
                        is_vul = (vul_str_for_deal == 'Both') or \
                                 (vul_str_for_deal == 'N_S' and d_char in ('N', 'S')) or \
                                 (vul_str_for_deal == 'E_W' and d_char in ('E', 'W'))

                        # Get score from self.scores_d
                        score_val = self.scores_d.get((level, s_char, tricks_for_contract, is_vul), 0)
                        current_col_scores.append(score_val)
                    dd_score_cols_data.append(pl.Series(name=col_name, values=current_col_scores, dtype=pl.Int16))

        dd_score_df = pl.DataFrame(dd_score_cols_data)

        # Concatenate new DataFrames to the original df
        # Ensure row counts match. dd_tricks_df, par_df, dd_score_df should have same height as df.
        df = pl.concat([df, dd_tricks_df, par_df, dd_score_df], how='horizontal')
        return df

    # Inputs: df column ['PBN']
    # Outputs: df columns related to SD probabilities ['Probs_NS_N_S_0', ..., 'Probs_EW_W_N_13']
    # Description: Calculates single dummy probabilities for each hand to take tricks.
    def _add_sd_probability_columns(self, df: pl.DataFrame, sd_productions: int = 100) -> pl.DataFrame:
        _assert_columns_exist(df, ['PBN'], "DDSdAugmenter._add_sd_probability_columns")

        unique_pbns_series = df.select('PBN').to_series().unique(maintain_order=True)

        pbns_to_calculate_sd = []
        for pbn_val in unique_pbns_series:
            if pbn_val not in self.hrs_d or 'SD' not in self.hrs_d[pbn_val] or \
               (pbn_val in self.hrs_d and self.hrs_d[pbn_val].get('SD', (0,))[0] != sd_productions): # Check if productions match
                pbns_to_calculate_sd.append(pbn_val)

        self._update_progress(0, len(pbns_to_calculate_sd), f"Calculating SD probabilities ({sd_productions} samples)")
        for i, pbn_str in enumerate(pbns_to_calculate_sd):
            # Original calculate_single_dummy_probabilities logic:
            # For a given PBN, it simulates holdings for hidden hands and gets DD results for each simulation.
            # This is computationally intensive.

            # Simplified version of original logic's caching and calculation:
            # Deal object from PBN string
            deal_obj = Deal(pbn_str)
            sd_results_for_pbn = {} # To store {(pair_dir, decl_dir, suit): [probs_0_tricks, ..., probs_13_tricks]}

            for pair_direction_declaring in NS_EW: # 'NS' or 'EW' (pair whose hands are known)
                # Reconstruct PBN for endplay.dealer.generate_deals predeal
                # generate_deals expects PBN format like "N:S.H.D.C S.H.D.C S.H.D.C S.H.D.C"
                # The Deal(pbn_string) constructor is more flexible.
                # We need to pass the known hands to generate_deals.
                predeal_obj = Deal() # Start with an empty deal
                known_hands_for_predeal = {}
                # original_hands = deal_obj.deal # This line will be removed by the logic below
                if pair_direction_declaring == 'NS':
                    known_hands_for_predeal[Player.north] = deal_obj[Player.north]
                    known_hands_for_predeal[Player.south] = deal_obj[Player.south]
                else: # EW known, NS unknown
                    known_hands_for_predeal[Player.east] = deal_obj[Player.east]
                    known_hands_for_predeal[Player.west] = deal_obj[Player.west]

                predeal_obj.hands = known_hands_for_predeal # Set known hands

                # constraints function (allow all valid deals for now)
                def any_deal_constraints(deal_to_check): return True

                # Generate deals with the known hands fixed
                generated_deals_tuples = generate_deals(
                    any_deal_constraints,
                    predeal=predeal_obj,
                    produce=sd_productions,
                    show_progress=False # Use outer progress
                )
                generated_deals_list = list(generated_deals_tuples)

                if not generated_deals_list: continue # Should not happen with any_deal_constraints

                # Calculate DD tables for all generated deals
                # todo: add progress for this sub-step if needed
                sim_dd_tables = calc_all_tables(generated_deals_list) # Returns list of DDTable objects

                # Aggregate results:
                # For each declarer (N,E,S,W) and each suit (S,H,D,C,N), count how many tricks are taken
                # across the sd_productions simulations.

                # This part of original code was:
                # SD_Tricks_df[ns_ew] = pl.DataFrame([... schema ...])
                # ns_ew_rows[(ns_ew,d,s)] = ... .value_counts(normalize=True) ...
                # This means for each (pair_direction_declaring, actual_declarer, suit_denom)
                # we get a distribution of tricks.

                # For each declarer direction d_char ('N', 'E', 'S', 'W')
                # And for each suit s_char ('S', 'H', 'D', 'C', 'N')
                # Accumulate trick counts from sim_dd_tables
                for d_char_enum in Player: # N, E, S, W
                    d_char = d_char_enum.abbr
                    for s_idx, s_char_denom in enumerate(Denom): # S, H, D, C, NT
                        s_char = SHDCN[s_idx]

                        trick_counts_for_this_setup = [0] * 14 # 0 to 13 tricks
                        for dd_table_sim in sim_dd_tables:
                            # dd_table_sim.to_list() is [[N_S, N_H..], [E_S, E_H..], ..]
                            # Player enum N=0, E=1, S=2, W=3. Denom enum S=0, H=1, D=2, C=3, NT=4
                            tricks = dd_table_sim[d_char_enum, s_char_denom]
                            if 0 <= tricks <= 13:
                                trick_counts_for_this_setup[tricks] += 1

                        # Normalize to probabilities
                        probs = [count / sd_productions for count in trick_counts_for_this_setup]
                        sd_results_for_pbn[(pair_direction_declaring, d_char, s_char)] = probs

            # Cache the result for this PBN
            if pbn_str not in self.hrs_d: self.hrs_d[pbn_str] = {}
            self.hrs_d[pbn_str]['SD'] = (sd_productions, sd_results_for_pbn)
            self._update_progress(i + 1, len(pbns_to_calculate_sd), f"Calculating SD probabilities ({sd_productions} samples)")

        # Construct DataFrame columns from cached SD probabilities
        sd_probs_cols_data = defaultdict(list)
        # Column names like 'Probs_NS_N_S_0' (PairKnown_Declarer_Suit_Tricks)
        for pbn_str_from_df in df['PBN']: # Iterate through original df's PBNs to maintain order
            _sd_prod, sd_data_dict = self.hrs_d[pbn_str_from_df]['SD']
            for (pair_dir, decl_dir, suit_char), prob_list in sd_data_dict.items():
                for i_trick, prob_val in enumerate(prob_list):
                    col_name = f'Probs_{pair_dir}_{decl_dir}_{suit_char}_{i_trick}'
                    sd_probs_cols_data[col_name].append(prob_val)

        sd_probs_df = pl.DataFrame(dict(sd_probs_cols_data)) # Convert defaultdict to dict for Polars
        df = pl.concat([df, sd_probs_df], how='horizontal')
        return df

    # Inputs: df columns ['PBN'], 'Probs_*' (from _add_sd_probability_columns), and uses self.scores_df_vuls
    # Outputs: df columns ['EV_NS_N_S_1_NV', ..., 'EV_EW_W_N_7_V'] (Expected Values for contracts)
    #          and ['EV_NS_N_S_1_NV_Max', etc.] (Max EV over levels for a given declarer/suit/vul)
    # Description: Calculates expected values for all possible contracts based on SD probabilities.
    def _add_sd_expected_value_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['PBN'], "DDSdAugmenter._add_sd_expected_value_columns")
        # Check if a sample Probs col exists
        # Example: Probs_NS_N_S_0 needs to exist
        if not any(col.startswith("Probs_") for col in df.columns):
             raise ValueError("Missing SD Probability columns (Probs_*) for EV calculation.")

        # self.scores_df_vuls should be pre-calculated (14 rows, cols like Score_1C_NV, Score_1C_V)

        ev_col_expressions = [] # For individual EV_[PD]_[DD]_[S]_[L]_[V]

        # These loops define each specific contract for which we calculate EV
        for pair_dir_known in NS_EW:      # Whose hands are known for SD probs ('NS', 'EW')
            for decl_dir_actual in NESW:  # The actual declarer ('N','E','S','W')
                for suit_char_contract in SHDCN: # Strain of the contract
                    for level_contract in range(1, 8): # Level of the contract (1-7)
                        for vul_state_char in ['NV', 'V']: # Vulnerability state ('NV', 'V')

                            # This will be one EV column, e.g., EV_NS_N_S_1_NV
                            # It's the EV if NS hands are known, N declares 1S Not Vulnerable.
                            ev_col_name = f'EV_{pair_dir_known}_{decl_dir_actual}_{suit_char_contract}_{level_contract}_{vul_state_char}'

                            # Sum over tricks_taken (0 to 13): Probs_..._trick * Score_..._trick
                            sum_expr_for_ev = None

                            # Get the score column from self.scores_df_vuls for this contract (level, suit, vul_state)
                            # e.g. 'Score_1S_NV'
                            score_col_for_contract_in_scores_df = f'Score_{level_contract}{suit_char_contract}_{vul_state_char}'

                            if score_col_for_contract_in_scores_df not in self.scores_df_vuls.columns:
                                # This should not happen if scores_df_vuls is correctly generated
                                print(f"Warning: Score column {score_col_for_contract_in_scores_df} not found in scores_df_vuls. Skipping EV for {ev_col_name}")
                                continue

                            for tricks_taken_idx in range(14): # 0 to 13 tricks
                                prob_col_name = f'Probs_{pair_dir_known}_{decl_dir_actual}_{suit_char_contract}_{tricks_taken_idx}'

                                if prob_col_name not in df.columns:
                                    # This implies SD probs were not calculated for this specific combination
                                    # Or, if decl_dir_actual is not in pair_dir_known, these probs might not be relevant/exist.
                                    # Original code iterated `for declarer_direction in pair_direction` in `calculate_sd_expected_values`
                                    # This means decl_dir_actual MUST be one of the pair_dir_known.
                                    if decl_dir_actual not in pair_dir_known:
                                        # This combination is not calculated by original logic, skip.
                                        # Example: if pair_dir_known is 'NS', decl_dir_actual can only be 'N' or 'S'.
                                        continue
                                    else: # Should exist if SD ran correctly
                                        print(f"Warning: Prob column {prob_col_name} not found in df. Skipping term for {ev_col_name}")
                                        continue


                                # Get the score for taking `tricks_taken_idx` tricks in this contract
                                # self.scores_df_vuls has 14 rows. We need the score from the row matching tricks_taken_idx.
                                # Score for making `level_contract` `suit_char_contract` when `tricks_taken_idx` tricks are made.
                                # The `self.scores_df_vuls[score_col_for_contract_in_scores_df]` is already a Series of 14 scores for that contract.
                                # We need the score AT `tricks_taken_idx`.

                                # Let's use the `scores_d` dict approach from original `calculate_sd_expected_values`
                                # `scores_d[(level,suit_char,tricks,is_vul)]`
                                # is_vul_bool = (vul_state_char == 'V')
                                # score_val_lit = pl.lit(self.scores_d.get((level_contract, suit_char_contract, tricks_taken_idx, is_vul_bool),0))

                                # Using pre-extracted scores from scores_df_vuls (which has 14 rows)
                                # score_series_for_contract = self.scores_df_vuls[score_col_for_contract_in_scores_df]
                                # score_lit_for_trick = score_series_for_contract[tricks_taken_idx] # This is a Python scalar
                                # This needs to be done carefully if scores_df_vuls is directly used in `with_columns`
                                # The multiplication `pl.col(prob_col_name).mul(score_lit_for_trick)` works.

                                # Python list of scores for the current contract (level, suit, vul_state)
                                current_contract_scores_list = self.scores_df_vuls[score_col_for_contract_in_scores_df].to_list()

                                term = pl.col(prob_col_name) * pl.lit(current_contract_scores_list[tricks_taken_idx], dtype=pl.Float32)

                                if sum_expr_for_ev is None:
                                    sum_expr_for_ev = term
                                else:
                                    sum_expr_for_ev = sum_expr_for_ev + term

                            if sum_expr_for_ev is not None:
                                ev_col_expressions.append(sum_expr_for_ev.alias(ev_col_name))

        if not ev_col_expressions:
            print("Warning: No EV column expressions were generated. Check loops and conditions in _add_sd_expected_value_columns.")
            return df

        df = df.with_columns(ev_col_expressions)

        # Now, create Max EV columns (max over levels for a given declarer/suit/vul_state)
        # Example: EV_NS_N_S_V_Max = max(EV_NS_N_S_1_V, EV_NS_N_S_2_V, ..., EV_NS_N_S_7_V)
        # And EV_NS_N_S_V_Max_Col = name of the column that had the max value.
        max_ev_expressions = []
        for pair_dir_known in NS_EW:
            for decl_dir_actual in NESW:
                 if decl_dir_actual not in pair_dir_known: continue # Match original logic constraint

                 for suit_char_contract in SHDCN:
                    for vul_state_char in ['NV', 'V']:
                        cols_to_max_over = [
                            f'EV_{pair_dir_known}_{decl_dir_actual}_{suit_char_contract}_{lvl}_{vul_state_char}'
                            for lvl in range(1, 8)
                            if f'EV_{pair_dir_known}_{decl_dir_actual}_{suit_char_contract}_{lvl}_{vul_state_char}' in df.columns # Ensure col exists
                        ]
                        if not cols_to_max_over: continue

                        max_val_expr = pl.max_horizontal(cols_to_max_over)
                        max_ev_expressions.append(max_val_expr.alias(f'EV_{pair_dir_known}_{decl_dir_actual}_{suit_char_contract}_{vul_state_char}_Max'))

                        # Max Col Name
                        current_max_col_expr = None
                        for col_name_in_max_group in cols_to_max_over:
                            condition = (pl.col(col_name_in_max_group) == max_val_expr)
                            if current_max_col_expr is None:
                                current_max_col_expr = pl.when(condition).then(pl.lit(col_name_in_max_group))
                            else:
                                current_max_col_expr = current_max_col_expr.when(condition).then(pl.lit(col_name_in_max_group))

                        if current_max_col_expr is not None:
                             max_ev_expressions.append(
                                 current_max_col_expr.otherwise(pl.lit(None)) # Or empty string
                                 .alias(f'EV_{pair_dir_known}_{decl_dir_actual}_{suit_char_contract}_{vul_state_char}_Max_Col')
                             )

        df = df.with_columns(max_ev_expressions)
        return df

    # Inputs: df columns from _add_sd_expected_value_columns (EV_*_Max), and 'Vul' (from DealAugmenter)
    # Outputs: df columns like ['EV_Max_NS', 'EV_Max_Col_NS', ...] (final EV choices based on board's Vul)
    # Description: Selects the best EV options based on the board's specific vulnerability.
    def _add_final_ev_choice_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['Vul'], "DDSdAugmenter._add_final_ev_choice_columns")

        # Removed the 'if not any(col.endswith("_Max") ...)' block that caused early return.
        # The function will now always attempt to generate final EV columns.
        # If precursor columns (e.g. EV_NS_N_S_V_Max) are missing from the previous step (_add_sd_expected_value_columns),
        # then the intermediate EV columns in this function might be sparse or empty,
        # leading to EV_Max_NS/EW being null, which is acceptable for ScoreAugmenter.

        final_ev_expressions = []
        # These vul_conditions are relative to the board's vulnerability, not the hypothetical vul_state of contract
        vul_cond_map = {
            'NS': pl.col('Vul').is_in(['N_S', 'Both']), # NS is vulnerable on this board
            'EW': pl.col('Vul').is_in(['E_W', 'Both'])  # EW is vulnerable on this board
        }

        for pair_dir_known in NS_EW: # This is about SD context usually (whose hands are known)
                                    # For final EV, we care about which pair is declaring.
                                    # The EV columns are EV_[PairWhoseHandsWereKnownForSD]_[ActualDeclarer]...
                                    # Let's assume for final choice, we pick based on actual declarer's partnership.
                                    # This part of HandAugmenter original was `_create_ev_columns` which used `pd` (pair direction)
                                    # as the primary loop, which seems to mean the "current pair".

            # Level: Max EV for the pair (NS or EW), considering board's Vul
            # We need EV_NS_N_S_V_Max, EV_NS_N_S_NV_Max, etc.
            # The EV columns for a pair (e.g. NS) would be max over declarers (N,S) and suits and levels.
            # This was handled by original `create_best_contracts` which created `EV_V_Max`, `EV_NV_Max` etc.
            # Let's refine `create_best_contracts` (now part of `_add_sd_expected_value_columns`'s latter half)
            # to make overall bests (EV_Overall_V_Max, EV_Overall_NV_Max).

            # The original `create_best_contracts` created hierarchical maxes.
            # Let's reproduce that structure for selecting based on board Vul.
            # Top Level: Overall best EV for the board (max of NS_V, NS_NV, EW_V, EW_NV chosen by Vul)

            # Loop for Pair (NS, EW) - this is the "declaring pair" context for final EV columns
            for current_pair_declaring in NS_EW:
                is_vulnerable_expr = vul_cond_map[current_pair_declaring]

                # Max EV for this pair, considering board Vul
                # Needs EV_[Pair]_V_Max and EV_[Pair]_NV_Max (these are max over declarers in pair, suits, levels)
                # These need to be created by `_add_sd_expected_value_columns` first.
                # Example: EV_NS_V_Max would be max over (N,S declarers), (S,H,D,C,N suits), (1-7 levels) for Vul contracts.
                # This requires another layer of aggregation in _add_sd_expected_value_columns or a new method.

                # For now, let's assume `EV_[Pair]_[Decl]_[Suit]_[VulState]_Max` columns exist.
                # The original `_create_ev_columns` in `HandAugmenter` built up from these.

                # Final EV for each (Declarer, Suit, Level) based on board Vul
                for decl_dir_actual in current_pair_declaring: # N,S if current_pair_declaring is NS
                    for suit_char_contract in SHDCN:
                        for level_contract in range(1, 8):
                            base_ev_col_name_v = f'EV_{current_pair_declaring}_{decl_dir_actual}_{suit_char_contract}_{level_contract}_V'
                            base_ev_col_name_nv = f'EV_{current_pair_declaring}_{decl_dir_actual}_{suit_char_contract}_{level_contract}_NV'

                            if base_ev_col_name_v in df.columns and base_ev_col_name_nv in df.columns:
                                final_ev_expressions.append(
                                    pl.when(is_vulnerable_expr).then(pl.col(base_ev_col_name_v))
                                    .otherwise(pl.col(base_ev_col_name_nv))
                                    .alias(f'EV_{current_pair_declaring}_{decl_dir_actual}_{suit_char_contract}_{level_contract}') # Final chosen EV
                                )

                # Max EV for each (Declarer, Suit) based on board Vul (max over levels)
                for decl_dir_actual in current_pair_declaring:
                    for suit_char_contract in SHDCN:
                        base_max_col_v = f'EV_{current_pair_declaring}_{decl_dir_actual}_{suit_char_contract}_V_Max'
                        base_max_col_nv = f'EV_{current_pair_declaring}_{decl_dir_actual}_{suit_char_contract}_NV_Max'
                        base_max_name_col_v = f'EV_{current_pair_declaring}_{decl_dir_actual}_{suit_char_contract}_V_Max_Col'
                        base_max_name_col_nv = f'EV_{current_pair_declaring}_{decl_dir_actual}_{suit_char_contract}_NV_Max_Col'

                        if all(c in df.columns for c in [base_max_col_v, base_max_col_nv, base_max_name_col_v, base_max_name_col_nv]):
                            final_ev_expressions.append(
                                pl.when(is_vulnerable_expr).then(pl.col(base_max_col_v))
                                .otherwise(pl.col(base_max_col_nv))
                                .alias(f'EV_{current_pair_declaring}_{decl_dir_actual}_{suit_char_contract}_Max')
                            )
                            final_ev_expressions.append(
                                pl.when(is_vulnerable_expr).then(pl.col(base_max_name_col_v))
                                .otherwise(pl.col(base_max_name_col_nv))
                                .alias(f'EV_{current_pair_declaring}_{decl_dir_actual}_{suit_char_contract}_Max_Col')
                            )

                # Max EV for each Declarer (max over suits, levels)
                # Max EV for the Pair (max over declarers, suits, levels)
                # These require further aggregation from the EV_[Pair]_[Decl]_[Suit]_Max columns.
                # Example for EV_[Pair]_Max:
                cols_for_pair_max = [
                    f'EV_{current_pair_declaring}_{dd}_{sc}_Max'
                    for dd in current_pair_declaring for sc in SHDCN
                ]
                # Filter to columns that were actually planned for creation in earlier loops of THIS function
                # or already exist in the DataFrame.
                
                # Collect names of columns that final_ev_expressions is attempting to create,
                # which match the pattern EV_[Pair]_[Decl]_[Suit]_Max
                prospective_cols_from_expressions = []
                for expr in final_ev_expressions:
                    expr_name = expr.meta.output_name()
                    if expr_name.startswith(f'EV_{current_pair_declaring}_') and expr_name.endswith('_Max') and expr_name.count('_') == 4:
                         prospective_cols_from_expressions.append(expr_name)
                
                # Combine with columns already in df that match the pattern (less likely for new computations)
                cols_existing_in_df_matching_pattern = [
                    col for col in df.columns 
                    if col.startswith(f'EV_{current_pair_declaring}_') and col.endswith('_Max') and col.count('_') == 4
                       and col not in prospective_cols_from_expressions # Avoid duplicates
                ]

                cols_that_will_exist_for_pair_max = prospective_cols_from_expressions + cols_existing_in_df_matching_pattern

                if cols_that_will_exist_for_pair_max: 
                     final_ev_expressions.append(
                         pl.max_horizontal(cols_that_will_exist_for_pair_max).alias(f'EV_Max_{current_pair_declaring}')
                     )
                else:
                    # Ensure EV_Max_NS/EW is added as an expression for null if no constituents were found
                    final_ev_expressions.append(
                        pl.lit(None, dtype=pl.Float32).alias(f'EV_Max_{current_pair_declaring}')
                    )
        
        # Apply all expressions generated so far. This will create EV_Max_NS and EV_Max_EW (either calculated or null).
        if final_ev_expressions:
            df = df.with_columns(final_ev_expressions)
        
        # Fallback: if EV_Max_NS or EV_Max_EW are still not in df.columns after applying expressions, add them as null.
        # This provides an additional layer of safety.
        cols_to_ensure_after_apply = {}
        if 'EV_Max_NS' not in df.columns:
            print(f"Fallback: EV_Max_NS still not in columns. Adding as null.")
            cols_to_ensure_after_apply['EV_Max_NS'] = pl.lit(None, dtype=pl.Float32)
        if 'EV_Max_EW' not in df.columns:
            print(f"Fallback: EV_Max_EW still not in columns. Adding as null.")
            cols_to_ensure_after_apply['EV_Max_EW'] = pl.lit(None, dtype=pl.Float32)
        
        if cols_to_ensure_after_apply:
            df = df.with_columns(**cols_to_ensure_after_apply)
            
        # Ensure EV_Max_Board exists, using the (now guaranteed to exist) EV_Max_NS and EV_Max_EW
        if 'EV_Max_NS' in df.columns and 'EV_Max_EW' in df.columns:
             df = df.with_columns(pl.max_horizontal('EV_Max_NS', 'EV_Max_EW').alias('EV_Max_Board'))
        elif 'EV_Max_Board' not in df.columns: # Should not be strictly necessary if above is perfect
             df = df.with_columns(pl.lit(None, dtype=pl.Float32).alias('EV_Max_Board'))
             
        return df

    # Inputs: df columns ['DD_N_S', ..., 'DD_W_N'] (DD tricks for each hand and suit)
    # Outputs: df columns ['CT_N_S', ..., 'CT_W_N_GSlam'] (Contract Type from DD perspective, and booleans)
    # Description: Classifies potential contract types based on DD results.
    def _add_dd_based_contract_type_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, [f'DD_{d}_{s}' for d in NESW for s in SHDCN], "DDSdAugmenter._add_dd_based_contract_type_columns")

        if 'CT_N_S' not in df.columns: # Check one
            ct_expressions = []
            for direction in NESW:
                for strain_char in SHDCN: # Contract strain
                    dd_tricks_col = f"DD_{direction}_{strain_char}" # DD tricks for this declarer & suit

                    # Contract type classification based on DD tricks
                    # This logic is from original ResultAugmenter._create_contract_types
                    ct_expr = (
                        pl.when(pl.col(dd_tricks_col) < 7).then(pl.lit("Pass")) # Less than book
                        .when((pl.col(dd_tricks_col) == 11) & (strain_char in ['C', 'D'])).then(pl.lit("Game")) # 5m needs 11
                        .when((pl.col(dd_tricks_col) >= 10) & (strain_char in ['H', 'S']) & (pl.col(dd_tricks_col) <=11 )).then(pl.lit("Game")) # 4M needs 10/11
                        .when((pl.col(dd_tricks_col) >= 9) & (strain_char == 'N') & (pl.col(dd_tricks_col) <=11 )).then(pl.lit("Game")) # 3NT needs 9/10/11
                        .when(pl.col(dd_tricks_col) == 12).then(pl.lit("SSlam")) # Small slam
                        .when(pl.col(dd_tricks_col) == 13).then(pl.lit("GSlam")) # Grand slam
                        .otherwise(pl.lit("Partial")) # Made but not game/slam
                        .alias(f"CT_{direction}_{strain_char}")
                    )
                    ct_expressions.append(ct_expr)
            df = df.with_columns(ct_expressions)

        # Add boolean columns for each contract type
        if 'CT_N_S_Game' not in df.columns: # Check one
            ct_bool_expressions = []
            for direction in NESW:
                for strain_char in SHDCN:
                    for ct_category in ["Pass", "Game", "SSlam", "GSlam", "Partial"]:
                        ct_bool_expressions.append(
                            (pl.col(f"CT_{direction}_{strain_char}") == pl.lit(ct_category))
                            .alias(f"CT_{direction}_{strain_char}_{ct_category}")
                        )
            df = df.with_columns(ct_bool_expressions)
        return df


    def perform_augmentations(self, df: pl.DataFrame, sd_productions: int = 40) -> pl.DataFrame:
        """Applies all DD and SD related augmentations."""
        df = _time_operation("DDSd: Add DD & Par Columns", self._add_dd_par_columns, df)
        df = _time_operation("DDSd: Add SD Probability Columns", self._add_sd_probability_columns, df, sd_productions=sd_productions)
        df = _time_operation("DDSd: Add SD Expected Value Columns", self._add_sd_expected_value_columns, df)
        df = _time_operation("DDSd: Add Final EV Choice Columns", self._add_final_ev_choice_columns, df)
        df = _time_operation("DDSd: Add DD-based Contract Types", self._add_dd_based_contract_type_columns, df)
        # Other methods like LoTT from HandEvaluationAugmenter might need to be called *after* some DD columns are made.
        # The orchestrator should handle this.
        return df

class ContractAugmenter:
    """Processing of the actual played contract (Declarer, Bid, links to DD/SD data)."""
    def __init__(self, all_scores_d): # Needs all_scores_d for score computation link
        self.declarer_to_LHO_d = {None: None, 'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
        self.declarer_to_dummy_d = {None: None, 'N': 'S', 'E': 'W', 'S': 'N', 'W': 'E'}
        self.declarer_to_RHO_d = {None: None, 'N': 'W', 'E': 'N', 'S': 'E', 'W': 'S'}
        self.all_scores_d = all_scores_d # For linking to computed scores

    # Inputs: df column ['Contract'] (original contract string)
    # Outputs: df column ['Contract_std'] (standardized contract string)
    # Description: Standardizes the contract string format.
    def _standardize_contract_string(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['Contract'], "ContractAugmenter._standardize_contract_string")
        if 'Contract_std' not in df.columns: # Assuming new column for standardized
             df = df.with_columns(
                pl.col('Contract').str.to_uppercase()
                .str.replace_all('♠', 'S').str.replace_all('♥', 'H')
                .str.replace_all('♦', 'D').str.replace_all('♣', 'C')
                .str.replace_all('NT', 'N')
                .alias('Contract_std')
            )
        # If overwriting 'Contract':
        # df = df.with_columns(pl.col('Contract').str.to_uppercase()... .alias('Contract'))
        return df

    # Inputs: df column ['Contract_std']
    # Outputs: df columns ['Declarer_Direction', 'BidLvl', 'BidSuit', 'Dbl']
    # Description: Extracts components from the standardized contract string.
    def _extract_contract_components(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['Contract_std'], "ContractAugmenter._extract_contract_components")

        # Define UDFs to parse contract string components safely
        def get_bid_level(contract_str_series: pl.Series) -> pl.Series:
            return contract_str_series.map_elements(lambda cs: int(cs[0]) if cs and cs != 'PASS' and cs[0].isdigit() else None, return_dtype=pl.UInt8)

        def get_bid_suit(contract_str_series: pl.Series) -> pl.Series:
            return contract_str_series.map_elements(lambda cs: cs[1] if cs and cs != 'PASS' and len(cs) > 1 and cs[1] in SHDCN else None, return_dtype=pl.String)

        def get_declarer_direction(contract_str_series: pl.Series) -> pl.Series:
            return contract_str_series.map_elements(lambda cs: cs[2] if cs and cs != 'PASS' and len(cs) > 2 and cs[2] in 'NESW' else None, return_dtype=pl.String)

        def get_doubled_status(contract_str_series: pl.Series) -> pl.Series:
            return contract_str_series.map_elements(lambda cs: cs[3:] if cs and cs != 'PASS' and len(cs) > 3 else '', return_dtype=pl.String)

        df = df.with_columns([
            get_bid_level(pl.col('Contract_std')).alias('BidLvl'),
            get_bid_suit(pl.col('Contract_std')).alias('BidSuit'),
            get_declarer_direction(pl.col('Contract_std')).alias('Declarer_Direction'),
            get_doubled_status(pl.col('Contract_std')).alias('Dbl')
        ])
        return df

    # Inputs: df column ['Declarer_Direction']
    # Outputs: df columns ['LHO_Direction', 'Dummy_Direction', 'RHO_Direction', 'Declarer_Pair_Direction', 'Opponent_Pair_Direction']
    # Description: Determines other player roles based on the declarer.
    def _determine_player_roles(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['Declarer_Direction'], "ContractAugmenter._determine_player_roles")
        df = df.with_columns([
            pl.col('Declarer_Direction').replace_strict(self.declarer_to_LHO_d, default=None).alias('LHO_Direction'),
            pl.col('Declarer_Direction').replace_strict(self.declarer_to_dummy_d, default=None).alias('Dummy_Direction'),
            pl.col('Declarer_Direction').replace_strict(self.declarer_to_RHO_d, default=None).alias('RHO_Direction'),
            pl.col('Declarer_Direction').replace(PlayerDirectionToPairDirection).alias('Declarer_Pair_Direction') # Use .replace()
        ])
        df = df.with_columns(
             pl.col('Declarer_Pair_Direction').replace(PairDirectionToOpponentPairDirection).alias('Opponent_Pair_Direction') # Use .replace()
        )
        return df

    # Inputs: df columns ['Declarer_Direction', 'Player_Name_N'...'W', 'Player_ID_N'...'W']
    # Outputs: df columns ['Declarer_Name', 'Declarer_ID']
    # Description: Identifies the declarer's name and ID.
    def _identify_declarer_info(self, df: pl.DataFrame) -> pl.DataFrame:
        # Ensure player name/ID columns exist (e.g. Player_Name_N, Player_ID_N)
        # These might need renaming from original N, E, S, W if they are just player names.
        # Assuming Player_Name_N, Player_ID_N etc. are present.
        # The original code renamed 'N' to 'Player_Name_N' etc. earlier.
        # If df has 'N','E','S','W' as names, we can use those or rename them first.
        # For this refactor, assume 'Player_Name_N', 'Player_ID_N' format.
        required_name_cols = [f'Player_Name_{d}' for d in NESW]
        required_id_cols = [f'Player_ID_{d}' for d in NESW]
        _assert_columns_exist(df, ['Declarer_Direction'] + required_name_cols + required_id_cols,
                              "ContractAugmenter._identify_declarer_info")

        # UDF to get name/ID based on declarer direction
        def get_player_attribute(row_struct: dict, attr_prefix: str) -> str:
            decl_dir = row_struct['Declarer_Direction']
            if not decl_dir: return None
            return row_struct.get(f'{attr_prefix}_{decl_dir}', None)

        df = df.with_columns([
            pl.struct(['Declarer_Direction'] + required_name_cols)
              .map_elements(lambda r: get_player_attribute(r, 'Player_Name'), return_dtype=pl.String)
              .alias('Declarer_Name'),
            pl.struct(['Declarer_Direction'] + required_id_cols)
              .map_elements(lambda r: get_player_attribute(r, 'Player_ID'), return_dtype=pl.String) # Assuming ID is string
              .alias('Declarer_ID')
        ])
        return df

    # Inputs: df columns ['Contract_std', 'BidLvl', 'BidSuit']
    # Outputs: df column ['ContractType'] (Pass, Partial, Game, SSlame, GSlam for the played contract)
    # Description: Classifies the type of the played contract.
    def _classify_played_contract_type(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['Contract_std', 'BidLvl', 'BidSuit'], "ContractAugmenter._classify_played_contract_type")

        df = df.with_columns(
            pl.when(pl.col('Contract_std') == 'PASS').then(pl.lit("Pass"))
            .when((pl.col('BidLvl') == 5) & pl.col('BidSuit').is_in(['C', 'D'])).then(pl.lit("Game"))
            .when((pl.col('BidLvl') == 4) & pl.col('BidSuit').is_in(['H', 'S'])).then(pl.lit("Game")) # 4H/4S
            .when((pl.col('BidLvl') == 5) & pl.col('BidSuit').is_in(['H', 'S'])).then(pl.lit("Game")) # 5H/5S is also game
            .when((pl.col('BidLvl') == 3) & (pl.col('BidSuit') == 'N')).then(pl.lit("Game")) # 3NT
            .when((pl.col('BidLvl') == 4) & (pl.col('BidSuit') == 'N')).then(pl.lit("Game")) # 4NT (less common game)
            .when((pl.col('BidLvl') == 5) & (pl.col('BidSuit') == 'N')).then(pl.lit("Game")) # 5NT (less common game)
            .when(pl.col('BidLvl') == 6).then(pl.lit("SSlam")) # Small Slam
            .when(pl.col('BidLvl') == 7).then(pl.lit("GSlam")) # Grand Slam
            .otherwise(pl.lit("Partial")) # If BidLvl is not null
            .alias('ContractType')
        )
        # Refine "Partial" for cases where BidLvl is null (e.g. PASS already handled)
        df = df.with_columns(
            pl.when(pl.col('BidLvl').is_null() & (pl.col('Contract_std') != 'PASS')).then(None) # Or some other indicator
            .otherwise(pl.col('ContractType')).alias('ContractType')
        )
        return df

    # Inputs: df columns ['Declarer_Direction', 'BidLvl', 'BidSuit', 'Vul_Declarer', 'DD_*', 'DD_Score_*']
    # Outputs: df columns ['DD_Tricks_Contract', 'DD_Tricks_Dummy_Contract', 'DD_Score_Contract_Declarer']
    # Description: Links the played contract to its corresponding DD values.
    def _link_contract_to_dd_data(self, df: pl.DataFrame) -> pl.DataFrame:
        # Requires DD_[Dir]_[Suit] (from DDSdAugmenter)
        # Requires DD_Score_[Level][Suit]_[Dir] (from DDSdAugmenter)
        # Requires Vul_Declarer (created here or by ScoreAugmenter)
        _assert_columns_exist(df, ['Declarer_Direction', 'Dummy_Direction', 'BidLvl', 'BidSuit', 'Vul',
                                   'Declarer_Pair_Direction'], # For Vul_Declarer
                              "ContractAugmenter._link_contract_to_dd_data (base)")

        # Create Vul_Declarer if not exists
        if 'Vul_Declarer' not in df.columns:
            df = df.with_columns(
                ( # Parentheses added here
                    ((pl.col('Declarer_Pair_Direction') == 'NS') & pl.col('Vul').is_in(['N_S', 'Both'])) |
                    ((pl.col('Declarer_Pair_Direction') == 'EW') & pl.col('Vul').is_in(['E_W', 'Both']))
                ).alias('Vul_Declarer') # And alias applied to the whole expression
            )

        # UDF to get DD tricks for declarer/dummy for the specific contract suit
        def get_dd_tricks_for_contract(row_struct: dict, player_role_col: str) -> int:
            player_dir = row_struct[player_role_col] # e.g. Declarer_Direction or Dummy_Direction
            bid_suit = row_struct['BidSuit']
            if not player_dir or not bid_suit: return None
            return row_struct.get(f'DD_{player_dir}_{bid_suit}', None)

        df = df.with_columns([
            pl.struct(['Declarer_Direction', 'BidSuit'] + [f'DD_{d}_{s}' for d in NESW for s in SHDCN])
              .map_elements(lambda r: get_dd_tricks_for_contract(r, 'Declarer_Direction'), return_dtype=pl.UInt8)
              .alias('DD_Tricks_Contract'),
            pl.struct(['Dummy_Direction', 'BidSuit'] + [f'DD_{d}_{s}' for d in NESW for s in SHDCN])
              .map_elements(lambda r: get_dd_tricks_for_contract(r, 'Dummy_Direction'), return_dtype=pl.UInt8)
              .alias('DD_Tricks_Dummy_Contract')
        ])

        # DD Score for the contract
        # Original: DD_Score_Declarer via DD_Score_Refs: 'DD_Score_'+BidLvl+BidSuit+'_'+Declarer_Direction
        # This means we need to pick from columns like 'DD_Score_1C_N' created by DDSdAugmenter.
        def get_dd_score_for_contract(row_struct: dict) -> int:
            bid_lvl = row_struct['BidLvl']
            bid_suit = row_struct['BidSuit']
            decl_dir = row_struct['Declarer_Direction']
            if bid_lvl is None or bid_suit is None or decl_dir is None: return None

            # The DD_Score columns from DDSd already factor in DD tricks and vul for that specific DD scenario
            dd_score_col_name = f'DD_Score_{bid_lvl}{bid_suit}_{decl_dir}'
            return row_struct.get(dd_score_col_name, None)

        # Need all DD_Score_[L][S]_[D] columns in the struct
        required_dd_score_cols = [f'DD_Score_{l}{s}_{d}' for l in range(1,8) for s in SHDCN for d in NESW]
        # Filter to only those that exist in df to prevent error if some are missing
        existing_dd_score_cols = [col for col in required_dd_score_cols if col in df.columns]


        df = df.with_columns(
            pl.struct(['BidLvl', 'BidSuit', 'Declarer_Direction'] + existing_dd_score_cols)
            .map_elements(get_dd_score_for_contract, return_dtype=pl.Int16)
            .alias('DD_Score_Contract_Declarer') # Score if contract is made with DD tricks by that declarer
        )
        return df

    # Inputs: df columns ['Declarer_Pair_Direction', 'Declarer_Direction', 'BidSuit', 'Probs_*']
    # Outputs: df columns ['Prob_Contract_Takes_0' ... '13', 'EV_ColName_For_Contract']
    # Description: Links played contract to SD probabilities and identifies EV column name.
    def _link_contract_to_sd_data(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['Declarer_Pair_Direction', 'Declarer_Direction', 'BidSuit'],
                              "ContractAugmenter._link_contract_to_sd_data")
        # Check for existence of Probs columns
        if not any(col.startswith("Probs_") for col in df.columns):
             print("Warning: SD Probability columns (Probs_*) not found. Skipping SD link for contract.")
             return df

        # Prob_Contract_Takes_T: Probability the current contract (Declarer_Direction, BidSuit) takes T tricks.
        # These probabilities come from Probs_[PairKnown]_[ActualDeclarer]_[Suit]_[Tricks]
        # Here, PairKnown is Declarer_Pair_Direction. ActualDeclarer is Declarer_Direction. Suit is BidSuit.
        prob_cols_for_contract = []
        for t in range(14): # 0 to 13 tricks
            # UDF to select the correct probability based on row's contract details
            def get_prob_for_trick_t(row_struct: dict, current_t: int) -> float:
                pair_dir = row_struct['Declarer_Pair_Direction']
                decl_dir = row_struct['Declarer_Direction']
                bid_suit = row_struct['BidSuit']
                if not pair_dir or not decl_dir or not bid_suit: return None
                prob_col_name = f'Probs_{pair_dir}_{decl_dir}_{bid_suit}_{current_t}'
                return row_struct.get(prob_col_name, None)

            prob_cols_for_contract.append(
                pl.struct(['Declarer_Pair_Direction', 'Declarer_Direction', 'BidSuit'] + \
                          [col for col in df.columns if col.startswith("Probs_")]) # Pass all Probs cols
                  .map_elements(lambda r, t_val=t: get_prob_for_trick_t(r, t_val), return_dtype=pl.Float32)
                  .alias(f'Prob_Contract_Takes_{t}')
            )
        df = df.with_columns(prob_cols_for_contract)

        # EV_ColName_For_Contract: Name of the EV column (e.g., EV_NS_N_S_1) that corresponds to this contract
        # This uses the "final choice" EV columns from DDSdAugmenter which are already Vul specific for the board.
        df = df.with_columns(
            pl.concat_str([
                pl.lit("EV_"),
                pl.col("Declarer_Pair_Direction"),
                pl.lit("_"),
                pl.col("Declarer_Direction"),
                pl.lit("_"),
                pl.col("BidSuit"),
                pl.lit("_"),
                pl.col("BidLvl").cast(pl.String)
            ]).alias("EV_ColName_For_Contract") # e.g., EV_NS_N_S_1 (without Vul state, as that's now selected)
        )
        return df

    def perform_augmentations(self, df: pl.DataFrame) -> pl.DataFrame:
        df = _time_operation("Contract: Standardize String", self._standardize_contract_string, df)
        df = _time_operation("Contract: Extract Components", self._extract_contract_components, df)
        df = _time_operation("Contract: Determine Player Roles", self._determine_player_roles, df)
        df = _time_operation("Contract: Identify Declarer Info", self._identify_declarer_info, df) # Requires Player_Name/ID cols
        df = _time_operation("Contract: Classify Played Contract Type", self._classify_played_contract_type, df)
        df = _time_operation("Contract: Link to DD Data", self._link_contract_to_dd_data, df) # Requires DD cols
        df = _time_operation("Contract: Link to SD Data", self._link_contract_to_sd_data, df) # Requires SD Prob cols
        return df

class ScoreAugmenter:
    """Calculations based on the contract's outcome (Result, Score, Differences, Ratings)."""
    def __init__(self, all_scores_d): # Needs all_scores_d for score computation
        self.all_scores_d = all_scores_d

    # Inputs: df column ['Contract_std'] (standardized contract string)
    # Outputs: df column ['Result'] (integer result, e.g. 0 for made, -1 for down 1)
    # Description: Extracts the numerical result from the contract string.
    def _add_result_from_contract_string(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['Contract_std'], "ScoreAugmenter._add_result_from_contract_string")
        # Original: convert_contract_to_result
        # e.g., "4HNE=" -> 0, "4HNX-1" -> -1, "4HNE+1" -> 1
        # Assumes Contract_std might have result like "4H N X -1" or "4H N =".
        # Polars str.extract using regex is good here.
        # Pattern: optional sign ([+-]) followed by one or more digits (\d+), or an equals sign (=).
        # And ensure it's at the end of the string.

        # Simplified from original: if contract string ends with = or 0 -> result 0.
        # if ends with +N -> result N. if ends with -N -> result -N.
        def parse_result(contract_series: pl.Series) -> pl.Series:
            return contract_series.map_elements(
                lambda cs: 0 if cs is None or cs == 'PASS' or cs.endswith("=") or cs.endswith("0") # check = or 0 at end
                           else int(cs[-1]) if cs[-2] == '+' and cs[-1].isdigit()
                           else -int(cs[-1]) if cs[-2] == '-' and cs[-1].isdigit()
                           else None, # No explicit result string, might be calculated from tricks
                return_dtype=pl.Int8,
            )
        if 'Result' not in df.columns: # Only add if not user-provided
             df = df.with_columns(parse_result(pl.col('Contract_std')).alias('Result'))
        return df

    # Inputs: df columns ['BidLvl', 'Result']
    # Outputs: df column ['TricksTaken'] (actual tricks taken by declarer)
    # Description: Calculates the number of tricks taken based on bid level and result.
    def _add_tricks_taken_column(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['BidLvl', 'Result'], "ScoreAugmenter._add_tricks_taken_column")
        if 'TricksTaken' not in df.columns: # Only add if not user-provided
            df = df.with_columns(
                (pl.col('BidLvl') + 6 + pl.col('Result')).cast(pl.UInt8).alias('TricksTaken')
            )
            # Handle PASS case where BidLvl might be null
            df = df.with_columns(
                pl.when(pl.col('Contract_std') == 'PASS').then(None) # Or 0 tricks if that's convention
                .otherwise(pl.col('TricksTaken')).alias('TricksTaken')
            )
        return df

    # Inputs: df columns ['Score'] (original score if exists), ['Declarer_Direction', 'BidLvl', 'BidSuit', 'TricksTaken', 'Vul_Declarer', 'Dbl', 'Contract_std', 'Result']
    # Outputs: df columns ['Score_NS', 'Score_EW', 'Score_Declarer', 'Computed_Score_Declarer']
    # Description: Calculates NS/EW scores from contract outcome. Validates against 'Score' if present.
    def _add_actual_score_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        # Vul_Declarer is crucial, should be made by ContractAugmenter
        _assert_columns_exist(df, ['Declarer_Direction', 'BidLvl', 'BidSuit', 'TricksTaken',
                                   'Vul_Declarer', 'Dbl', 'Contract_std', 'Result',
                                   'Declarer_Pair_Direction'],
                              "ScoreAugmenter._add_actual_score_columns")

        # Compute score based on contract details
        def compute_score_udf(row_struct: dict) -> int:
            if row_struct['Contract_std'] == 'PASS': return 0
            if row_struct['BidLvl'] is None or \
               row_struct['BidSuit'] is None or \
               row_struct['TricksTaken'] is None or \
               row_struct['Vul_Declarer'] is None or \
               row_struct['Dbl'] is None or \
               row_struct['Result'] is None: # Result from contract string might be None if not explicit
                return None # Cannot compute

            # Use self.all_scores_d: {(level, suit_char, tricks_taken, is_vul, penalty_abbr): score}
            key = (
                row_struct['BidLvl'],
                row_struct['BidSuit'],
                row_struct['TricksTaken'],
                row_struct['Vul_Declarer'],
                row_struct['Dbl']
            )
            return self.all_scores_d.get(key, None) # Return None if score not found (e.g. impossible result)

        df = df.with_columns(
            pl.struct(['Contract_std', 'BidLvl', 'BidSuit', 'TricksTaken', 'Vul_Declarer', 'Dbl', 'Result'])
            .map_elements(compute_score_udf, return_dtype=pl.Int16)
            .alias('Computed_Score_Declarer')
        )

        # If original 'Score' column exists, it's usually from declarer's perspective.
        # If not, Computed_Score_Declarer is the main score from declarer's view.
        # Create Score_Declarer: if 'Score' exists use it, else use Computed_Score_Declarer.
        if 'Score' in df.columns: # 'Score' is typically the score as reported, from declarer's view
            df = df.with_columns(pl.col('Score').alias('Score_Declarer'))
            # Assert Computed_Score_Declarer against Score if desired
            # non_matching_scores = df.filter(pl.col('Computed_Score_Declarer') != pl.col('Score_Declarer'))
            # if not non_matching_scores.is_empty():
            #     print("Warning: Computed_Score_Declarer does not match original Score for some rows.")
            #     # print(non_matching_scores.select(['Contract_std', 'Score', 'Computed_Score_Declarer']))
        else:
            df = df.with_columns(pl.col('Computed_Score_Declarer').alias('Score_Declarer'))

        # Create Score_NS and Score_EW
        df = df.with_columns([
            pl.when(pl.col('Declarer_Pair_Direction') == 'NS').then(pl.col('Score_Declarer'))
            .when(pl.col('Declarer_Pair_Direction') == 'EW').then(-pl.col('Score_Declarer'))
            .otherwise(0) # For PASS or if Declarer_Pair_Direction is None
            .alias('Score_NS'),
            pl.when(pl.col('Declarer_Pair_Direction') == 'EW').then(pl.col('Score_Declarer'))
            .when(pl.col('Declarer_Pair_Direction') == 'NS').then(-pl.col('Score_Declarer'))
            .otherwise(0)
            .alias('Score_EW')
        ])
        return df

    # Inputs: df columns ['Result', 'TricksTaken', 'DD_Tricks_Contract']
    # Outputs: df columns ['OverTricks', 'JustMade', 'UnderTricks', 'Tricks_Declarer_Actual', 'Tricks_DD_Diff_Declarer']
    # Description: Analyzes trick-taking performance against contract and DD.
    def _analyze_trick_taking(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['Result', 'TricksTaken', 'DD_Tricks_Contract'],# DD_Tricks_Contract from ContractAugmenter
                              "ScoreAugmenter._analyze_trick_taking")
        df = df.with_columns([
            (pl.col('Result') > 0).alias('OverTricks'),
            (pl.col('Result') == 0).alias('JustMade'),
            (pl.col('Result') < 0).alias('UnderTricks'),
            pl.col('TricksTaken').alias('Tricks_Declarer_Actual'), # Renaming for clarity vs DD_Tricks
            (pl.col('TricksTaken').cast(pl.Int8) - pl.col('DD_Tricks_Contract').cast(pl.Int8)).alias('Tricks_DD_Diff_Declarer')
        ])
        return df

    # Inputs: df columns ['Score_NS', 'Score_EW', 'Par_NS', 'Par_EW', 'EV_Max_NS', 'EV_Max_EW']
    # Outputs: df columns ['Par_Diff_NS', 'Par_Diff_EW', 'EV_Max_Diff_NS', 'EV_Max_Diff_EW']
    # Description: Calculates differences between actual scores and Par/EV scores.
    def _add_score_differences(self, df: pl.DataFrame) -> pl.DataFrame:
        # Par_NS, Par_EW from DDSdAugmenter
        # EV_Max_NS, EV_Max_EW from DDSdAugmenter (final EV choices)
        _assert_columns_exist(df, ['Score_NS', 'Score_EW', 'Par_NS', 'Par_EW',
                                   'EV_Max_NS', 'EV_Max_EW'], # EV_Max from DDSd
                              "ScoreAugmenter._add_score_differences")
        df = df.with_columns([
            (pl.col('Score_NS') - pl.col('Par_NS')).alias('Par_Diff_NS'),
            (pl.col('Score_EW') - pl.col('Par_EW')).alias('Par_Diff_EW'), # Or just -Par_Diff_NS
            (pl.col('Score_NS').cast(pl.Float32) - pl.col('EV_Max_NS').cast(pl.Float32)).alias('EV_Max_Diff_NS'),
            (pl.col('Score_EW').cast(pl.Float32) - pl.col('EV_Max_EW').cast(pl.Float32)).alias('EV_Max_Diff_EW')
        ])
        return df

    # Inputs: df columns ['Tricks_DD_Diff_Declarer', 'Declarer_ID', 'Score_Declarer', 'Par_Declarer', 'LHO_Direction', 'RHO_Direction', 'Player_ID_*']
    # Outputs: df columns ['Declarer_Rating', 'Defender_OnLead_Rating', 'Defender_NotOnLead_Rating']
    # Description: Calculates player ratings based on performance.
    def _add_player_ratings(self, df: pl.DataFrame) -> pl.DataFrame:
        # Par_Declarer: Score_Declarer perspective of Par. Needs Declarer_Pair_Direction, Par_NS, Par_EW.
        # OnLead / NotOnLead Player IDs also needed.
        _assert_columns_exist(df, ['Tricks_DD_Diff_Declarer', 'Declarer_ID', 'Score_Declarer',
                                   'Declarer_Pair_Direction', 'Par_NS', 'Par_EW', # For Par_Declarer
                                   'LHO_Direction', # OnLead is LHO of declarer
                                   ] + [f'Player_ID_{d}' for d in NESW], # For OnLead_ID, NotOnLead_ID
                              "ScoreAugmenter._add_player_ratings")

        # Create Par_Declarer
        if 'Par_Declarer' not in df.columns:
            df = df.with_columns(
                pl.when(pl.col('Declarer_Pair_Direction') == 'NS').then(pl.col('Par_NS'))
                .when(pl.col('Declarer_Pair_Direction') == 'EW').then(pl.col('Par_EW'))
                .otherwise(None).alias('Par_Declarer') # For PASS etc.
            )

        # Create Defender_Par_GE (True if defenders did better or equal to par against contract)
        # This means Score_Declarer <= Par_Declarer
        if 'Defender_Par_GE' not in df.columns:
             df = df.with_columns(
                (pl.col('Score_Declarer') <= pl.col('Par_Declarer')).alias('Defender_Par_GE')
             )

        # Get OnLead Player ID (LHO of declarer)
        if 'OnLead_ID' not in df.columns:
            def get_playerid_from_direction_col(row_struct: dict, direction_col_name: str) -> str:
                direction = row_struct[direction_col_name]
                if not direction: return None
                return row_struct.get(f'Player_ID_{direction}', None)

            df = df.with_columns(
                pl.struct(['LHO_Direction'] + [f'Player_ID_{d}' for d in NESW])
                .map_elements(lambda r: get_playerid_from_direction_col(r, 'LHO_Direction'), return_dtype=pl.String)
                .alias('OnLead_ID')
            )
            # Similarly for NotOnLead_ID (RHO of declarer, if needed, original used 'NotOnLead' which was RHO)
            # RHO_Direction should exist from ContractAugmenter
            _assert_columns_exist(df, ['RHO_Direction'], "ScoreAugmenter._add_player_ratings (RHO)")
            df = df.with_columns(
                pl.struct(['RHO_Direction'] + [f'Player_ID_{d}' for d in NESW])
                .map_elements(lambda r: get_playerid_from_direction_col(r, 'RHO_Direction'), return_dtype=pl.String)
                .alias('NotOnLead_ID') # Assuming NotOnLead refers to RHO
            )


        # Calculate ratings (mean over player ID)
        df = df.with_columns([
            pl.col('Tricks_DD_Diff_Declarer').mean().over('Declarer_ID').alias('Declarer_Rating'),
            pl.col('Defender_Par_GE').cast(pl.Float32).mean().over('OnLead_ID').alias('Defender_OnLead_Rating'),
            pl.col('Defender_Par_GE').cast(pl.Float32).mean().over('NotOnLead_ID').alias('Defender_NotOnLead_Rating')
        ])
        return df

    def perform_augmentations(self, df: pl.DataFrame) -> pl.DataFrame:
        df = _time_operation("Score: Add Result from Contract String", self._add_result_from_contract_string, df)
        df = _time_operation("Score: Add Tricks Taken", self._add_tricks_taken_column, df)
        df = _time_operation("Score: Add Actual Score Columns", self._add_actual_score_columns, df) # Uses Vul_Declarer
        df = _time_operation("Score: Analyze Trick Taking", self._analyze_trick_taking, df)
        df = _time_operation("Score: Add Score Differences", self._add_score_differences, df)
        df = _time_operation("Score: Add Player Ratings", self._add_player_ratings, df)
        return df

class MatchPointAugmenter:
    """Matchpoint scoring calculations."""
    def __init__(self):
        # Define which score columns to calculate MPs for.
        # These are scores from NS perspective primarily.
        self.discrete_score_columns_ns = ['DD_Score_Contract_Declarer_NS', 'Par_NS', 'EV_Max_NS']
        # DD_Score_Contract_Declarer_NS: DD score of the actual contract, from NS perspective
        # Par_NS: Par score from NS perspective
        # EV_Max_NS: Max EV score for NS from SD analysis

        # Columns for all possible DD contracts by declarer.
        self.all_dd_contract_score_cols = [f'DD_Score_{l}{s}_{d}' for l in range(1,8) for s in SHDCN for d in NESW]
        # Columns for all possible EV contracts by declarer/pair (final board-vul specific)
        self.all_ev_contract_score_cols = [f'EV_{pd}_{d}_{s}_{l}' for pd in NS_EW for d in pd for s in SHDCN for l in range(1,8)]


    # Calculate matchpoints for a generic score column against actual scores ('Score_NS')
    def _calculate_mps_for_score_col(self, df_group: pl.DataFrame, score_col_name: str) -> pl.Series:
        """
        Helper to calculate MPs for a given score column within a group.
        df_group is a group from df.group_by(...).
        score_col_name is the name of the column (e.g., 'Par_NS') whose values are being compared.
        """
        # For each value in df_group[score_col_name], compare it against all values in df_group['Score_NS']
        # Polars equivalent of original calculate_matchpoints_group
        # This is complex with map_groups if score_col_name and Score_NS are both varying.
        # Original `pl.map_groups(exprs=[col, 'Score_NS'], function=...)`
        # The function received two series for the group.

        # Assuming this function is called within a map_groups context where `series_list` is [col_series, score_ns_series]
        # For direct use in with_columns, if we need to rank `df[score_col_name]` values based on `df['Score_NS']` distribution *per board*.

        # This function is better as the UDF for map_groups as in original code.
        # `series_list[0]` is the column we're scoring (e.g. Par_NS values for this board)
        # `series_list[1]` is the list of actual Score_NS values on this board

        # This method should be static or defined inside the caller if it's for map_groups
        # For now, keep the logic here, to be used by _add_all_score_matchpoints
        pass


    # Inputs: df columns ['Score_NS', 'session_id', 'PBN', 'Board'] (Board can be any unique board identifier per session)
    # Outputs: df column ['MP_Top']
    # Description: Calculates the maximum possible matchpoints for each board.
    def _add_mp_top_column(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['Score_NS', 'session_id', 'PBN', 'Board'], "MatchPointAugmenter._add_mp_top_column")
        group_cols = ['session_id', 'PBN', 'Board'] # Group by unique board instance
        df = df.with_columns(
            (pl.col('Score_NS').count().over(group_cols) - 1).alias('MP_Top')
        )
        return df

    # Inputs: df columns ['Score_NS', 'Score_EW', 'session_id', 'PBN', 'Board']
    # Outputs: df columns ['MP_NS', 'MP_EW']
    # Description: Calculates matchpoints for North-South and East-West.
    def _add_actual_matchpoint_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['Score_NS', 'Score_EW', 'session_id', 'PBN', 'Board'],
                             "MatchPointAugmenter._add_actual_matchpoint_columns")
        group_cols = ['session_id', 'PBN', 'Board']
        df = df.with_columns([
            (pl.col('Score_NS').rank(method='average', descending=False).over(group_cols) - 1.0).alias('MP_NS'),
            (pl.col('Score_EW').rank(method='average', descending=False).over(group_cols) - 1.0).alias('MP_EW')
        ])
        return df

    # Inputs: df columns ['MP_NS', 'MP_EW', 'MP_Top']
    # Outputs: df columns ['Pct_NS', 'Pct_EW']
    # Description: Calculates matchpoint percentages.
    def _add_percentage_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['MP_NS', 'MP_EW', 'MP_Top'], "MatchPointAugmenter._add_percentage_columns")
        df = df.with_columns([
            (pl.col('MP_NS') / pl.col('MP_Top')).fill_null(0.5).fill_nan(0.5).alias('Pct_NS'), # Handle MP_Top=0 case (single result)
            (pl.col('MP_EW') / pl.col('MP_Top')).fill_null(0.5).fill_nan(0.5).alias('Pct_EW')
        ])
        # Ensure Pct is between 0 and 1 if MP_Top can be 0.
        # If MP_Top is 0, means 1 pair played, score is 0/0 -> NaN. Should be 50% or 100% depending on convention for 1 result.
        # rank() - 1 gives 0 MPs for the single pair. 0 / 0 = NaN.
        # If only one result, Pct is often considered 0.5 or not calculated.
        # fill_null(0.5) assumes 50% if MP_Top is 0 (rank is 1, MP_NS is 0, MP_Top is 0).
        return df

    # Inputs: df columns ['Declarer_Direction', 'Pct_NS', 'Pct_EW']
    # Outputs: df column ['Declarer_Pct']
    # Description: Percentage from the declarer's perspective.
    def _add_declarer_percentage_column(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['Declarer_Direction', 'Pct_NS', 'Pct_EW'],
                             "MatchPointAugmenter._add_declarer_percentage_column")
        df = df.with_columns(
            pl.when(pl.col('Declarer_Direction').is_in(['N', 'S'])).then(pl.col('Pct_NS'))
            .when(pl.col('Declarer_Direction').is_in(['E', 'W'])).then(pl.col('Pct_EW'))
            .otherwise(None) # For PASS etc.
            .alias('Declarer_Pct')
        )
        return df

    # Inputs: df, various score columns (e.g. Par_NS, DD_Score_1C_N_NS)
    # Outputs: df columns for MPs of these scores (e.g. MP_Par_NS)
    # Description: Calculates MPs for hypothetical scores against the actual distribution.
    def _add_hypothetical_matchpoint_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        group_cols = ['session_id', 'PBN', 'Board']

        # Static helper for map_groups UDF
        def _calculate_matchpoints_group_udf(series_list: list[pl.Series]) -> pl.Series:
            # series_list[0] = values of the hypothetical score column for the group (e.g., Par_NS for all rows of this board)
            # series_list[1] = actual Score_NS values for all rows of this board
            hypothetical_scores = series_list[0].fill_null(0.0) # Fill nulls for comparison
            actual_scores_on_board = series_list[1].fill_null(0.0)

            results = []
            for h_score in hypothetical_scores: # For each row's hypothetical score in this group
                wins = 0.0
                ties = 0.0
                for act_score in actual_scores_on_board:
                    if h_score > act_score:
                        wins += 1.0
                    elif h_score == act_score:
                        ties += 1.0
                results.append(wins + ties * 0.5)
            return pl.Series(values=results, dtype=pl.Float64)

        # Create NS-perspective version of DD_Score_Contract_Declarer
        if 'DD_Score_Contract_Declarer' in df.columns and 'Declarer_Pair_Direction' in df.columns:
            df = df.with_columns(
                pl.when(pl.col('Declarer_Pair_Direction') == 'NS').then(pl.col('DD_Score_Contract_Declarer'))
                .when(pl.col('Declarer_Pair_Direction') == 'EW').then(-pl.col('DD_Score_Contract_Declarer'))
                .otherwise(0).alias('DD_Score_Contract_Declarer_NS') # NS perspective
            )
        else: # Ensure column exists even if it can't be computed
            if 'DD_Score_Contract_Declarer_NS' not in df.columns:
                 df = df.with_columns(pl.lit(0, dtype=pl.Int16).alias('DD_Score_Contract_Declarer_NS'))


        cols_to_process = [col for col in self.discrete_score_columns_ns if col in df.columns]

        # Also for all DD contract scores (from NS perspective) and EV scores (NS perspective)
        # This part of original code was very expansive. Let's focus on key ones.
        # For example, MP for player N declaring 1C based on DD_Score_1C_N
        # These need to be put into NS perspective first.

        # Example: Processing DD_Score_1C_N (which is already a score)
        # We need its NS-perspective version. If N declared, it's Score_NS. If E declared, it's -Score_EW for N.
        # This becomes complex if we are calculating MPs for every possible DD contract.
        # The original `all_score_columns` included raw DD_Score_[L][S]_[D] and EV_[PD]_[D]_[S]_[L].
        # These are already scores. We need to define their "NS perspective".
        # For a column like DD_Score_1C_N: if D is N or S, it's an NS score. If D is E or W, it's an EW score (so -ve for NS).

        temp_hypothetical_score_cols = [] # Store names of temp NS-perspective cols

        for col_raw_dd_score in self.all_dd_contract_score_cols:
            if col_raw_dd_score in df.columns:
                # Determine if this score is for an NS declarer or EW declarer from its name
                # DD_Score_1C_N -> declarer is N (NS pair)
                declarer_char = col_raw_dd_score[-1]

                ns_perspective_col_name = f"{col_raw_dd_score}_NS_perspective"
                temp_hypothetical_score_cols.append(ns_perspective_col_name)

                if declarer_char in ('N', 'S'):
                    df = df.with_columns(pl.col(col_raw_dd_score).alias(ns_perspective_col_name))
                elif declarer_char in ('E', 'W'):
                    df = df.with_columns((-pl.col(col_raw_dd_score)).alias(ns_perspective_col_name))

        # Similar for EV scores EV_[PairDeclaring]_[ActualDeclarer]_[Suit]_[Level]
        # If PairDeclaring is NS, it's an NS score. If EW, it's an EW score.
        for col_raw_ev_score in self.all_ev_contract_score_cols:
            if col_raw_ev_score in df.columns:
                pair_decl_char = col_raw_ev_score.split('_')[1] # EV_NS_N_S_1 -> NS

                ns_perspective_col_name = f"{col_raw_ev_score}_NS_perspective"
                temp_hypothetical_score_cols.append(ns_perspective_col_name)

                if pair_decl_char == 'NS':
                    df = df.with_columns(pl.col(col_raw_ev_score).alias(ns_perspective_col_name))
                elif pair_decl_char == 'EW':
                     df = df.with_columns((-pl.col(col_raw_ev_score)).alias(ns_perspective_col_name))

        cols_to_process.extend(temp_hypothetical_score_cols) # Add these to the list for MP calculation

        for score_col in cols_to_process:
            if score_col in df.columns and 'Score_NS' in df.columns: # Ensure both exist
                mp_col_name = score_col.replace("_NS_perspective","").replace("_Declarer_NS","") # Clean up name
                mp_col_name = f"MP_{mp_col_name}"

                if mp_col_name not in df.columns:
                    df = df.with_columns(
                        pl.map_groups(
                            exprs=[score_col, 'Score_NS'], # Pass these two series to UDF for each group
                            function=_calculate_matchpoints_group_udf,
                        ).over(group_cols).alias(mp_col_name)
                    )
            else:
                print(f"Warning: Skipping MP calc for {score_col} due to missing columns (itself or Score_NS).")

        # Optionally, drop temporary _NS_perspective columns
        # df = df.drop(temp_hypothetical_score_cols)
        return df


    def _add_hypothetical_percentage_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        _assert_columns_exist(df, ['MP_Top'], "MatchPointAugmenter._add_hypothetical_percentage_columns")
        # For every 'MP_SomeScore' column, create 'Pct_SomeScore'
        mp_cols = [col for col in df.columns if col.startswith("MP_") and col not in ['MP_NS', 'MP_EW', 'MP_Top']]
        pct_expressions = []
        for mp_col_name in mp_cols:
            # pct_col_name = mp_col_name.replace("MP_", "Pct_") # Pct_Par_NS etc.
            # If mp_col_name is MP_DD_Score_1C_N, pct_col_name is Pct_DD_Score_1C_N
            # Need to also create EW perspective for these percentages if they are NS-based MPs

            # Example: if mp_col_name is "MP_Par_NS"
            # Pct_Par_NS = MP_Par_NS / MP_Top
            # Pct_Par_EW = (MP_Top - MP_Par_NS) / MP_Top = 1 - Pct_Par_NS

            pct_ns_col_name = mp_col_name.replace("MP_", "Pct_") # Assumes mp_col_name is NS-oriented

            # Check if it's an NS-specific MP column (e.g. ends with _NS or is one of the discrete NS ones)
            is_ns_oriented = mp_col_name.endswith("_NS") or \
                             any(disc_ns_base in mp_col_name for disc_ns_base in [s.replace("_NS","") for s in self.discrete_score_columns_ns])

            if is_ns_oriented:
                pct_expressions.append(
                    (pl.col(mp_col_name) / pl.col('MP_Top')).fill_null(0.5).fill_nan(0.5).alias(pct_ns_col_name)
                )
                # Create EW counterpart if it's an NS score like Par_NS
                if pct_ns_col_name.endswith("_NS"):
                    pct_ew_col_name = pct_ns_col_name.replace("_NS", "_EW")
                    pct_expressions.append(
                        (1.0 - pl.col(pct_ns_col_name)).alias(pct_ew_col_name) # (MP_Top - MP_xxx_NS) / MP_Top
                    )
            # else: # It's a specific contract's MP, e.g. MP_DD_Score_1C_N
                  # This is already an "absolute" MP for that contract.
                  # Its percentage is just MP / MP_Top.
            #     pct_expressions.append(
            #         (pl.col(mp_col_name) / pl.col('MP_Top')).fill_null(0.5).fill_nan(0.5).alias(pct_ns_col_name)
            #     )


        # Specific final percentages from original MatchPointAugmenter._calculate_final_scores
        # This includes various Max Pct columns.
        # e.g., DD_Score_Pct_NS_Max = MP_DD_Score_NS_Max / MP_Top
        # Need to create MP_DD_Score_NS_Max first (max over MP_DD_Score_1C_N, MP_DD_Score_1D_N etc.)
        # This part requires careful reconstruction of the original logic for specific aggregate MP/Pct cols.
        # For this refactor, we'll focus on the pattern above. Detailed final scores would expand this.

        if pct_expressions:
            df = df.with_columns(pct_expressions)
        return df


    def perform_augmentations(self, df: pl.DataFrame) -> pl.DataFrame:
        df = _time_operation("MatchPoint: Add MP Top", self._add_mp_top_column, df)
        df = _time_operation("MatchPoint: Add Actual MPs", self._add_actual_matchpoint_columns, df)
        df = _time_operation("MatchPoint: Add Percentages", self._add_percentage_columns, df)
        df = _time_operation("MatchPoint: Add Declarer Percentage", self._add_declarer_percentage_column, df)
        df = _time_operation("MatchPoint: Add Hypothetical MPs", self._add_hypothetical_matchpoint_columns, df)
        df = _time_operation("MatchPoint: Add Hypothetical Percentages", self._add_hypothetical_percentage_columns, df)
        return df

class ImpAugmenter:
    """Placeholder for IMP scoring calculations."""
    def __init__(self):
        pass

    def perform_augmentations(self, df: pl.DataFrame) -> pl.DataFrame:
        print("ImpAugmenter.perform_augmentations - Not yet implemented.")
        # Example: Calculate IMPs vs Par_NS
        # if 'Score_NS' in df.columns and 'Par_NS' in df.columns:
        #     # Basic IMP scale (highly simplified)
        #     def score_diff_to_imp(diff_series: pl.Series) -> pl.Series:
        #         # Apply IMP scale logic
        #         return diff_series.map_elements(lambda diff: (diff // 100) if diff is not None else 0,
        #                                       return_dtype=pl.Int8)
        #
        #     df = df.with_columns(
        #         score_diff_to_imp(pl.col('Score_NS') - pl.col('Par_NS')).alias('IMP_vs_Par')
        #     )
        return df

# ## Main Orchestrator
class BridgeDataAugmenter:
    def __init__(self, hrs_d_cache=None, progress_callback=None, sd_productions=40):
        """
        Initializes the BridgeDataAugmenter.
        Args:
            hrs_d_cache (dict, optional): Cache for hand results (DD/SD). Defaults to None (new cache).
            progress_callback (function, optional): Callback for progress updates (current, total, message).
            sd_productions (int): Number of productions for Single Dummy simulations.
        """
        self.hrs_d = hrs_d_cache if hrs_d_cache is not None else {}
        self.progress_callback = progress_callback
        self.sd_productions = sd_productions # Store this for DDSdAugmenter

        # Initialize individual augmenters
        self.deal_augmenter = DealAugmenter()
        self.hand_eval_augmenter = HandEvaluationAugmenter()

        # DDSdAugmenter needs score tables and cache
        self.ddsd_augmenter = DDSdAugmenter(hrs_d=self.hrs_d, progress_callback=self.progress_callback)

        # ContractAugmenter and ScoreAugmenter might need all_scores_d from ddsd_augmenter
        self.contract_augmenter = ContractAugmenter(all_scores_d=self.ddsd_augmenter.all_scores_d)
        self.score_augmenter = ScoreAugmenter(all_scores_d=self.ddsd_augmenter.all_scores_d)

        self.matchpoint_augmenter = MatchPointAugmenter()
        self.imp_augmenter = ImpAugmenter()

        # Store player name columns if they exist for renaming, or define expected ones
        self.player_name_cols_original = ['N', 'E', 'S', 'W'] # If original df uses these for names
        self.player_name_cols_target = [f'Player_Name_{d}' for d in NESW]
        self.player_id_cols_target = [f'Player_ID_{d}' for d in NESW] # Assume these need to exist or be created

    def _rename_player_columns_if_needed(self, df: pl.DataFrame) -> pl.DataFrame:
        """Renames N,E,S,W to Player_Name_N etc. if they exist and Player_Name_* do not."""
        renames = {}
        for i, orig_col in enumerate(self.player_name_cols_original):
            target_col = self.player_name_cols_target[i]
            if orig_col in df.columns and target_col not in df.columns:
                renames[orig_col] = target_col
        if renames:
            print(f"Renaming player columns: {renames}")
            df = df.rename(renames)

        # Ensure Player_ID columns exist, if not, create placeholders from names or default
        for i, id_col in enumerate(self.player_id_cols_target):
            if id_col not in df.columns:
                 name_col_for_id = self.player_name_cols_target[i]
                 if name_col_for_id in df.columns: # Create ID from name if name exists
                      df = df.with_columns(pl.col(name_col_for_id).alias(id_col))
                 else: # Create default ID (e.g. 'Unknown_N')
                      df = df.with_columns(pl.lit(f"Unknown_{NESW[i]}").alias(id_col))
        return df

    def _perform_legacy_renames(self, df: pl.DataFrame) -> pl.DataFrame:
        """Applies column renames as in original Perform_Legacy_Renames."""
        # This function needs to be carefully updated based on the new column names
        # and whether these renames are still desired or handled by new structure.
        # Example renames from original:
        # pl.col('Declarer_Name').alias('Name_Declarer'),
        # pl.col('Declarer_ID').alias('Number_Declarer'),
        # For now, this is a placeholder for specific final renames.
        print("BridgeDataAugmenter._perform_legacy_renames - Placeholder, adapt as needed.")
        # if 'Declarer_Name' in df.columns and 'Name_Declarer' not in df.columns:
        #     df = df.rename({'Declarer_Name': 'Name_Declarer'})
        return df

    def _create_fake_predictions(self, df: pl.DataFrame) -> pl.DataFrame:
        """Adds placeholder columns for ML predictions if needed."""
        print("BridgeDataAugmenter._create_fake_predictions - Placeholder for fake prediction columns.")
        # Example:
        # if 'Pct_NS_Pred' not in df.columns and 'Pct_NS' in df.columns:
        #    df = df.with_columns(pl.col('Pct_NS').alias('Pct_NS_Pred'))
        return df


    def augment_data(self, df: pl.DataFrame, augment_acbl_specific=False) -> pl.DataFrame:
        """
        Main method to apply all augmentations to the DataFrame.
        """
        df = _time_operation("Orchestrator: Rename Player Columns", self._rename_player_columns_if_needed, df)

        df = self.deal_augmenter.perform_augmentations(df)
        df = self.hand_eval_augmenter.perform_augmentations(df) # Initial hand evals

        # DDSd calculations (can be lengthy)
        df = self.ddsd_augmenter.perform_augmentations(df, sd_productions=self.sd_productions)

        # LoTT depends on DD columns, call it after DDSdAugmenter
        df = _time_operation("HandEval: Add LoTT Columns (post-DD)", self.hand_eval_augmenter._add_lott_columns, df)


        # Contract processing (depends on base deal info, and potentially links to DD/SD)
        # Player_Name_N/E/S/W and Player_ID_N/E/S/W must exist before this.
        df = self.contract_augmenter.perform_augmentations(df)

        # Scoring (depends on contract details and outcomes)
        df = self.score_augmenter.perform_augmentations(df)

        # Competitive Scoring
        df = self.matchpoint_augmenter.perform_augmentations(df)
        df = self.imp_augmenter.perform_augmentations(df)

        # Final legacy renames or placeholder predictions
        df = _time_operation("Orchestrator: Perform Legacy Renames", self._perform_legacy_renames, df)
        df = _time_operation("Orchestrator: Create Fake Predictions", self._create_fake_predictions, df)

        if augment_acbl_specific:
            # Original AugmentACBLHandRecords called HandAugmenter.
            # HandAugmenter's logic is now distributed.
            # ACBL specific might involve date parsing or ID casting from original script.
            if 'game_date' in df.columns and 'Date' not in df.columns:
                df = df.with_columns(
                    pl.col('game_date').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S', strict=False)
                    .cast(pl.Date).alias('Date') # Cast to Date if only date part needed
                )
            if 'hand_record_id' in df.columns and df['hand_record_id'].dtype != pl.String:
                df = df.with_columns(pl.col('hand_record_id').cast(pl.String))

        print("Bridge data augmentation complete.")
        return df

# ## Example Usage (Conceptual)
#
# ```python
# # Assuming 'df_input' is your initial Polars DataFrame with columns like 'PBN', 'Contract', 'Score', etc.
# # and player names as 'N', 'E', 'S', 'W' or 'Player_Name_N', etc.
# # and 'board_number' or 'Vul', 'Dealer'.
#
# # Initialize the main augmenter
# data_augmenter = BridgeDataAugmenter(sd_productions=10) # Using fewer SD productions for speed in example
#
# # Perform all augmentations
# df_augmented = data_augmenter.augment_data(df_input.clone()) # Use .clone() if you want to keep original df_input
#
# # df_augmented now contains all the new columns.
# print(df_augmented.head())
# print(f"DataFrame shape after augmentation: {df_augmented.shape}")
# # print(df_augmented.columns)
# ```