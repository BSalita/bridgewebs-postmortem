"""
PBN Results Calculator Streamlit Application
"""

# streamlit program to display Bridge game deal statistics from a PBN file.
# Invoke from system prompt using: streamlit run CalculatePBNResults_Streamlit.py

# todo:
# before production, ask claude to look for bugs and concurrency issues.


import streamlit as st
import streamlit_chat
from streamlit_extras.bottom_container import bottom
from stqdm import stqdm

import pathlib
import polars as pl
import duckdb
import json
from collections import defaultdict
from datetime import datetime, timezone
import sys
import os
import platform
from dotenv import load_dotenv
import html

# Only declared to display version information
import numpy as np
import pandas as pd

import endplay # for __version__
from endplay.parsers import pbn
from endplay.types import Deal, Contract, Denom, Player, Penalty, Vul

sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))  # global
#sys.path.append(str(pathlib.Path.cwd().joinpath('mlBridgeLib')))  # global # Requires "./mlBridgeLib" be in extraPaths in .vscode/settings.json

import streamlitlib
from mlBridge.mlBridgePostmortemLib import PostmortemBase
from mlBridge.mlBridgeAugmentLib import (
    AllAugmentations,
)
from mlBridge.mlBridgeBWLib import BridgeWebResultsParser, read_pbn_file_from_url, merge_parsed_and_pbn_dfs


def get_db_connection():
    """Get or create a session-specific database connection.
    
    This ensures each Streamlit session has its own database connection,
    preventing concurrency issues when multiple users access the app.
    
    Returns:
        duckdb.DuckDBPyConnection: Session-specific database connection
    """
    if 'db_connection' not in st.session_state:
        # Create a new connection for this session
        st.session_state.db_connection = duckdb.connect()
        print(f"Created new database connection for session")
    return st.session_state.db_connection


def ShowDataFrameTable(df, key, query='SELECT * FROM self', show_sql_query=True):
    with st.session_state.main_section_container.container():
        if show_sql_query and st.session_state.show_sql_query:
            st.text(f"SQL Query: {query}")

        # if query doesn't contain 'FROM ', add 'FROM self ' to the beginning of the query. issue is for non-self tables such as exploded_auctions_df.
        # can't just check for startswith 'from self'. Not universal because 'from self' can appear in subqueries or after JOIN.
        # this syntax makes easy work of adding FROM but isn't compatible with polars SQL. duckdb only.
        if 'from ' not in query.lower():
            query = 'FROM self ' + query

        try:
            con = get_db_connection()
            result_df = con.execute(query).pl()
            if show_sql_query and st.session_state.show_sql_query:
                st.text(f"Result is a dataframe of {len(result_df)} rows.")
            streamlitlib.ShowDataFrameTable(result_df, key) # requires pandas dataframe.
        except Exception as e:
            st.error(f"duckdb exception: error:{e} query:{query}")
            return None
    
    return result_df


def app_info() -> None:
    """Display app information"""
    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Query engine is duckdb. Bridge lib is endplay. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita")
    st.caption(f"App:{st.session_state.app_datetime} Streamlit:{st.__version__} Query Params:{st.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}")
    st.caption(f"Python:{'.'.join(map(str, sys.version_info[:3]))} pandas:{pd.__version__} polars:{pl.__version__} endplay:{endplay.__version__}")
    return

from fsspec.utils import infer_storage_options

def get_url_protocol(path):
    # Use fsspec to infer the storage type
    options = infer_storage_options(path)
    # Check if the protocol is 'file', which indicates a local file
    return options['protocol']


def chat_input_on_submit():
    prompt = st.session_state.main_prompt_chat_input_key
    ShowDataFrameTable(st.session_state.df, query=prompt, key='user_query_main_doit_key')


def sample_count_on_change():
    st.session_state.single_dummy_sample_count = st.session_state.single_dummy_sample_count_number_input
    change_game_state()


def show_sql_query_change():
    # toggle whether to show sql query
    st.session_state.show_sql_query = st.session_state.sql_query_checkbox


def prepare_for_player_selection(results):
    """Lightweight processing to prepare for player selection."""
    parser_dfs = (
        results.get('north_south_results', pl.DataFrame()),
        results.get('east_west_results', pl.DataFrame()),
        results.get('tournament_info', pl.DataFrame())
    )
    st.session_state.parser_dfs = parser_dfs # IMPORTANT!!!: this line causes a hang after section selection. Appears to be a memory management disposal bug in polars.
    st.session_state.tournament_info = parser_dfs[2]
    
    combined_ns_ew_pairs = pl.concat([parser_dfs[0], parser_dfs[1]])
    st.session_state.combined_ns_ew_pairs = combined_ns_ew_pairs
    
    player_names_series = combined_ns_ew_pairs['players'].str.split('&').list.explode().str.strip_chars()
    st.session_state.player_names = player_names_series.unique().sort().to_list()


def section_selection_on_change():
    """Callback for when a user selects a new section."""
    try:
        # Clear the main container window when section changes
        st.session_state.main_section_container.empty()
        
        selected_section_key = st.session_state.get('selected_section_key')
        if not selected_section_key:
            return

        # Clear old player-specific and processed data
        st.session_state.player_names = None
        # Clear the selected player to reset the player selection widget
        if 'selected_player_key' in st.session_state:
            del st.session_state.selected_player_key
        st.session_state.df = None
        st.session_state.df_unfiltered = None
        
        # Clear report-related session state to prevent auto-rendering
        st.session_state.session_id = None
        st.session_state.player_name = None
        st.session_state.partner_name = None
        st.session_state.ScorePercent = None
        
        results = st.session_state.all_results[selected_section_key]
        # Store the msec value for the selected section for PBN file download
        st.session_state.selected_section_msec = results.get('msec', '1')
        prepare_for_player_selection(results)
        
    except Exception as e:
        with st.session_state.main_section_container.container():
            st.error(f"Error processing section selection: {str(e)}")
            st.text("Please try selecting a different section.")


def player_selection_on_change():
    #try:
        selected_player = st.session_state.get('selected_player_key')
        if not selected_player:
            return

        # --- DEFERRED HEAVY PROCESSING ---
        if 'df_unfiltered' not in st.session_state or st.session_state.df_unfiltered is None:
            with st.spinner('Downloading and processing deal data... This may take a moment.'):
                url = st.session_state.game_results_url
                parser_dfs = st.session_state.parser_dfs

                # Get the correct msec value for the selected section
                msec = st.session_state.get('selected_section_msec', '1')
                file_content = read_pbn_file_from_url(url, msec=msec)
                boards = pbn.loads(file_content)
                print(f"Parsed {len(boards)} boards from PBN file")
                for b in boards.copy():
                    if len(b.deal.to_pbn()) != 69:
                        st.error(f"Ignoring invalid Deal: Board:{b.board_num} {b.deal.to_pbn()}")
                        boards.remove(b)
                path_url = pathlib.Path(url)
                df = merge_parsed_and_pbn_dfs(path_url, boards, parser_dfs)
                
                augmenter = AllAugmentations(df, None, sd_productions=st.session_state.single_dummy_sample_count, progress=st.progress(0), lock_func=perform_hand_augmentations_queue)
                df, _ = augmenter.perform_all_augmentations()
                    
                assert df.select(pl.col(pl.Object)).is_empty(), f"Found Object columns: {[col for col, dtype in df.schema.items() if dtype == pl.Object]}"
                
                st.session_state.df_unfiltered = df
                st.session_state.session_id = parser_dfs[2]['results_session'].item()
                st.session_state.group_id = parser_dfs[2]['club'].item()
                
                # Register DataFrame with DuckDB
                con = get_db_connection()
                con.register(st.session_state.con_register_name, df)

        # --- END OF DEFERRED PROCESSING ---
        
        all_pairs = st.session_state.combined_ns_ew_pairs
        # warning: following statement failed on polars==1.31 with missing column "" [sic]. Had to downgrade to 1.30 and change requirements.txt file.
        pair_row_df = all_pairs.filter(
            pl.col('players').str.split('&').list.eval(pl.element().str.strip_chars()).list.contains(selected_player)
        ).head(1)

        if pair_row_df.is_empty():
            with st.session_state.main_section_container.container():
                st.error(f"Could not find pair for player: {selected_player}")
            return

        pair_row = pair_row_df.row(0, named=True)

        st.session_state.player_name = selected_player
        st.session_state.player_id = selected_player
        
        players_list = [p.strip() for p in pair_row['players'].split('&')]
        partner_index = 1 - players_list.index(selected_player)
        st.session_state.partner_name = players_list[partner_index]
        st.session_state.partner_id = st.session_state.partner_name

        st.session_state.pair_direction = 'NS' if pair_row['direction'].startswith('N') else 'EW'
        st.session_state.pair_id = pair_row['pair_number']

        if players_list.index(selected_player) == 0:
            st.session_state.player_direction = st.session_state.pair_direction[0]
            st.session_state.partner_direction = st.session_state.pair_direction[1]
        else:
            st.session_state.player_direction = st.session_state.pair_direction[1]
            st.session_state.partner_direction = st.session_state.pair_direction[0]

        st.session_state.opponent_pair_direction = 'EW' if st.session_state.pair_direction == 'NS' else 'NS'

        df_filtered = filter_dataframe(
            st.session_state.df_unfiltered,
            st.session_state.pair_direction,
            st.session_state.pair_id,
            st.session_state.player_direction,
            st.session_state.player_id,
            st.session_state.partner_direction,
            st.session_state.partner_id
        )
        
        st.session_state.df = df_filtered
        # Re-register the filtered dataframe for querying
        con = get_db_connection()
        con.register(st.session_state.con_register_name, st.session_state.df)

        # Set ScorePercent based on pair direction
        score_col = f"ScorePercent_{st.session_state.pair_direction}"
        if score_col in df_filtered.columns:
            st.session_state.ScorePercent = df_filtered.select(pl.col(score_col).first()).item()
        else:
            st.session_state.ScorePercent = None

        read_configs()
        
    # except Exception as e:
    #     with st.session_state.main_section_container.container():
    #         st.error(f"Error processing player selection: {str(e)}")
    #         st.text(f"Player: {st.session_state.get('selected_player_key', 'Unknown')}")
    #         st.text("Please try selecting a different player.")
    #         # Clear the problematic player selection
    #         if 'selected_player_key' in st.session_state:
    #             st.session_state.selected_player_key = None


def change_game_state():

    st.session_state.main_section_container.empty()

    # not working in this location but does work in write_report(). works well in ffbridge though so there's some subtle difference.
    # st.markdown('<div style="height: 50px;"><a name="top-of-report"></a></div>', unsafe_allow_html=True)

    try:
        with st.spinner(f'Preparing Bridge Game Postmortem Report...'):
            reset_game_data() # wipe out all game state data

            url = st.session_state.create_sidebar_text_input_url_key.strip()
            url = html.unescape(url)
            st.session_state.game_results_url = url
            with st.session_state.main_section_container.container():
                st.text(f"Selected URL: {url}")

            if url is None or url == '' or (get_url_protocol(url) == 'file' and ('/' in url and '\\' in url and '&' in url)):
                with st.session_state.main_section_container.container():
                    st.warning("Please enter a valid BridgeWebs URL.")
                return

            parser = BridgeWebResultsParser(url)
            all_results = parser.get_all_results()
            st.session_state.all_results = all_results

            if not all_results:
                with st.session_state.main_section_container.container():
                    st.error(f"Could not parse any results from the URL: {url}")
                    st.error(f"Please try a different URL. FYI, I only like to discuss mitchell movements.")
                    st.stop()
                return

            # If multiple sections are found, prompt user to select one first.
            if len(all_results) > 1:
                st.session_state.section_names = list(all_results.keys())
                # Clear player names to ensure player selector is not shown
                st.session_state.player_names = None
                # Reset sidebar widget states
                if 'selected_player_key' in st.session_state:
                    st.session_state.selected_player_key = None
                if 'selected_section_key' in st.session_state:
                    st.session_state.selected_section_key = None
                return
            else:
                # Otherwise, process the single section to get player names.
                first_section_key = next(iter(all_results))
                results = all_results[first_section_key]
                # Store the msec value for the single section
                st.session_state.selected_section_msec = results.get('msec', '1')
                prepare_for_player_selection(results)
                # Reset sidebar widget states for single section
                if 'selected_player_key' in st.session_state:
                    st.session_state.selected_player_key = None
                    
    except Exception as e:
        # Catch any unexpected errors and display them
        with st.session_state.main_section_container.container():
            st.error(f"An error occurred while processing the URL: {str(e)}")
            st.text(f"URL: {st.session_state.get('game_results_url', 'Unknown')}")
            st.text("Please try again or use a different URL.")
            # Log the full error for debugging
            import traceback
            st.text("Full error details:")
            st.code(traceback.format_exc())
    return


# this version of perform_hand_augmentations_locked() uses self for class compatibility, older versions did not.
def perform_hand_augmentations_queue(self, hand_augmentation_work):
    return streamlitlib.perform_queued_work(self, hand_augmentation_work, "Hand analysis")


def filter_dataframe(df, pair_direction, pair_number, player_direction, player_id, partner_direction, partner_id):

    df = df.with_columns(
        pl.col('PairId_'+pair_direction).eq(str(pair_number)).alias('Boards_I_Played')
    )

    df = df.with_columns(
        pl.col('Boards_I_Played').and_(pl.col('Declarer_Direction').eq(player_direction)).alias('Boards_I_Declared'),
        pl.col('Boards_I_Played').and_(pl.col('Declarer_Direction').eq(partner_direction)).alias('Boards_Partner_Declared'),
    )

    df = df.with_columns(
        pl.col('Boards_I_Played').alias('Boards_We_Played'),
        pl.col('Boards_I_Played').alias('Our_Boards'),
        (pl.col('Boards_I_Declared') | pl.col('Boards_Partner_Declared')).alias('Boards_We_Declared'),
    )

    df = df.with_columns(
        (pl.col('Boards_I_Played') & ~pl.col('Boards_We_Declared') & pl.col('Contract').ne('PASS')).alias('Boards_Opponent_Declared'),
    )

    return df


def create_sidebar():
    st.sidebar.caption('Build:'+st.session_state.app_datetime)

    # example valid urls
    #default_url = 'https://raw.githubusercontent.com/BSalita/Calculate_PBN_Results/master/DDS_Camrose24_1-%20BENCAM22%20v%20WBridge5.pbn'
    #default_url = 'file://c:/sw/bridge/ML-Contract-Bridge/src/Calculate_PBN_Results/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    #default_url = r'file://c:\sw/bridge\ML-Contract-Bridge\src\Calculate_PBN_Results/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    #default_url = r'file://DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    #default_url = 'https://raw.githubusercontent.com/BSalita/Calculate_PBN_Results/master/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    #default_url = 'DDS_Camrose24_1- BENCAM22 v WBridge5.pbn' #'1746217537-NzYzEYivsA@250502.PBN' # '3494191054-1682343601-bsalita.lin'
    default_url = 'https://www.bridgewebs.com/cgi-bin/bwoq/bw.cgi?pid=display_rank&event=20250526_1&club=irelandimps'
    #default_url = 'GIB-Thorvald-8638-2024-08-23.pbn'
    st.sidebar.text_input('Enter BridgeWebs URL:', default_url, on_change=change_game_state, key='create_sidebar_text_input_url_key', help='Enter a URL or pathless local file name.') # , on_change=change_game_state
    # using css to change button color for the entire button width. The color was choosen to match the the restrictive text colorizer (:green-background[Go]) used in st.info() below.
    css = """section[data-testid="stSidebar"] div.stButton button {
        background-color: rgba(33, 195, 84, 0.1);
        width: 50px;
        }"""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.sidebar.button('Go', on_click=change_game_state, key='create_sidebar_go_button_key', help='Load PBN data from URL.')

    # Let user select a section if there are multiple
    if 'section_names' in st.session_state and st.session_state.section_names:
        st.sidebar.selectbox(
            'Select Section:',
            options=st.session_state.section_names,
            index=None,
            key='selected_section_key',
            on_change=section_selection_on_change,
            placeholder="Select a section"
        )

    if st.session_state.get('player_names') is None:
        return

    if 'tournament_info' in st.session_state and not st.session_state.tournament_info.is_empty():
        club = st.session_state.tournament_info['club'].item() if 'club' in st.session_state.tournament_info.columns else "N/A"
        # grab event_title from title (best) or event column (if title is missing).
        st.session_state.event_title = st.session_state.tournament_info['title'].item() if 'title' in st.session_state.tournament_info.columns else st.session_state.tournament_info['event'].item() if 'event' in st.session_state.tournament_info.columns else "N/A"
        st.sidebar.info(f"**Club:** {club}\n\n**Event:** {st.session_state.event_title}")
        if 'game_results_url' in st.session_state and st.session_state.game_results_url:
            st.sidebar.markdown(f"**[Results Page]({st.session_state.game_results_url})**")

    st.sidebar.selectbox(
        'Select Player for Postmortem:',
        options=st.session_state.player_names,
        index=None,
        key='selected_player_key',
        on_change=player_selection_on_change,
        placeholder="Select a player"
    )

    st.session_state.pdf_link = st.sidebar.empty()

    with st.sidebar.expander('Developer Settings', False):
        # do not use .sidebar in expander. it's already in the sidebar.
        # SELECT Board, Vul, ParContract, ParScore_NS, Custom_ParContract FROM self
        st.checkbox('Show SQL Query',value=st.session_state.show_sql_query,key='sql_query_checkbox',on_change=show_sql_query_change,help='Show SQL used to query dataframes.')
        # These files are reloaded each time for development purposes. Only takes a second.
        # todo: put filenames into a .json or .toml file?
        st.session_state.single_dummy_sample_count = st.number_input('Single Dummy Sample Count',value=st.session_state.single_dummy_sample_count,key='single_dummy_sample_count_number_input',on_change=sample_count_on_change,min_value=1,max_value=1000,step=1,help='Number of random deals to generate for calculating single dummy probabilities. Larger number (10 to 30) is more accurate but slower. Use 1 to 5 for fast, less accurate results.')
        
        # Display params dictionary used in download_pbn_file
        st.text("BridgeWebs API Parameters:")
        params = {
            'pid': 'display_hands',
            'msec': '1',
            'event': 'extracted_from_url',
            'wd': '1',
            'club': 'extracted_from_url',
            'deal_format': 'pbn'
        }
        st.json(params)

    # Automated Postmortem Apps
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Automated Postmortem Apps**")
    st.sidebar.markdown("🔗 [ACBL Postmortem](https://acbl.postmortem.chat)")
    st.sidebar.markdown("🔗 [French ffbridge Postmortem](https://ffbridge.postmortem.chat)")
    #st.sidebar.markdown("🔗 [BridgeWebs Postmortem](https://bridgewebs.postmortem.chat)")
    
    return

# todo: put this in PBNResultsCalculator class?
def read_configs():

    st.session_state.default_favorites_file = pathlib.Path(
        'default.favorites.json')
    st.session_state.player_id_custom_favorites_file = pathlib.Path(
        f'favorites/{st.session_state.player_id}.favorites.json')
    st.session_state.debug_favorites_file = pathlib.Path(
        'favorites/debug.favorites.json')

    if st.session_state.default_favorites_file.exists():
        with open(st.session_state.default_favorites_file, 'r') as f:
            favorites = json.load(f)
        st.session_state.favorites = favorites
        #st.session_state.vetted_prompts = get_vetted_prompts_from_favorites(favorites)
    else:
        st.session_state.favorites = None

    if st.session_state.player_id_custom_favorites_file.exists():
        with open(st.session_state.player_id_custom_favorites_file, 'r') as f:
            player_id_favorites = json.load(f)
        st.session_state.player_id_favorites = player_id_favorites
    else:
        st.session_state.player_id_favorites = None

    if st.session_state.debug_favorites_file.exists():
        with open(st.session_state.debug_favorites_file, 'r') as f:
            debug_favorites = json.load(f)
        st.session_state.debug_favorites = debug_favorites
    else:
        st.session_state.debug_favorites = None

    # display missing prompts in favorites
    if 'missing_in_summarize' not in st.session_state:
        # Get the prompts from both locations
        summarize_prompts = st.session_state.favorites['Buttons']['Summarize']['prompts']
        vetted_prompts = st.session_state.favorites['SelectBoxes']['Vetted_Prompts']

        # Process the keys to ignore leading '@'
        st.session_state.summarize_keys = {p.lstrip('@') for p in summarize_prompts}
        st.session_state.vetted_keys = set(vetted_prompts.keys())

        # Find items in summarize_prompts but not in vetted_prompts. There should be none.
        st.session_state.missing_in_vetted = st.session_state.summarize_keys - st.session_state.vetted_keys
        assert len(st.session_state.missing_in_vetted) == 0, f"Oops. {st.session_state.missing_in_vetted} not in {st.session_state.vetted_keys}."

        # Find items in vetted_prompts but not in summarize_prompts. ok if there's some missing.
        st.session_state.missing_in_summarize = st.session_state.vetted_keys - st.session_state.summarize_keys

        print("\nItems in Vetted_Prompts but not in Summarize.prompts:")
        for item in st.session_state.missing_in_summarize:
            print(f"- {item}: {vetted_prompts[item]['title']}")
    return


# todo: use this similar to bridge_game_postmortem_streamlit.py
def reset_game_data():

    # Default values for session state variables
    reset_defaults = {
        'event_title_default': None,
        'group_id_default': None,
        'session_id_default': None,
        'section_name_default': None,
        'player_id_default': None,
        'partner_id_default': None,
        'player_name_default': None,
        'partner_name_default': None,
        'player_direction_default': None,
        'partner_direction_default': None,
        'pair_id_default': None,
        'pair_direction_default': None,
        'opponent_pair_direction_default': None,
        'player_names_default': None,
        'section_names_default': None,
    }
    
    # Initialize default values if not already set
    for key, value in reset_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize additional session state variables that depend on defaults.
    reset_session_vars = {
        'df': None,
        'df_unfiltered': None,
        'event_title': st.session_state.event_title_default,
        'group_id': st.session_state.group_id_default,
        'session_id': st.session_state.session_id_default,
        'section_name': st.session_state.section_name_default,
        'player_id': st.session_state.player_id_default,
        'partner_id': st.session_state.partner_id_default,
        'player_name': st.session_state.player_name_default,
        'partner_name': st.session_state.partner_name_default,
        'player_direction': st.session_state.player_direction_default,
        'partner_direction': st.session_state.partner_direction_default,
        'pair_id': st.session_state.pair_id_default,
        'pair_direction': st.session_state.pair_direction_default,
        'opponent_pair_direction': st.session_state.opponent_pair_direction_default,
        'player_names': st.session_state.player_names_default,
        'section_names': st.session_state.section_names_default,
        'analysis_started': False,   # new flag for analysis sidebar rewrite
        'vetted_prompts': [],
        'pdf_assets': [],
        'sql_query_mode': False,
        'sql_queries': [],
        'game_urls_d': {},
        'tournament_session_urls_d': {},
        'current_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # Clear data that could affect display
        'all_results': None,
        'parser_dfs': None,
        'tournament_info': None,
        'combined_ns_ew_pairs': None,
        'favorites': None,
        'ScorePercent': None,
        'selected_section_msec': '1',
    }
    
    # Actually reset the session state variables to clear previous game data
    for key, value in reset_session_vars.items():
        st.session_state[key] = value

    return


def initialize_website_specific():

    st.session_state.assistant_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/master/assets/logo_assistant.gif?raw=true' # 🥸 todo: put into config. must have raw=true for github url.
    st.session_state.guru_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/master/assets/logo_guru.png?raw=true' # 🥷todo: put into config file. must have raw=true for github url.
    st.session_state.game_results_url_default = None
    st.session_state.game_name = 'bridgewebs'
    st.session_state.game_results_url = st.session_state.game_results_url_default
    # todo: put filenames into a .json or .toml file?
    st.session_state.rootPath = pathlib.Path('e:/bridge/data')
    #st.session_state.acblPath = st.session_state.rootPath.joinpath('acbl')
    #st.session_state.favoritesPath = pathlib.joinpath('favorites'),
    st.session_state.savedModelsPath = st.session_state.rootPath.joinpath('SavedModels')

    streamlit_chat.message(
        f"Hi. I'm Morty. Your friendly postmortem chatbot. I only want to chat about {st.session_state.game_name} pair matchpoint games using a Mitchell movement and not shuffled.", key='intro_message_1', logo=st.session_state.assistant_logo)
    streamlit_chat.message(
        "I'm optimized for large screen devices such as a notebook or monitor. Do not use a smartphone.", key='intro_message_2', logo=st.session_state.assistant_logo)
    #streamlit_chat.message(
    #    f"To start our postmortem chat, I'll need an {st.session_state.game_name} player number. I'll use it to find player's latest {st.session_state.game_name} club game. It will be the subject of our chat.", key='intro_message_3', logo=st.session_state.assistant_logo)
    streamlit_chat.message(
        f"To start our postmortem chat, I'll need a BridgeWebs URL. It will be the subject of our chat.", key='intro_message_3', logo=st.session_state.assistant_logo)
    #streamlit_chat.message(
    #    f"Enter any {st.session_state.game_name} player number in the left sidebar.", key='intro_message_4', logo=st.session_state.assistant_logo)
    streamlit_chat.message(
        "I'm just a Proof of Concept so don't double me.", key='intro_message_5', logo=st.session_state.assistant_logo)
    app_info()
    return


# todo: this class should be universal. its methods should initialize generic values followed by call outs to app-specific methods.
class PBNResultsCalculator(PostmortemBase):
    """PBN Results Calculator Streamlit application."""
    
    def __init__(self):
        super().__init__()
        # App-specific initialization
    
    # App-specific methods
    def parse_pbn_file(self, pbn_content):
        """Parse PBN file content."""
        # Implementation for parsing PBN files
        pass
    
    def calculate_results(self, pbn_data):
        """Calculate results from PBN data."""
        # Implementation for calculating results
        pass
    
    def export_results(self, results, format="csv"):
        """Export results in various formats."""
        # Implementation for exporting results
        pass
    
    def file_uploader_callback(self):
        """Handle file upload events."""
        # Implementation
        pass
    
    # Override abstract methods
    def initialize_session_state(self):
        """Initialize app-specific session state."""
        # First initialize common session state
        self.initialize_common_session_state()
        
        # BridgeWebs-specific session state
        bridgewebs_specific_vars = {
            'pbn_file': None,
            'pbn_data': None,
            'results': None,
            'player_names': None,
            'recommended_board_max': 100,
            'save_intermediate_files': False,
        }
        
        for key, value in bridgewebs_specific_vars.items():
            if key not in st.session_state:
                st.session_state[key] = value

        if 'player_id' in st.query_params:
            player_id = st.query_params['player_id']
            if not isinstance(player_id, str):
                st.error(f'player_id must be a string {player_id}')
                st.stop()
            st.session_state.player_id = player_id
        else:
            st.session_state.player_id = None

        self.reset_game_data()
        self.initialize_website_specific()
        return


    def reset_game_data(self):
        """Reset game data."""
        # First reset common game data
        self.reset_common_game_data()
        # BridgeWebs-specific reset handled by global function for now

    def initialize_website_specific(self):
        """Initialize app-specific components."""
        # Implementation
        initialize_website_specific()
    
    def process_prompt_macros(self, sql_query):
        """Process app-specific prompt macros."""
        # First process standard macros
        sql_query = super().process_prompt_macros(sql_query)
        # Then app-specific macros
        # Implementation
        return sql_query
    
    # Customize standard methods as needed
    def create_sidebar(self):
        """Create app-specific sidebar."""
        # Call global function for BridgeWebs-specific sidebar
        create_sidebar()

    def create_ui(self):
        """Creates the main UI structure with player selection prompts."""
        self.create_sidebar()
        if not st.session_state.sql_query_mode:
            # Check if we should show player selection prompt
            if (st.session_state.get('player_names') is not None and 
                len(st.session_state.player_names) > 0 and 
                st.session_state.get('selected_player_key') is None and
                st.session_state.session_id is None):
                
                # Show Morty message prompting user to select a player
                with st.session_state.main_section_container.container():
                    import streamlit_chat
                    streamlit_chat.message(
                        "Please select the player name from Player for Postmortem selection box in the left sidebar. It will be used to generate a personalized postmortem report.",
                        key='morty_select_player_prompt',
                        logo=st.session_state.assistant_logo
                    )
            
            # Show report if player is selected
            if st.session_state.session_id is not None:
                self.write_report()
        self.ask_sql_query()

    def create_main_content(self):
        """Create app-specific main content."""
        # Implementation
        st.title("PBN Results Calculator")
        
        # File upload section
        st.header("Upload PBN File")
        uploaded_file = st.file_uploader("Choose a PBN file", type="pbn", on_change=self.file_uploader_callback)
        
        if st.session_state.pbn_data is not None:
            # Display PBN data
            st.header("PBN Data")
            st.dataframe(st.session_state.pbn_data, width='stretch')
            
            # Calculate button
            if st.button("Calculate Results"):
                results = self.calculate_results(st.session_state.pbn_data)
                st.session_state.results = results
        
        if st.session_state.results is not None:
            # Results section
            st.header("Results")
            self.ShowDataFrameTable(
                st.session_state.results, 
                "results_table",
                show_sql_query=False
            )
            
            # Export options
            st.header("Export")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export as CSV"):
                    self.export_results(st.session_state.results, "csv")
            with col2:
                if st.button("Export as PDF"):
                    self.export_results(st.session_state.results, "pdf")

    def write_report(self):
        """Use the standard report generation from base class with scroll to top."""
        # Scroll to top of the page before rendering the report
        st.markdown("""
            <script>
                setTimeout(function() {
                    window.scrollTo({top: 0, behavior: 'smooth'});
                }, 50);
            </script>
        """, unsafe_allow_html=True)
        
        self.write_standard_report()

    def ask_sql_query(self):
        """Use the standard SQL query interface from base class."""
        self.ask_standard_sql_query()


if __name__ == "__main__":
    if 'app' not in st.session_state:
        st.session_state.app = PBNResultsCalculator()
    st.session_state.app.main() 