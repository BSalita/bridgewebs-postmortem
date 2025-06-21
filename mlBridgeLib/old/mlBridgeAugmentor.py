#!/usr/bin/env python
# coding: utf-8

# Create additional columns in dataframe.
# Per specified club directories, reads 'club.games.pandas.pkl' and outputs 'featured_bdf.pkl'.

# previous steps:
# TheCommonGamePickleToPandas.py created 'club.games.pandas.pkl'.

# next steps:
# mlBridgeTrain.py reads 'featured_bdf.pkl' and outputs trained model results files.
# todo: why does Create-TCG-Standardized-Hand-Records.ipynb use 'club.games.pandas.pkl' instead of 'featured_bdf.pkl'?
# Create-TCG-Standardized-Hand-Records.ipynb reads 'club.games.pandas.pkl' and creates 'standardized_tcg_hand_records_df.pkl'.
# merged-standardized-hand-record-dfs.ipynb merges acbl and tcg standardized dfs into 'merged_standardized_hand_records.pkl'.

# takes 180s/208s for 108571 with/without-multiprocessing
# average 1500s single thread, 450s for MP -- 3x speedup using MP
# 1100s for giant tcg_boards_df.pkl

# todo:
# change some previous tcg web page processing step to output hands using dict of suits ('SHDC') to avoid confusion?
# DDmakes() is too slow.
# 'HCP_*' and 'Hand_P_' are redundant. Remove HCP?
# HandMaxSuit_Tricks is miscounting for *Rank.html: EventBoard:191221E_02 HandMaxSuit_Tricks:10 HandMaxSuit_L:15 LoTT_Tricks:15 LoTT_L:15 LoTT_V:0


import matplotlib.pyplot as plt
import TheCommonGameOptionsLib

import pandas as pd
import ast
from collections import defaultdict
import pathlib
import pickle
from mlBridgeLib.mlBridgeLib import *

import argparse
import multiprocessing
import time

from IPython.display import display  # needed for VSCode


def Vul(df):
    for direction in ['NS', 'EW']:
        AssignToColumn(df, 'Vul_'+direction,
                       (df['Vul'] == '_'.join(direction)) | (df['Vul'] == 'Both'))  # ,'int8')


def Dealer(df):
    df['Dealer'] = [BoardNumberToDealer(
        int(r['EventBoard'].split('_')[1])) for i, r in df.iterrows()]


def DDmakes(df, ddmakesd):
    #print(type(df['DDmakes'][0]), df['DDmakes'][0])
    #print(type(ddmakesd[0]), ddmakesd[0])
    nsewd = defaultdict(list)
    for dd in ddmakesd:
        for dk, d in dd.items():
            for ddk, dd in zip(list('CDHSN'), d):
                # display(ddk,dd)
                nsewd[(ddk, dk)].append(dd)

    prefix = 'DD'
    for nsewk, nsewl in nsewd.items():
        suit, direction = nsewk
        AssignToColumn(df, MakeColName(prefix, 'Tricks',
                                       suit, direction), nsewl, 'int8')
        # note: dep_var (ground truth) is itself
        dep_vars = MakeColName(prefix, 'Tricks', suit,
                               direction), None, suit, direction, ''
        InsertScoringColumnsTricks(df, dep_vars, prefix)

    for d in ['NS', 'EW']:
        for suit in 'CDHSN':
            l = [MakeColName(prefix, 'Tricks', suit, dd) for dd in d]
            AssignToColumn(df, MakeColName(prefix, 'Tricks',
                                           suit, d), df[l].max(axis=1), 'int8')  # was bdf


def HCP(df, hcpld):
    nsewd = defaultdict(list)
    for hcps in hcpld:
        for hk, h in hcps.items():
            nsewd['HCP_'+hk].append(h)
    for nsewk, nsewl in nsewd.items():
        AssignToColumn(df, nsewk, nsewl, 'int8')  # was bdf
    df['HCP_EW'] = df['HCP_E']+df['HCP_W']  # was bdf
    df['HCP_NS'] = df['HCP_N']+df['HCP_S']  # was bdf
    assert ((df['HCP_NS']+df['HCP_EW']) == 40).all()  # was bdf


def LoTT(bdf, lottl):
    nsewd = defaultdict(list)
    for lott in lottl:
        # 'LoTT_DD' is combined trick count, 'LoTT_L' is combined length, 'LoTT_V' is variance (diff)
        for dk, d in zip(['Tricks', 'L', 'V'], lott):
            nsewd['LoTT_'+dk].append(d)
    for nsewk, nsewl in nsewd.items():
        AssignToColumn(bdf, nsewk, nsewl, 'int8')


# bdf MUST HAVE INDEX in ORDER
# Hands is about suits, not NT
# For each hand, calculate binary representation of cards, length of suit, quick tricks
def Hands(bdf, hands):
    nsewd = defaultdict(list)
    # binary representation of 'AK..32' weights
    w = [2**i for i in reversed(range(0, 13))]
    # print(w)
    for handk, hand in hands.items():
        # display(handk,hand)
        for dk, d in hand.items():
            # display(dk,d)
            # using 'SHDC' because suits are presented in that order. Must not use 'CDHS'
            for sk, s in zip('SHDC', d):
                nsewd[MakeColName('Hand', 'C', sk, dk)].append(s)
                b = sum([w['AKQJT98765432'.find(c)]
                         for c in s])  # create binary representation
                #print(f'{s} {b:013b}')
                nsewd[MakeColName('Hand', 'B', sk, dk)].append(b)
                nsewd[MakeColName('Hand', 'L', sk, dk)].append(len(s))
                hcp = sum(['JQKA'.find(c)+1 for c in s])
                nsewd[MakeColName('Hand', 'P', sk, dk)].append(hcp)
                if s.startswith('AK'):
                    qt = 2
                elif s.startswith('AQ'):
                    qt = 1.5
                elif s.startswith('A'):
                    qt = 1
                elif s.startswith('KQ'):
                    qt = 1
                elif s.startswith('K'):
                    qt = .5
                else:
                    qt = 0
                nsewd[MakeColName('Hand', 'Q', sk, dk)].append(qt)
    for nsewk, nsewl in nsewd.items():
        blpq = nsewk[-5]  # extract blq from 'Hand_[BLPQ]_[CDHS]_[NSEW]'
        if blpq == 'B':
            AssignToColumn(bdf, nsewk, nsewl, 'int16')
        elif blpq == 'L':
            AssignToColumn(bdf, nsewk, nsewl, 'int8')
        elif blpq == 'P':
            AssignToColumn(bdf, nsewk, nsewl, 'int8')
        elif blpq == 'Q':
            AssignToColumn(bdf, nsewk, nsewl, 'int8')
    tp = 0
    for d in ['NS', 'EW']:
        hp = hq = 0  # total quick tricks in a direction
        for sk in 'CDHS':
            AssignToColumn(bdf, MakeColName('Hand', 'B', sk, d), (bdf[MakeColName(
                'Hand', 'B', sk, d[0])]+bdf[MakeColName('Hand', 'B', sk, d[1])]), 'int16')
            AssignToColumn(bdf, MakeColName('Hand', 'L', sk, d), (bdf[MakeColName(
                'Hand', 'L', sk, d[0])]+bdf[MakeColName('Hand', 'L', sk, d[1])]), 'int8')
            AssignToColumn(bdf, MakeColName('Hand', 'P', sk, d), (bdf[MakeColName(
                'Hand', 'P', sk, d[0])]+bdf[MakeColName('Hand', 'P', sk, d[1])]), 'int8')
            AssignToColumn(bdf, MakeColName('Hand', 'Q', sk, d), (bdf[MakeColName(
                'Hand', 'Q', sk, d[0])]+bdf[MakeColName('Hand', 'Q', sk, d[1])]), 'int8')
            hp += bdf[MakeColName('Hand', 'P', sk, d)]
            hq += bdf[MakeColName('Hand', 'Q', sk, d)]
        AssignToColumn(bdf, MakeColName('Hand', 'P', '', d), hp, 'int8')
        AssignToColumn(bdf, MakeColName('Hand', 'Q', '', d), hq, 'int8')
        tp += hp
    assert (tp == 40).all(), 'Hands: Total HCP points ({tp}) <> 40'


def HandsMax(bdf):
    prefix = 'HandMaxSuit'
    for d in ['NS', 'EW']:
        # Note: Example of technique to sort groups of columns. Less code than using idxmax().
        # returns list of tuple (suit length, number of tricks, longest suit col)
        ms = bdf.apply(lambda r: sorted([(r[MakeColName('Hand', 'L', suit, d)], r[MakeColName(
            'DD', 'Tricks', suit, d)], MakeColName('Hand', 'L', suit, d)) for suit in 'CDHS'], reverse=True)[0], axis=1)
        suits = [t[2][-4] for t in ms]  # extract suit letter
        AssignToColumn(bdf, MakeColName(prefix, 'Suit', '', d),
                       suits, 'string')  # assign longest suit to each row
        maxlengths = [t[0] for t in ms]  # get longest suit lengths
        # assign longest suit lengths to each row
        AssignToColumn(bdf, MakeColName(
            prefix, 'L', '', d), maxlengths, 'int8')
        maxtricks = [t[1] for t in ms]
        AssignToColumn(bdf, MakeColName(
            prefix, 'Tricks', '', d), maxtricks, 'int8')
    AssignToColumn(bdf, MakeColName(prefix, 'L', '', ''), (bdf[MakeColName(
        prefix, 'L', '', 'NS')]+bdf[MakeColName(prefix, 'L', '', 'EW')]), 'int8')
    AssignToColumn(bdf, MakeColName(prefix, 'Tricks', '', ''), (bdf[MakeColName(
        prefix, 'Tricks', '', 'NS')]+bdf[MakeColName(prefix, 'Tricks', '', 'EW')]), 'int8')
    # will assert if LoTT is wrong in source data -- hasn't happened so far.
    for i, r in bdf.iterrows():
        if r[MakeColName(prefix, 'Tricks', '', '')] != r[MakeColName('LoTT', 'Tricks', '', '')]:
            print(f"EventBoard:{r['EventBoard']} HandMaxSuit_Tricks:{r[MakeColName(prefix,'Tricks','','')]} HandMaxSuit_L:{r[MakeColName(prefix,'L','','')]} LoTT_Tricks:{r[MakeColName('LoTT','Tricks','','')]} LoTT_L:{r[MakeColName('LoTT','L','','')]} LoTT_V:{r[MakeColName('LoTT','V','','')]}")
        if r[MakeColName(prefix, 'L', '', '')] != r[MakeColName('LoTT', 'L', '', '')]:
            print(f"EventBoard:{r['EventBoard']} HandMaxSuit_Tricks:{r[MakeColName(prefix,'Tricks','','')]} HandMaxSuit_L:{r[MakeColName(prefix,'L','','')]} LoTT_Tricks:{r[MakeColName('LoTT','Tricks','','')]} LoTT_L:{r[MakeColName('LoTT','L','','')]} LoTT_V:{r[MakeColName('LoTT','V','','')]}")
# temp? Can't always reconstruct *Rank.html trick count.
#    assert (bdf[MakeColName(prefix, 'Tricks', '', '')] ==
#            bdf[MakeColName('LoTT', 'Tricks', '', '')]).all()
    assert (bdf[MakeColName(prefix, 'L', '', '')] ==
            bdf[MakeColName('LoTT', 'L', '', '')]).all()


# 1) why is Par needed at all? Shouldn't DD be even better because it has direction?
#    Ah, problem with DD is it doesn't have complete info on Nash equilib -- no sets, which are always doubled, only good for max makes
# 2) Par has own Score for open vs 299
# 3) Aren't Score, TCG_Score and TCG299_Score redundant?
# todo: rename bdf to df
def Par(bdf, parl, tcgd, tcg299d):
    nsewd = defaultdict(list)
    for pars in parl:
        nsewd['Par_'+'Score'].append(pars[0])
        # display(pars[1],len(pars[1]))
        for par in pars[1]:
            if pars[0] == 0:  # passed out e.g. clubNumber 116798
                # todo: workaround for missing items in pars[1] due to Passed Out being improperly composed in previous step
                par = [0, '', '', 0]
            for lbrk, lbr in zip(['Level', 'Suit', 'Double', 'Result'], par):
                nsewd['Par_'+lbrk].append(lbr)
            # todo: problem: there can be several contracts that yield identical par scores. Taking only the first.
            break
    print(nsewd.keys(), [(dk, len(d)) for dk, d in nsewd.items()])
    for nsewk, nsewl in nsewd.items():
        if nsewk == 'Par_Level' or nsewk == 'Par_Result':
            AssignToColumn(bdf, nsewk, nsewl, 'int8')
        elif nsewk == 'ParScore':
            AssignToColumn(bdf, nsewk, nsewl, 'int16')
        elif nsewk == 'Par_Double' or nsewk == 'Par_Suit':
            AssignToColumn(bdf, nsewk, nsewl, 'string')
        else:
            assert False, f"Unexpected Par item:{nsewk}"

    AssignToColumnLoc(bdf, ((bdf['ParScore'] > 0) & (bdf['Par_Result'] >= 0)) | (
        (bdf['ParScore'] < 0) & (bdf['Par_Result'] < 0)), 'Par_Dir', 'NS', 'string')
    AssignToColumnLoc(bdf, ((bdf['ParScore'] < 0) & (bdf['Par_Result'] >= 0)) | (
        (bdf['Par_Score'] > 0) & (bdf['Par_Result'] < 0)), 'Par_Dir', 'EW', 'string')
    #AssignToColumnLoc(bdf, ~bdf['Par_Dir'].isin(['NS', 'EW']), 'Par_Dir', '', 'string')
    bhcp = bdf['Par_Dir'].notna()  # Omit pd.NA in Par_Dir
    AssignToColumnLoc(bdf, bhcp, 'Par_HCP', bdf.loc[bhcp].apply(
        lambda r: r['HCP_'+r['Par_Dir']], axis=1), 'Int8')  # Must use bhcp twice. Par_HCP is pd.NA for Passed Out. Change 'HCP_' to 'Hand_P'?
    # todo: obsolete double in dep_vars?
    dep_vars = MakeColName('Par', 'Level', '', ''), None, '', '', ''
    InsertScoringColumnsPar(bdf, dep_vars, 'Par')
    # todo: aren't there two scores for tcgd and tcg299d?
    InsertTcgColumns(bdf, dep_vars, 'Par', tcgd, tcg299d)


def CreateBoardDataFeaturesMp(args):
    clubNumber, pickleFileToProcess, featuredFileToProcess, tcgd, tcg299d = args
    return CreateBoardDataFeatures(None, clubNumber, pickleFileToProcess, featuredFileToProcess, tcgd, tcg299d)


def CreateBoardDataFeatures(pool, clubNumber, pickleFileToProcess, featuredFileToProcess, tcgd, tcg299d):
    if not pool is None:
        return pool.apply_async(CreateBoardDataFeaturesMp, [[clubNumber, pickleFileToProcess, featuredFileToProcess, tcgd, tcg299d]])

    startTime = time.time()
    print(
        f'{time.strftime("%X")}: {__file__}: CreateBoardDataFeatures: Begin: ClubNumber: {clubNumber}: Total elapsed seconds: {time.time()-startTime:.2f}')

    if (not featuredFileToProcess.exists()) or (pickleFileToProcess.stat().st_mtime > featuredFileToProcess.stat().st_mtime):  # temp for debugging?

        print(pickleFileToProcess.stat())

        with open(pickleFileToProcess, 'rb') as f:
            ddf = pickle.load(f)
        #ddf = pd.read_pickle(pickleFileToProcess)
        display([[k, type(v)] for k, v in ddf.items()])

        for dfKey, df in ddf.items():
            display(dfKey, df.dtypes)
            display(df.head())

        rdf = ddf['resultsdf']
        display(rdf.head(50))

        bdf = ddf['boardDetaildf']
        bdf.reset_index(drop=True, inplace=True)  # temp

        CreateBoardDataFeaturesFromBDF(bdf, clubNumber, tcgd, tcg299d)

        bdf.to_pickle(featuredFileToProcess)

    print(f'{time.strftime("%X")}: {__file__}: CreateBoardDataFeatures: Done: ClubNumber: {clubNumber}: Total elapsed time: {time.time()-startTime:.2f}')


def CreateBoardDataFeaturesFromBDF(bdf, clubNumber, tcgd, tcg299d):
    # change to string extension type or leave alone? longterm it's probably a good idea.
    # shortterm, there's small bugs in value_count, polyfit and plt that require changing to non-extension types
    if 'Key' in bdf.columns:  # won't exist for Rank.html data
        bdf['Key'] = bdf['Key'].astype('string')
    bdf['EventBoard'] = bdf['EventBoard'].astype('string')
    # content rendering problem if https:// used for tcgcloud.bridgefinesse.com/Results/ is used so using http instead
    bdf['TCG_Link'] = 'http://tcgcloud.bridgefinesse.com/Results/20' + \
        bdf['EventBoard'].str[0:6]+'Rank.html#tab_board_' + \
        bdf['Board'].str.lstrip('0')
    bdf['TCG_Link'] = bdf['TCG_Link'].astype('string')
    # however, for tcgcloud.bridgefinesse.com/ClubWebHost/ https:// works ok
    if clubNumber is not None:
        bdf['Club_Link'] = 'https://tcgcloud.bridgefinesse.com/ClubWebHost/'+clubNumber.name + \
            '/'+bdf['EventBoard'].str[0:-3] + \
            '.html#board_results'+bdf['Board'].str.lstrip('0')
    if 'Club_Link' in bdf.columns:
        bdf['Club_Link'] = bdf['Club_Link'].astype('string')
    bdf['Board'] = bdf['Board'].astype('string')
    bdf['DDmakes'] = bdf['DDmakes'].astype('string')
    bdf['Dealer'] = bdf['Dealer'].astype('string')
    bdf['HCP'] = bdf['HCP'].astype('string')
    bdf['Hands'] = bdf['Hands'].astype('string')
    bdf['LoTT'] = bdf['LoTT'].astype('string')
    bdf['Par'] = bdf['Par'].astype('string')
    bdf['Vul'] = bdf['Vul'].astype('string')
    # sanity check for proper indexing. Needed in 'Hand' processing.
    idx = bdf.index
    # display(idx)
    bdf.reset_index(drop=True, inplace=True)
    # display(bdf.index)
    assert (idx == bdf.index).all()
    # display(bdf[bdf['Key'].str.startswith('2')])
    # display(bdf.head())
    # display(bdf.dtypes)

    Vul(bdf)
    # display(bdf.head())

    #print([BoardNumberToVul(bn+1) for bn in range(32)])

    # display('EventBoard board number != Board number', [(r['EventBoard'], r['Board'], r['Vul_NS'], r['Vul_EW'], BoardNumberToVul(
    #    int(r['Board']))) for i, r in bdf.iterrows() if r['Board'] != r['EventBoard'].split('_')[1]])
    # display('EventBoard board number vulnerability != calculated vulnerability', [(r['EventBoard'], r['Board'], r['Vul_NS'], r['Vul_EW'], BoardNumberToVul(
    #    int(r['EventBoard'].split('_')[1]))) for i, r in bdf.iterrows() if BoardNumberToVul(int(r['EventBoard'].split('_')[1])) != (r['Vul_NS']+r['Vul_EW']*2)])

    #print([BoardNumberToDealer(bn+1) for bn in range(32)])

    # Dealer(bdf)
    #print(['NESW'[(int(r['EventBoard'].split('_')[1])-1)&3] for i,r in bdf.iterrows()])
    for i, r in bdf.iterrows():
        if r['Dealer'] != 'NESW'[(int(r['EventBoard'].split('_')[1])-1) & 3]:
            print(f"Invalid declarer: {r['EventBoard']}")
#    assert sum([r['Dealer'] != 'NESW'[(int(r['EventBoard'].split('_')[1])-1) & 3]
#                for i, r in bdf.iterrows()]) == 0
    # print(bdf.columns)
    #assert 'Dealer' not in bdf.columns
    #display(bdf[['EventBoard', 'Dealer']])
    # error

    ddmakesd = bdf['DDmakes'].map(ast.literal_eval)
    DDmakes(bdf, ddmakesd)
    # display(bdf.head())

    hcpld = bdf['HCP'].map(ast.literal_eval)
    HCP(bdf, hcpld)
    # print(bdf.dtypes)
    # display(bdf.head(3))

    lottl = bdf['LoTT'].map(ast.literal_eval)
    LoTT(bdf, lottl)
    # print(bdf.dtypes)
    # display(bdf.head(3))

    hands = bdf['Hands'].map(ast.literal_eval)
    Hands(bdf, hands)
    # print(bdf.columns)
    # print(bdf.dtypes)
    # display(bdf.head(1))

    HandsMax(bdf)
    # print(bdf.columns)
    # print(bdf.dtypes)
    # display(bdf.head(2))

    # display(bdf[[MakeColName('DD', 'Tricks', suit, direction)
    #             for direction in ['NS', 'EW'] for suit in 'CDHS']].head(20))
    # display(bdf[[MakeColName('Hand', 'L', suit, direction)
    #             for direction in ['NS', 'EW'] for suit in 'CDHS']].head(20))

    # display(bdf[[MakeColName('HandMaxSuit', c, '', direction) for direction in [
    #    'NS', 'EW'] for c in ['Suit', 'L', 'Tricks']]].head(20))

    # display(bdf[[MakeColName('HandMaxSuit', c, '', '') for c in ['Tricks', 'L']] +
    #    [MakeColName('LoTT', c, '', '') for c in ['Tricks', 'L', 'V']]].head(20))

    parl = bdf['Par'].map(ast.literal_eval)
    Par(bdf, parl, tcgd, tcg299d)

    # display(bdf.head(4))
    # display(bdf.dtypes)


def main(args):
    print(f'{time.strftime("%X")}: {__file__}: main: Augment Brige Data for Machine Learning')
    startTime = time.time()

    options = TheCommonGameOptionsLib.getOptions(args)

    # Initial testing shows multiprocessing causes an 80% speed up on my notebook. Using apply_async() gains another 10%
    # turns on multiprocessing. flip on/off to test performance difference. Default is on.
    useMultiprocessing = options.multiprocess
    #useMultiprocessing = False  # temp for debugging
    pool = None
    nCPUs = options.cpus

    rootPathStr = '../../.' # options.path
    rootPath = pathlib.Path(rootPathStr)
    clubsPath = rootPath.joinpath('clubs')

    clubNumbers = options.clubNumbers
    clubNumbers = [clubsPath.joinpath('108571')]  # temp for debugging '275966'
    forceRewriteOfOutputFiles = options.forcerewrite
    deleteOutputFiles = options.delete

    print(f'{time.strftime("%X")}: {__file__}: main: Processing clubs:{clubNumbers}')

    forceRewriteOfOutputFiles = deleteOutputFiles = False  # temp for debugging

    inputFiles = ['club.games.pandas.pkl']
    outputFiles = ['featured_bdf.pkl']
    listOfClubs = ListOfClubsToProcess(
        clubNumbers, inputFiles, outputFiles, clubsPath, forceRewriteOfOutputFiles, deleteOutputFiles, reverse=useMultiprocessing)

    tcgd = pd.read_pickle(rootPath.joinpath('tcg_results_dict.pkl'))
    display(list(tcgd.items())[:10])

    tcg299d = pd.read_pickle(rootPath.joinpath('tcg_results299_dict.pkl'))
    display(list(tcg299d.items())[:10])

    # todo: incomplete attempt to make global df across all clubs.
    #pickleFileToProcess = rootPath.joinpath('tcg_boards_df.pkl')
    #featuredFileToProcess = rootPath.joinpath('featured_bdf.pkl')
    #clubNumber = None  # todo: change to clubNumberName like other mlBridge files -- clubNumberName = clubNumber.name
    #CreateBoardDataFeatures(
    #    pool, clubNumber, pickleFileToProcess, featuredFileToProcess)
    #assert False  # temp

    if useMultiprocessing:
        pool = multiprocessing.Pool(processes=nCPUs)

    poolStartTime = time.time()

    for loc in listOfClubs:
        clubNumber = loc[1]
        clubDir = loc[2]
        pickleFileToProcessFile = loc[3][0]
        featuredFileToProcessFile = loc[4][0]

        pickleFileToProcess = clubDir.joinpath(pickleFileToProcessFile)
        featuredFileToProcess = clubDir.joinpath(featuredFileToProcessFile)

        # todo: for debugging, erase output files
        if featuredFileToProcess.exists():
            featuredFileToProcess.unlink()

        CreateBoardDataFeatures(
            pool, clubNumber, pickleFileToProcess, featuredFileToProcess, tcgd, tcg299d)

    if not pool is None:
        print(
            f'{time.strftime("%X")}: {__file__}: main: PickledPandas file creation scheduling complete: Elapsed pool seconds: {time.time()-poolStartTime:.2f}')
        pool.close()
        pool.join()
        pool = None

    print(
        f'{time.strftime("%X")}: {__file__}: main: PickledPandas file creation complete: Elapsed seconds: {time.time()-startTime:.2f}')

    print(f'{time.strftime("%X")}: {__file__}: main: Total elapsed time: Elapsed seconds: {time.time()-startTime:.2f}')


if __name__ == "__main__":
    main(None)
