#!/usr/bin/env python
# coding: utf-8

# takes 130s/100s for 108571 with/without multiprocessing. Why does mp perform so much slower?
# Per club, uses trained model in 'modelresults.pkl', 'tcg_results_dict', 'tcg_results299_dict' outputting 'optresults.pkl'.

# next steps:
# 'optresults.pkl' used in charting notebooks.

# previous steps:
# mlBridgeTrain.py reads 'featured_bdf.pkl' files and outputs 'modelresults.pkl' file.

import numpy as np
import pandas as pd
from collections import defaultdict
import pathlib
import re
import math
import pickle
from mlBridgeLib.mlBridgeLib import *
import TheCommonGameOptionsLib
from IPython.display import display  # needed for VSCode

import argparse
import multiprocessing
import time


def CreateActualPreddf(validdf, suitdfs, dep_vars_list):

    prefixes = ['Prob', 'CumSum', 'Exp']
    suits = 'CDHSN'
    directions = ['NS', 'EW']
    trickCount = 14
    for prefix in prefixes:
        prefixdf = suitdfs[[MakeColName(prefix, '', suit, direction)+str(
            n).zfill(2) for direction in directions for suit in suits for n in range(trickCount)]]
        #prefixdf = prefixdf.rename({'_'.join([str(t),suit,direction]):'_'.join([prefix,suit,direction])+str(t) for direction in directions for suit in suits for t in range(trickCount)},axis=1)
        # display(prefixdf)
        prefixdf = pd.wide_to_long(prefixdf.reset_index(), [MakeColName(
            prefix, '', suit, direction) for direction in directions for suit in suits], i='index', j='suits/tricks').stack().unstack(1)
        display(prefixdf.head(20).style.apply(highlight_last_max, axis=1))

    print(dep_vars_list[0])
    actuals = [c for c in validdf.columns if c.startswith('Actual')]
    DDTricks = [c for c in validdf.columns if re.match(
        r'^DD_Tricks_[CDHSN]_(NS|EW)$', c) is not None]
    predTricks = [c for c in validdf.columns if re.match(
        r'^Pred_Tricks_[CDHSN]_(NS|EW)$', c) is not None]
    DDScores = [c for c in validdf.columns if re.match(
        r'^DD_Score_[CDHSN]_(NS|EW)$', c) is not None]
    predScores = [c for c in validdf.columns if re.match(
        r'^Pred_Score_[CDHSN]_(NS|EW)$', c) is not None]
    predLevels = [c for c in validdf.columns if re.match(
        r'^Pred_Level_[CDHSN]_(NS|EW)$', c) is not None]
    predResults = [c for c in validdf.columns if re.match(
        r'^Pred_Result_[CDHSN]_(NS|EW)$', c) is not None]
    apdf = validdf[[cc for c in zip(actuals, DDTricks, predTricks, DDScores, predScores,
                                    predLevels, predResults) for cc in c]]  # convert list of tuples to list
    display(apdf.head())
    return apdf


def computeExpected(i, rcumsumdf, dep_vars, makeScoresd, setScoresd):
    # calculate greatest expected contract by evaluation each contract level (e.g. 1C-7C)
    # evaulation method calculates the sum of the product of probabilities and set/make scores, always 14 possibilities
    # step 1) start calculations with contract of 1C, calculate all 8 possible sets (0-7), and 6 possible makes (8-13)
    # step 2) for sets, calculate (two vectors of length 14) using (1-rcumsum)*penaltyScore
    # step 3) for makes, calculate (two vectors of length 14) using rcumsum*makingScore
    # step 4) sum the result (two vectors of length 14) into a scaler
    # step 5) repeat calculations for 2C to 7C
    # step 6) return list of scaler sums for 1C to 7C. we're padding on left with [0]*7 to return 14 values
    sc = 1-rcumsumdf.values
    #display('1-rcumsumdf (sets):',sc)
    mc = rcumsumdf.values
    #display('rcumsumdf (makes):',mc)
    dep_var, new_dep_var, suit, direction, double = dep_vars
    suml = []
    for level in range(0, 7):
        # get set scores (level, suit, vulnerability, double, declarer)
        #ss = setScoresd[(level, list('CDHSN').index(suit), i['Vul_'+direction], 0, 0)]
        # use doubled score for worst case scenario
        doubled = True  # controls whether to use doubled score for down contracts
        ss = setScoresd[(level, list('CDHSN').index(
            suit), i['Vul_'+direction], doubled, 0)]
        # display('ss:',ss)
        # get make scores (level, suit, vulnerability, double, declarer)
        # use normal make (undoubled)
        ms = makeScoresd[(level, list('CDHSN').index(suit),
                          i['Vul_'+direction], 0, 0)]
        # display('ms:',ms)
        # calculate expected scores if set
        se = sc*ss
        # display('es:',se)
        # calculate expected scores if made
        me = mc*ms
        # display('em:',me)
        # calculate sums of expected sets/makes
        e = se+me
        # display(level,suit,direction,e)
        # calculate scaler sum of expectations
        sume = sum(e)
        # display('sum:',sume)
        # making list of sume for each contract level
        suml.append(sume)
    return [0]*7+suml  # return list of sume


def CreateExpParColumns(validdf, expectedPar, tcgd, tcg299d):
    # Note: passed out hands can contain pd.NA (<NA>) in some columns. What about support for anomlous data such as spoiled boards?
    bids = []
    suits = []
    directions = []
    tricks = []
    contracttypes = []
    estscores = []
    scores = []
    doubles = []
    results = []
    for ep in expectedPar:
        bid = ep[-1]
        if bid[1] == 'Pass':
            # Hand was passed out. Assigning pd.NA to many columns of their columns.
            bids.append('Pass')
            suits.append(pd.NA)
            directions.append(pd.NA)
            tricks.append(pd.NA)
            contracttypes.append('Pass')
            estscores.append(0)
            scores.append(0)
            doubles.append(pd.NA)
            results.append(pd.NA)
        else:
            m = re.match(r'Exp_([CDHSN])_(..)(\d+)', bid[1])
            assert m.lastindex == 3
            suits.append(m[1])
            directions.append(m[2])
            itricks = int(m[3])
            tricks.append(itricks)
            # todo: implement as function
            if m[1] == 'Pass':
                contract = 'Pass'
            elif itricks == 12:
                contract = 'SSlam'
            elif itricks == 13:
                contract = 'GSlam'
            elif ((m[1] in 'CD') and itricks >= 11) or ((m[1] in 'HS') and itricks >= 10) or ((m[1] in 'N') and itricks >= 9):
                contract = 'Game'
            else:
                contract = 'Partial'
            contracttypes.append(contract)
            estscores.append(bid[0])
            result = validdf.loc[bid[4], MakeColName(
                'DD', 'Tricks', m[1], m[2])]-itricks
            results.append(result)
            # example of repeating a char n times
            bids.append(str(itricks-6)+m[1]+'*' *
                        bid[3]+' '+m[2]+' '+str(result))
            scores.append(score(itricks-7, list('CDHSN').index(
                m[1]), bid[3], (m[2] == 'EW')*2, validdf.loc[bid[4], 'Vul_'+m[2]], result))  # EW score negative ok?
            doubles.append(bid[3])
            #print(bid[4],validdf.loc[bid[4], 'EventBoard'])
            #assert validdf.loc[bid[4], 'EventBoard'] != '200102A_17'
    prefix = 'Exp_Par'
    AssignToColumn(validdf, prefix+'_Bidding', expectedPar,
                   'object')  # 'object' because it holds a list
    AssignToColumn(validdf, prefix+'_Bid', bids, 'string')
    AssignToColumn(validdf, prefix+'_Dir', directions, 'string')
    AssignToColumn(validdf, prefix+'_Suit', suits, 'string')
    # tricks contains pd.NA (<NA>) for passed out hands
    AssignToColumn(validdf, prefix+'_Tricks', tricks, 'Int8')
    AssignToColumn(validdf, prefix+'_Type', contracttypes, 'string')
    AssignToColumn(validdf, prefix+'_Level',
                   validdf[prefix+'_Tricks']-6, 'Int8')  # Tricks contains pd.NA (<NA>) for passed out hands
    AssignToColumn(validdf, prefix+'_EScore', estscores, 'float')
    # should NOT contain a pd.NA (<NA>) for passed out hands but what about spoiled boards?
    AssignToColumn(validdf, prefix+'_Score', scores, 'int16')
    # contains pd.NA (<NA>) for passed out hands
    AssignToColumn(validdf, prefix+'_Double', doubles, 'Int8')
    # contains pd.NA (<NA>) for passed out hands
    AssignToColumn(validdf, prefix+'_Result', results, 'Int8')
    AssignToColumn(validdf, prefix+'_TCG_Key', validdf['EventBoard'].str.replace(
        'E2A', 'A').str.cat(validdf[prefix+'_Score'].map(str), sep='_'), 'string')
    AssignToColumn(validdf, prefix+'_TCG_MP_NS',
                   GetTcgMPs(tcgd, validdf[prefix+'_TCG_Key']), 'float')
    AssignToColumn(validdf, prefix+'_TCG_MP_EW', 1 -
                   validdf[prefix+'_TCG_MP_NS'], 'float')
    AssignToColumn(validdf, prefix+'_TCG299_MP_NS',
                   GetTcgMPs(tcg299d, validdf[prefix+'_TCG_Key']), 'float')
    AssignToColumn(validdf, prefix+'_TCG299_MP_EW', 1 -
                   validdf[prefix+'_TCG299_MP_NS'], 'float')


def FindOptimalScore(nsi, nsr, cb):
    nsidxmax = nsr.iloc[cb:].idxmax()
    return (nsr.iloc[cb:].max(), nsidxmax, nsr.index.get_loc(nsidxmax), 0, nsi)


def NextBid(expByDirection, nextBidder, previousBidder, cb, bids):
    nsi, nsr, ns = nextBidder
    ewi, ewr, ew = previousBidder
    cb = ew[2]+1  # next available bid
    # print(nsi,cb)
    # display(nsr)
    if cb < len(nsr):
        # find remaining optimal score
        # bid choices: pass, bid something, double, redouble (if already doubled)
        ns = FindOptimalScore(nsi, nsr, cb)
    else:
        # opponent bid 7nt, no futher bid possible other than (re)double
        # todo: need to process possible double of 7ntew
        ns = ew
    # possibility of (re)doubling being better than bidding optimal score
    # if expecting set and set > next bid and not doubled
    # double if not doubled and expecting set and set > next optimal bid
    if ew[0] < 0 and -ew[0] > ns[0] and ew[3] == 0:
        # using ew expected info but inserting a double
        ns = (ew[0], ew[1], ew[2], 1, ew[4])
    # redouble if already doubled and expecting positive score and positive score > next optimal bid
    elif ew[3] == 1 and ew[0] > 0 and ew[0]*2 > ns[0]:
        # todo: need to compute redoubled expectation instead of just using *2. Use suitdfs to compute re-double.
        # using ew expected info but inserting a double
        ns = (ew[0]*2, ew[1], ew[2], 2, ew[4])
    # 7nt was bid but wasn't (re)-doubled so no further bid possible
    if cb == len(nsr) and (ns[3] != 1 or ew[3] == 1):
        return
    # dealer makes opening pass (no improvement possible). ew has not yet bid.
    if ns[0] <= 0 and ew[2] < 0:
        ew = FindOptimalScore(ewi, ewr, cb)
        if ew[0] <= 0:  # ew can't improve so pass this hand out
            ew = (0, 'Pass', cb, 0, 0)  # passed out
        # either passed out or ew makes opening bid which can't be doubled because it has a positive expectation so just return.
        assert len(ew) == 5
        bids.append(ew)
        return  # auction over
    # next bidder can't improve so pass out.
    if ns[3] == 0 and -ns[0] >= ew[0]:
        return
    assert len(ns) == 5
    bids.append(ns)  # NS bids

    previousBidder = nsi, nsr, ns
    nextBidder = ewi, ewr, ew
    # nextBidder is now previousBidder and visa-versa
    NextBid(expByDirection, nextBidder, previousBidder, cb, bids)
    return


# find par score (Nash equilibrium)
def ParFinder(suitdfs):

    rotation = ['NS', 'EW']
    expByDirection = {}
    # create dataframes of expected scores by direction, by suit, by double
    for i, direction in enumerate(rotation):
        for double in range(0, 3):
            listOfExpColumnsByDirection = suitdfs[[g for g in suitdfs.columns if re.match(
                r'Exp_[CDHSN]_?\d+'.replace('?', direction), g)]]
            # display(listOfExpColumnsByDirection.head(2))
            expColumnsByLevelBySuit = listOfExpColumnsByDirection[[MakeColName('Exp', '', suit, direction)+str(level).zfill(2)
                                                                   for level in range(7, 14) for suit in 'CDHSN']]
            # display(expColumnsByLevelBySuit.head(2))
            assert len(expColumnsByLevelBySuit.columns) == 5 * \
                7  # five suits by seven bidding levels
            expByDirection[(i, double)] = expColumnsByLevelBySuit
    expectedPar = []
    for (nsi, nsr), (ewi, ewr) in zip(expByDirection[(0, 0)].iterrows(), expByDirection[(1, 0)].iterrows()):
        assert nsi == ewi
        assert len(nsr) == len(ewr)
        assert len(nsr.index) == len(ewr.index)
        bids = []
        cb = -1
        ns = ew = (0, '', cb, 0, 0, 0)
        # todo: change ns to dealer
        dealer = nsi, nsr, ns
        opponent = ewi, ewr, ew
        NextBid(expByDirection, dealer, opponent, cb, bids)
        # print(cb,nsi,bids)
        # expectedPar.append(bids[-1][1]+['','X','XX'][bids[-1][3]])
        expectedPar.append(bids)
        # final bid's expectation must be either >= 0 or doubled
        assert bids[-1][0] >= 0 or bids[-1][3] > 0
    return expectedPar


def CreatePredProbExpExpPar(validdf, probdfs, dep_vars_list, tcgd, tcg299d, makeScoresd, setScoresd):

    AssignToColumn(validdf, 'Par_Key',
                   (validdf['Par_Suit']+'_'+validdf['Par_Dir']).map(lambda x: 'DD_'+x), 'string')
    suitdfs = pd.DataFrame()
    for dep_vars in dep_vars_list:
        dep_var, new_dep_var, suit, direction, double = dep_vars

        # augment with 'Par' values
        InsertScoringColumnsTricks(validdf, dep_vars, 'DD')
        # InsertTcgColumns(validdf,dep_vars,prefix,tcgd,tcg299d) todo: shouldn't this be used to insert Scores TCG_MP

        prefix = 'Pred'
        InsertScoringColumnsTricks(validdf, dep_vars, prefix)
        InsertTcgColumns(validdf, dep_vars, prefix, tcgd, tcg299d)

        # for each suit, get class probabilities of trick counts (0-13)
        probdf = probdfs.loc[:, MakeColName(
            'Prob', '', suit, direction)+str(0).zfill(2):MakeColName('Prob', '', suit, direction)+str(13).zfill(2)]  # todo: create MakeColNameLevel()
        # todo: temp
        # assert probdf.isna().sum().sum() == 0
        # display(probdf)
        # (highest prob contract, highest class (tricks) prob, column index of highest class (tricks) prob)
        maxpredictionstuple = [
            (i, probdf[i].max(), probdf.columns.get_loc(i)) for i in probdf.idxmax(axis=1)]
        # display(maxpredictionstuple[:20])
        maxpredictions = [t[2] for t in maxpredictionstuple]
        maxprobabilities = [t[1] for t in maxpredictionstuple]
        AssignToColumn(validdf, MakeColName(
            prefix, 'Prob', suit, direction), maxprobabilities, 'float')

        # reverse cumsum() # can't pd.DataFrame with .columns in one call so must make two lines
        rcumsumdf = (-probdf).cumsum(axis=1).add(1).shift(1,
                                                          axis=1, fill_value=1.0)
        rcumsumdf.columns = MakeSuitCols('CumSum', suit, direction)
        assert probdf.shape == rcumsumdf.shape
        assert rcumsumdf.isna().sum().sum() == 0

        prefix = 'Exp'
        # double=0, (direction == 'NS')*2 is zero or 2, result is zero
        expecteddf = validdf.apply(lambda x: computeExpected(
            x, rcumsumdf.loc[x.name, :], dep_vars, makeScoresd, setScoresd), axis=1, result_type='expand')
        expecteddf.columns = MakeSuitCols('Exp', suit, direction)
        # display(expecteddf.head(2).style.apply(highlight_last_max,axis=1))
        # display(rcumsumdf,expecteddf)
        assert expecteddf.shape == rcumsumdf.shape
        assert expecteddf.isna().sum().sum() == 0
        # assign tricks from max expected score
        maxTricks = expecteddf.idxmax(axis=1).map(expecteddf.columns.get_loc)
        # assign tricks from max expected score
        AssignToColumn(validdf, MakeColName(
            prefix, 'Tricks', suit, direction), maxTricks, 'float')
        InsertScoringColumnsTricks(validdf, dep_vars, prefix)
        InsertTcgColumns(validdf, dep_vars, prefix, tcgd, tcg299d)

        suitdfs = pd.concat([suitdfs, probdf, rcumsumdf, expecteddf], axis=1)

    return suitdfs


def Validate(validdf, probdfs, dep_vars_list, tcgd, tcg299d, makeScoresd, setScoresd):

    assert validdf is not None and probdfs is not None and dep_vars_list is not None

    suitdfs = CreatePredProbExpExpPar(validdf, probdfs, dep_vars_list, tcgd, tcg299d, makeScoresd, setScoresd)

    validdf.sort_index(axis=1, inplace=True)
    # display(validdf.head(8))
    suitdfs = pd.concat(
        [validdf['EventBoard'], validdf['Vul_NS'], validdf['Vul_EW'], suitdfs], axis=1)
    # fyi, iloc accepts multiple column ranges when expressed as a list (dups ignored)
    # display(*[suitdfs.head(2).iloc[:, [0]+list(range(g, g+14))]
    #          for g in range(3, len(suitdfs.columns), 14)])

    #prefix = 'DD'
    #df = validdf
    # AssignToColumnLoc(df, df[MakeColName(prefix, 'Tricks', suit, direction)] > 6, MakeColName(
    #    prefix, 'Level', suit, direction), df[MakeColName(prefix, 'Tricks', suit, direction)]-6, 'Int8')

    # fyi, iloc accepts multiple column ranges when expressed as a list (dups ignored)
    # display(*[suitdfs.head(2).iloc[:, [0]+list(range(g, g+14))]
    #          for g in range(3, len(suitdfs.columns), 14)])

    expectedPar = ParFinder(suitdfs)

    CreateExpParColumns(validdf, expectedPar, tcgd, tcg299d)

    CreateActualPreddf(validdf, suitdfs, dep_vars_list)

    # display(validdf.head(4))

    # display(validdf['Exp_Par_TCG_MP_NS'].convert_dtypes().head(50))

    display(validdf[['EventBoard', 'Exp_Par_Bidding', 'Par_Bid', 'Par_Dir', 'Par_TCG_MP_NS', 'Exp_Par_Bid', 'Exp_Par_Dir', 'Exp_Par_Score',
                 'Exp_Par_TCG_MP_NS', 'Exp_Par_Double', 'Exp_Par_Result', 'Exp_Par_Tricks', 'Exp_Par_Level', 'Exp_Par_Suit']])

    # print(validdf['DD_Level_C_NS'].dtype)

    # display(validdf[MakeColName('Exp', 'Tricks', 'S', 'NS')], validdf[MakeColName(
    #    'Exp', 'Bid', 'S', 'NS')], validdf[MakeColName('Exp', 'Score', 'S', 'NS')])

    # display(suitdfs.head(10))

    #import qgrid
    # qgrid.show_grid(validdf, grid_options={
    #                'forceFitColumns': False, 'defaultColumnWidth': 110})  # ,show_toolbar=True)
    # validdf.head()

    # show a dataframe of column names and dtype in sorted order
    # display(pd.DataFrame(sorted([(c, validdf[c].dtype)
    #                             for c in validdf.columns])))

    return validdf, suitdfs, dep_vars_list


def ValidateModelMp(args):
    clubNumber, filterBoards, modelResultsFile, optResultsFile, tcgd, tcg299d, makeScoresd, setScoresd = args
    return ValidateModel(None, clubNumber, filterBoards, modelResultsFile, optResultsFile, tcgd, tcg299d, makeScoresd, setScoresd)


def ValidateModel(pool, clubNumber, filterBoards, modelResultsFile, optResultsFile, tcgd, tcg299d, makeScoresd, setScoresd):
    if not pool is None:
        return pool.apply_async(ValidateModelMp, [[clubNumber, filterBoards, modelResultsFile, optResultsFile, tcgd, tcg299d, makeScoresd, setScoresd]])

    startTime = time.time()
    print(
        f'{time.strftime("%X")}: {__file__}: ValidateModel: Begin: ClubNumber: {clubNumber}: Total elapsed seconds: {time.time()-startTime:.2f}')

    # todo: add 'True or' for debugging - temp
    if filterBoards is not None or (not optResultsFile.exists()) or (modelResultsFile.stat().st_mtime > optResultsFile.stat().st_mtime):

        with open(modelResultsFile, 'rb') as f:
            modelResultsDict = pickle.load(
                f)

        validdf = modelResultsDict['validdf']
        probdfs = modelResultsDict['probdfs']
        dep_vars_list = modelResultsDict['dep_vars_list']
        predictionsCoefficientsd = modelResultsDict['predictionsCoefficientsd'] # dict of dfs

        #assert validdf['Pred_Tricks_C_NS'].le(13).all(),[v for v in validdf['Pred_Tricks_C_NS'] if v < 0 or v > 13]
        if filterBoards is not None:
            validdf = validdf[validdf['EventBoard'].isin(
                filterBoards)].copy()  # select only some boards
            # select only some board probabiliites
            probdfs = probdfs.loc[validdf.index, :].copy()
        validdf, suitdfs, dep_vars_list = Validate(
            validdf, probdfs, dep_vars_list, tcgd, tcg299d, makeScoresd, setScoresd)

        optResultsDict = {'validdf': validdf, 'suitdfs': suitdfs,
                          'dep_vars_list': dep_vars_list, 'predictionsCoefficientsd': predictionsCoefficientsd}
        with open(optResultsFile, 'wb') as f:
            pickle.dump(optResultsDict, f)

    print(f'{time.strftime("%X")}: {__file__}: ValidateModel: Done: ClubNumber: {clubNumber}: Total elapsed time: {time.time()-startTime:.2f}')


def main(args):
    print(f'{time.strftime("%X")}: {__file__}: main: mlBridgeTrain: Train and Validate Bridge Deals')
    startTime = time.time()

    options = TheCommonGameOptionsLib.getOptions(args)

    # Initial testing shows multiprocessing causes an 80% speed up on my notebook. Using apply_async() gains another 10%
    # turns on multiprocessing. flip on/off to test performance difference. Default is on.
    useMultiprocessing = options.multiprocess
    # todo: find out why global tcgd is None only when below is commented out. Some interaction between multiprocessing and global.
    useMultiprocessing = False  # temp for debugging
    pool = None
    nCPUs = options.cpus

    # initialize "read only" module variables

    rootPathStr = '../..'  # todo: temp options.path
    rootPath = pathlib.Path(rootPathStr)

    tcgd = pd.read_pickle(rootPath.joinpath('tcg_results_dict.pkl'))
    display(list(tcgd.items())[:10])

    tcg299d = pd.read_pickle(rootPath.joinpath('tcg_results299_dict.pkl'))
    display(list(tcg299d.items())[:10])

    scoresd, setScoresd, makeScoresd = ScoreDicts()

    clubsPath = rootPath.joinpath('clubs')

    clubNumbers = [pathlib.Path('108571')]  # options.clubNumbers
    # temp for debugging ftlbc: 104034 small: 275966 missing 2020 data: 267096
    # clubNumbers = [pathlib.PosixPath('275966')] #,pathlib.PosixPath('108571')]  # temp 209080 275966 108571
    forceRewriteOfOutputFiles = options.forcerewrite
    deleteOutputFiles = options.delete

    # todo: where should display options go? In getOptions?
    # pd.set_option('display.max_columns', 500) # show all rows
    pd.set_option('display.max_rows', 50)
    pd.options.display.max_columns = None
    pd.options.display.width = None

    print(f'{time.strftime("%X")}: {__file__}: main: Processing clubs:{clubNumbers}')

    forceRewriteOfOutputFiles = deleteOutputFiles = False  # temp

    inputFiles = ['modelresults.pkl']
    outputFiles = ['optresults.pkl']

    listOfClubs = ListOfClubsToProcess(
        clubNumbers, inputFiles, outputFiles, clubsPath, forceRewriteOfOutputFiles, deleteOutputFiles, reverse=useMultiprocessing)

    # not implemented. attempt to process a global file (acbl+tcg) instead of per club file.
    # todo: properly implement clubsPath, clubPath.joinpath(clubnumber)
    #modelResultsFile = pathlib.Path('modelresults.pkl')
    #optResultsFile = pathlib.Path('OptResults.pkl')
    # if optResultsFile.exists(): # temp
    #    optResultsFile.unlink()
    #clubNumberName = None
    # filterBoards = ['200101A_01','200102A_17'] # 200101A_01 will bid 6S-2 if not using doubles. 200102A_17 will bid 3S if using doubles (3C) otherwise.
    # filterBoards = None  # None means include all boards
    # ValidateModel(pool, clubNumberName, filterBoards,
    #              modelResultsFile, optResultsFile, tcgd, tcg299d)
    # assert False  # temp

    # for each club, create list of event files (*.game.json), read and merge into club.games.json
    if useMultiprocessing:
        pool = multiprocessing.Pool(processes=nCPUs)

    poolStartTime = time.time()

    for loc in listOfClubs:
        print(f"loc: {loc}")
        # modelResultsFileNameSize = loc[0] # ignore. only used for sorting.
        clubNumber = loc[1]
        clubDir = loc[2]
        modelResultsFileName = loc[3][0]
        optResultsFileName = loc[4][0]

        print(
            f"Start processing club {clubNumber.name} in directory {clubDir}: modelResults:{modelResultsFileName}: optResults:{optResultsFileName}")

        modelResultsFile = clubDir.joinpath(modelResultsFileName)
        optResultsFile = clubDir.joinpath(optResultsFileName)
        clubNumberName = clubNumber.name
        filterBoards = None  # None means include all boards
    
        # todo: for debugging only? delete output file to force recreation.
        if optResultsFile.exists():
            optResultsFile.unlink()

        ValidateModel(pool, clubNumberName, filterBoards,
                      modelResultsFile, optResultsFile, tcgd, tcg299d, makeScoresd, setScoresd)

        print(
            f"Finished processing club {clubNumber.name} in directory {clubDir}")

    if not pool is None:
        print(
            f'{time.strftime("%X")}: {__file__}: main: mlBridgeTrain file creation scheduling complete: Elapsed pool seconds: {time.time()-poolStartTime:.2f}')
        pool.close()
        pool.join()
        pool = None

    print(
        f'{time.strftime("%X")}: {__file__}: main: mlBridgeTrain complete: Elapsed seconds: {time.time()-startTime:.2f}')

    print(f'{time.strftime("%X")}: {__file__}: main: Total elapsed time: Elapsed seconds: {time.time()-startTime:.2f}')


if __name__ == "__main__":
    main(None)
