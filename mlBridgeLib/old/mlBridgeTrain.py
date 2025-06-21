#!/usr/bin/env python
# coding: utf-8

# Per club, reads 'featured_bdf.pkl' files, trains on each dep_var, outputs individual dep_var files for archiving.
# Also, per club, creates 'modelresults.pkl' which always contains the latest model to be trained.

# next steps:
# mlBridgeInfer.py reads 'modelresults.pkl'

# Previous steps:
# mlBridgeAugmentor.py per club, reads 'club.games.pandas.pkl' and outputs 'featured_bdf.pkl' file.

# priority: need to implement 000000 club number

# takes 740s/1370s for 108571 max_iter=200 hidden_layer_sizes=(400, 200, 100) with/without multiprocessing. Score 0.25.
# warning: using more than 5 nCPUs is causing threads to abort because of No Memory.
# takes 150s for 000000 max_iter=200 using 5 nCPUs in multiprocessing. Takes 300s using no-multiprocessing but with n_jobs=12 for sklearn regression.
# takes 1200s for 000000 max_iter=2000 using 5 nCPUs in multiprocessing. Looks like 5/2000 is 65% faster than n_jobs=12.
# takes 7800s for 000000 max_iter=20000 (n_iter avg 13000) using 5 nCPUs in multiprocessing.  65% faster than n_jobs=12.
# takes 18000s for MLPRegressor using hidden_layer_sizes = (400, 100)

# todo:
# make each dep_var use a multiprocessing pool thread with n_jobs=1
# what to do with 'Tricks' and 'L' which originate in BridgeFinesseRankHtmlToPickle.py? They're helpful for debugging and dropped here.
# slow for Rank.* use multiprocessing for each dep_var instead of sklearn's loki
# write results after every training to allow restarts
# needs progress bar counter
# implement method for selecting type of estimator: linearregression, logisticregressionclassifier, neural net, etc.
# add estimator name to output file names. e.g. DD_Tricks_LogisticRegressionClassifier_20000Iters.pkl
# implement additional dep_vars such as DD_Tricks_S_NS_Length_10

# notes:
# Removed global variables because they're problematic when used with multiprocessing. Values were sometimes undef.


import numpy as np
import pandas as pd
from collections import defaultdict
import pathlib
import re
import math
import sklearn
from sklearn import linear_model
import pickle
from mlBridgeLib.mlBridgeLib import *
import TheCommonGameOptionsLib
from IPython.display import display  # needed for VSCode

import argparse
import multiprocessing
import time

import sklearn.neural_network
import inspect


def debuginfo(message):
    caller = inspect.getframeinfo(inspect.stack()[1][0])
    print(f"{caller.filename}:{caller.function}:{caller.lineno} - {message}")


def MLPRegressor(dep_var, trainx, trainy, validx, validy, n_jobs, max_iter, hidden_layer_sizes, **kwargs):
    print(f'{MLPRegressor.__name__}:{dep_var}')
    print(f'{MLPRegressor.__name__}:{max_iter}')
    debuginfo(f'{MLPRegressor.__name__}:{hidden_layer_sizes}')
    print(
        f'MLPRegressor: Start: {dep_var}: max_iter={max_iter} hidden_layer_sizes={hidden_layer_sizes}')
    # todo: need to print any non-numeric and non-boolean column names and dtypes
    assert sum([not pd.api.types.is_numeric_dtype(c)
                and not pd.api.types.is_bool_dtype(c)
                for c in trainx.dtypes]) == 0  # pd.api.types.is_categorical_dtype(c) and string not allowed
    print(f'MLPRegressor:calling')
    # MLPRegressor or MLPClassifier? If regression, there's issues of how to round to trick and availability of probs.
    m = sklearn.neural_network.MLPClassifier(
        random_state=1, max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes)  # , **kwargs)
    print(f'MLPRegressor:fitting')
    m.fit(trainx, trainy)  # .values.ravel())
    print(f'MLPRegressor:completing')
    print(f'MLPRegressor: Completed: {dep_var}: n_iter_={m.n_iter_}')
    print(f'MLPRegressor: score:{m.score(validx,validy)}')
    predictionsCoefficients = m.coefs_
    print(len(predictionsCoefficients), len(trainx.columns))
    assert len(predictionsCoefficients[0]) == len(
        trainx.columns)  # feature count
    # other items in predictionsCoefficients differ according to hidden_layer_sizes
    # assert len(predictionsCoefficients[1]) == hidden_layer_sizes[0] # verify layer sizes
    # dataframe of coefficents and features
    predictionsCoefficientsdf = pd.DataFrame(
        predictionsCoefficients[0].T, columns=validx.columns)
    # print(f'len(predictionsCoefficients):{len(predictionsCoefficients)}')
    # print(f'len(columns):{len(trainx.columns)}')
    #print('predictionsCoefficients:',sorted([(coef,col) for coef,col in zip(predictionsCoefficients,trainx.columns)]))
    predictionsEmbedded = m.predict(validx)
    assert all(predictionsEmbedded >= 0) and all(predictionsEmbedded <= 13), [
        v for v in predictionsEmbedded if v < 0 or v > 13]
    # print(f'len(predictionsEmbedded):{len(predictionsEmbedded)}')
    #print('predictionsEmbedded:', predictionsEmbedded)
    # for classification, not regression
    probabilities = m.predict_proba(validx)
    print(f'len(probabilities):{len(probabilities)}')
    print('probabilities:', probabilities)
    # n_classes, n_classes_ unavailable for LogisticRegressionClassifier
    print('Number of classes actually used:', len(m.classes_))
    print('Classes actually used:', m.classes_)
    # todo: for regression, not classification. np.full creates array of given shape full of np.nan
    # probabilities = np.full((len(validx), 14), np.nan) # todo: why bother with probs of all np.nan? What are actual probs?
    #assert probabilities.shape == (len(validx), 14)
    return m, predictionsEmbedded, probabilities, predictionsCoefficientsdf


def LinearRegression(dep_var, trainx, trainy, validx, validy, n_jobs, max_iter, hidden_layer_sizes, **kwargs):
    print(
        f'LinearRegressionClassifier: Start: {dep_var}: n_jobs={n_jobs}')
    assert sum([not pd.api.types.is_numeric_dtype(
        c) and not pd.api.types.is_bool_dtype(c) for c in trainx.dtypes]) == 0
    m = sklearn.linear_model.LinearRegression(**kwargs)
    m.fit(trainx, trainy.values.ravel())
    print(f'LinearRegression: Completed: {dep_var}')
    predictionsCoefficients = m.coef_
    #print(predictionsCoefficients.shape, len(trainx.columns))
    assert predictionsCoefficients.shape == (
        len(trainx.columns),)  # feature count
    # dataframe of coefficents and features
    predictionsCoefficientsdf = pd.DataFrame(
        predictionsCoefficients, index=validx.columns)
    # print(f'len(predictionsCoefficients):{len(predictionsCoefficients)}')
    # print(f'len(columns):{len(trainx.columns)}')
    #print('predictionsCoefficients:',sorted([(coef,col) for coef,col in zip(predictionsCoefficients,trainx.columns)]))
    predictionsEmbedded = m.predict(validx)
    # print(f'len(predictionsEmbedded):{len(predictionsEmbedded)}')
    #print('predictionsEmbedded:', predictionsEmbedded)
    #probabilities = m.predict_proba(validx)
    # print(f'len(probabilities):{len(probabilities)}')
    #print('probabilities:', probabilities)
    # n_classes, n_classes_ unavailable for LogisticRegressionClassifier
    #print('Number of classes actually used:', len(m.classes_))
    #print('Classes actually used:', m.classes_)
    probabilities = np.full([len(validx), 14], np.nan)
    assert probabilities.shape == (len(validx), 14)
    return m, predictionsEmbedded, probabilities, predictionsCoefficientsdf


def LogisticRegressionClassifier(dep_var, trainx, trainy, validx, validy, n_jobs, max_iter, hidden_layer_sizes, **kwargs):
    print(
        f'LogisticRegressionClassifier: Start: {dep_var}: max_iter={max_iter} n_jobs={n_jobs}')
    assert sum([not pd.api.types.is_numeric_dtype(
        c) and not pd.api.types.is_bool_dtype(c) for c in trainx.dtypes]) == 0
    m = sklearn.linear_model.LogisticRegression(**kwargs)
    m.fit(trainx, trainy.values.ravel())
    print(
        f'LogisticRegressionClassifier: Completed: {dep_var}: n_iter_={m.n_iter_}')
    predictionsCoefficients = m.coef_
    assert predictionsCoefficients.shape == (
        14, len(trainx.columns))  # 14 x feature count
    # dataframe of coefficents and features
    predictionsCoefficientsdf = pd.DataFrame(
        predictionsCoefficients, columns=validx.columns)
    # print(f'len(predictionsCoefficients):{len(predictionsCoefficients)}')
    # print(f'len(columns):{len(trainx.columns)}')
    #print('predictionsCoefficients:',sorted([(coef,col) for coef,col in zip(predictionsCoefficients,trainx.columns)]))
    predictionsEmbedded = m.predict(validx)
    # print(f'len(predictionsEmbedded):{len(predictionsEmbedded)}')
    #print('predictionsEmbedded:', predictionsEmbedded)
    probabilities = m.predict_proba(validx)
    assert probabilities.shape == (len(validx), 14)
    # print(f'len(probabilities):{len(probabilities)}')
    #print('probabilities:', probabilities)
    # n_classes, n_classes_ unavailable for LogisticRegressionClassifier
    print('Number of classes actually used:', len(m.classes_))
    print('Classes actually used:', m.classes_)
    return m, predictionsEmbedded, probabilities, predictionsCoefficientsdf


def CreateDepVarsList(df):
    # todo: include dep_var dtype in dep_vars_list
    # todo: remove double
    dep_vars_list = [(MakeColName('DD', 'Tricks', suit, direction), MakeColName('Actual', 'Tricks', suit, direction),
                      suit, direction, double) for direction in ['NS', 'EW'] for suit in 'CDHSN' for double in ['']]
    return dep_vars_list


def CreateSavedColumnList(df):
    # todo: convert saved... and drop... into one method. using names of include and exclude columns?
    saved_columns_names = ['Key', 'EventBoard', 'TCG_Link',
                           'Club_Link', 'Vul_NS', 'Vul_EW', 'Par_.*']
    saved_columns_regex = r'^(?:'+'|'.join(saved_columns_names)+'$)'
    print('Saved columns regex:', saved_columns_regex)
    saved_columns_names = list(
        set([c for c in df if re.search(saved_columns_regex, c)]))
    print('New columns to save:', saved_columns_names)
    # some are needed for InsertScoringColumnsTricks()
    return saved_columns_names


def CreateDropList(df, direction):
    # drop_column_names = [dep_var] # remove dep_var from df
    drop_column_names = ['Key', 'Board', 'Club_Link', 'DDmakes', 'Dealer', 'EventBoard', 'Hand_C_.*',
                         'Hands', 'HCP', 'L', 'LoTT', 'Par.*', 'Results', 'TCG_Link', 'Tricks', 'Vul']  # worthless data to any classifier
    # determine opponents direction
    if direction == 'NS':
        odirection = 'EW'
    else:
        odirection = 'NS'
    # remove all info about opponents hands
    drop_column_names += [
        'HCP_'+odirection,
        'HCP_['+odirection+']',
        'HandMaxSuit.*'+odirection,
        'HandMaxSuit.*['+odirection+']',
        'DD_(Bid|Double|L|Level|Result|Score|Suit|Tricks|Type)_.*'+odirection,
        'DD_(Bid|Double|L|Level|Result|Score|Suit|Tricks|Type)_.*['+odirection+']',
        # 'Hand_[CBLPQ]_.*'+odirection,
        # 'Hand_[CBLPQ]_.*['+odirection+']'
    ]
    # remove improper correlations to prevent tainting of classifier
    drop_column_names += [
        'DD_(Bid|Double|L|Level|Result|Score|Suit|Tricks|Type)_[CDHSN]_.*',
        # HandMaxSuit_L is sum of lengths of longest ns + ew suit
        'HandMaxSuit_(L|Suit|Tricks).*',
        'HCP_.*',
        'LoTT_.*'
    ]
    # drop columns which are unneeded or unduly biased
    drop_columns_regex = r'^(?!(?:'+'|'.join(drop_column_names)+')$)'
    print('Drop regex:', drop_columns_regex)
    dropping_columns = [c for c in df.columns if re.search(
        drop_columns_regex, c) is None]
    print('Dropping columns:', ', '.join(dropping_columns))
    return drop_columns_regex


def TrainValidateSplit(df, clubNumber):

    # show column names and dtype in sorted order
    print(sorted([(c, df[c].dtype) for c in df.columns]))

    # ,suit=['S']) #,direction='NS') #,doubles=['']) # direction='NS',
    df = FilterBoards(df, clubNumber)

    # create filtering list (T/F) of rows by year. Useful for train/validate split using years.
    select2018 = df['EventBoard'].str.startswith('18')
    select2019 = df['EventBoard'].str.startswith('19')
    select201x = df['EventBoard'].str.startswith('1')  # e.g. years 2013-2019
    select2020 = df['EventBoard'].str.startswith('20')
    # display(df[select2018].tail())
    print('Len of lists:', len(select2018), len(
        select2019), len(select201x), len(select2020))
    print('Count of lists:', select2018.count(),
          select2019.count(), select201x.count(), select2020.count())
    print('Sum of True items:', sum(select2018), sum(select2019), sum(select201x), sum(
        select2020), sum(select2018)+sum(select2019)+sum(select201x)+sum(select2020))

    # create train/validate split based on year. Train using 2019. Validate using 2020.
    # only need to create list of df indexes, not new dataframes.
    # df should already be sorted by Key (club+event+board)
    #train_idx = df[select2019].index
    train_idx = df[select201x].index
    valid_idx = df[select2020].index

    if len(train_idx) == 0 or len(valid_idx) == 0:
        return None

    # doesn't work with multilabels df.loc[valid_idx,new_dep_var] = None # just in case, remove any values in new_dep_var
    display(df.loc[train_idx].head(5))
    display(df.loc[valid_idx].head(5))

    # create tuple of lists for train/validate split
    splits = (list(train_idx), list(valid_idx))
    print(splits[0][:5], splits[1][:5])
    return splits


def GetPredictions(df, dep_var, trainx, trainy, validx, validy, n_jobs, max_iter, hidden_layer_sizes):
    # todo: finish implementing multiple output labels
    print(f'GetPredictions: {df.columns}')
    assert dep_var in df.columns
    print(f'GetPredictions: {df[dep_var].notna().all()}')
    assert df[dep_var].notna().all()
    #print('df.columns:', ', '.join(df.columns))
    # show column names and dtype in sorted order
    #print('df:', sorted([(c, df[c].dtype) for c in df.columns]))
    cat_names, cont_names = SetupCatCont(df)
    print(f'GetPredictions: {dep_var}')
    print(f'trainx:{trainx.columns}')
    print(f'trainy:{trainy}')
    print(f'validx:{validx.columns}')
    print(f'validy:{validy}')
    # print('After SetupCatCont:', [(c, df[c].dtype) for c in cat_names], [
    #    (c, df[c].dtype) for c in cont_names])
    # print(df.head())

    #display('Before TP:', df)
    #print(cat_names, cont_names)
    # convergence occurs between 10,000 and 100,000
    # either MLPRegressor, or LinearRegression, or LogisticRegressionClassifier
    m, predictionsEmbedded, probabilities, predictionsCoefficientsdf = MLPRegressor(dep_var,
                                                                                    trainx, trainy, validx, validy, n_jobs, max_iter, hidden_layer_sizes)  # , max_iter=max_iter, n_jobs=n_jobs)

    #print('(embedding number, predicted number):',[(m.classes_[n],round(c,2)) for l in predictionsEmbedded for n,c in enumerate(l) if c > 0])
    #print('(embedding number, predicted number):')
    #print(probabilities[0], sum(probabilities[0]))
    # print([(m.classes_[l.argmax()],round(l.max(),2)) for l in probabilities])# create ActualProb column containing statistics of actual and predicted probabilities
    # debug print([(m.classes_[i],round(l,2)) for ll in probabilities for i,l in enumerate(ll)])# create ActualProb column containing statistics of actual and predicted probabilities

    #print(f'len(trainx):{len(trainx)} len(trainy):{len(trainy)}')
    #print(f'len(validx):{len(validx)} len(validy):{len(validy)}')
    # todo: why did this work in 17.ipynb?
    #    print(
    #        f'rmse train:{m_rmse(m, trainx, trainy)} rmse valid:{m_rmse(m, validx, validy)}')
    # Mean absolute error
    print(
        f'Mean absolute error:{sklearn.metrics.mean_absolute_error(validy.to_numpy(), predictionsEmbedded)}')
    # Mean squared error
    msqe = sklearn.metrics.mean_squared_error(
        validy.to_numpy(), predictionsEmbedded)
    print(f'Mean squared error:{msqe}')
    # Root mean squared error
    if msqe < 0:
        print(
            f"Error: Mean squared error:{msqe} is negative. How's that possible?")
    else:
        print(
            f'Root mean squared error:{math.sqrt(sklearn.metrics.mean_squared_error(validy.to_numpy(), predictionsEmbedded))}')
    # The coefficient of determination: 1 is perfect prediction
    print(
        f'Coefficient of determination:{sklearn.metrics.r2_score(validy.to_numpy(), predictionsEmbedded)}')

    # display validation set -- independent and dependent variables
    #display(validx, type(validy), validy)

    return m, predictionsEmbedded, probabilities, predictionsCoefficientsdf


def TrainModelMp(args):
    df, train_idx, valid_idx, dep_vars, depVarFile, n_jobs, max_iter, hidden_layer_sizes = args
    return TrainModel(None, df, train_idx, valid_idx, dep_vars, depVarFile, n_jobs, max_iter, hidden_layer_sizes)


def TrainModel(pool, df, train_idx, valid_idx, dep_vars, depVarFile, n_jobs, max_iter, hidden_layer_sizes):
    if not pool is None:
        return pool.apply_async(TrainModelMp, [[df, train_idx, valid_idx, dep_vars, depVarFile, n_jobs, max_iter, hidden_layer_sizes]])

    startTime = time.time()
    print(
        f'{time.strftime("%X")}: {__file__}: TrainModel: Begin: dep_vars: {dep_vars}:: Total elapsed seconds: {time.time()-startTime:.2f}')

    # todo: include dep_var dtype in dep_vars_list
    dep_var, new_dep_var, suit, direction, double = dep_vars
    drop_columns_regex = CreateDropList(df, direction)

    # select columns for training. Drop unwanted columns.
    # can also use traindf.select()
    # need copy() because of drop()
    filtereddf = df.filter(regex=drop_columns_regex, axis=1).copy()
    print('Keeping columns:', filtereddf.columns)
    #assert len(filtereddf.select_dtypes(['object'])) == 0
    if dep_var not in filtereddf:
        print(f'dep_var {dep_var} was dropped ... re-adding')
        filtereddf[dep_var] = df[dep_var]
    display('filtereddf', filtereddf.head(4))

    # create train and validate dataframes
    trainx = filtereddf.loc[train_idx, :]
    trainy = filtereddf.loc[train_idx, dep_var]
    trainx.drop(dep_var, axis=1, inplace=True)
    validx = filtereddf.loc[valid_idx, :]
    validx.drop(dep_var, axis=1, inplace=True)
    validy = filtereddf.loc[valid_idx, dep_var]

    # call AI model to get predictions
    m, predictionsEmbedded, probabilities, predictionsCoefficientsdf = GetPredictions(
        filtereddf, dep_var, trainx, trainy, validx, validy, n_jobs, max_iter, hidden_layer_sizes)

    assert validy.ge(0).all() and validy.le(
        13).all(), validy[validy.le(0) | validy.gt(13)]
    assert len(predictionsEmbedded) == len(valid_idx)

    # when multiprocessing, easiest way to deal with updating values is to write them out and read them after multiprocessing has completed.
    predictionsDict = {'predictionsEmbedded': predictionsEmbedded,
                       'probabilities': probabilities, 'predictionsCoefficientsdf': predictionsCoefficientsdf}
    with open(depVarFile, 'wb') as f:
        pickle.dump(predictionsDict, f)

    print(f'{time.strftime("%X")}: {__file__}: TrainModel: Done: dep_vars: {dep_vars}: Total elapsed time: {time.time()-startTime:.2f}')


def Train(pool, clubNumber, clubPath, savedModelsPath, n_jobs, max_iter, hidden_layer_sizes):

    startTime = time.time()
    print(
        f'{time.strftime("%X")}: {__file__}: Train: Begin: clubNumber: {clubNumber}:: Total elapsed seconds: {time.time()-startTime:.2f}')

    featuredFile = clubPath.joinpath('featured_bdf.pkl')
    modelResultsFile = clubPath.joinpath(
        'modelresults.pkl')  # checkpoint results of model

    # temp or permanent? Forcing model to be recalculated
    if True or (not modelResultsFile.exists()) or (featuredFile.stat().st_mtime > modelResultsFile.stat().st_mtime):

        df = pd.read_pickle(featuredFile)
        assert ((df['Par_Result'] < 0) == (df['Par_Double'] == '*')).all()
        dep_vars_list = CreateDepVarsList(df)
        display(dep_vars_list)

        # remove any existing temporary files (depVarFiles).
        for dep_vars in dep_vars_list:
            # break
            dep_var, new_dep_var, suit, direction, double = dep_vars
            depVarFile = savedModelsPath.joinpath(
                f"{dep_var}_{max_iter}Iters.pkl")
            if depVarFile.exists():
                depVarFile.unlink()

        splits = TrainValidateSplit(df, clubNumber)
        if splits is None:  # unable to split because splitting criteria results in missing or insufficient data
            print("Error: Unable to split train-validate dataset")
            return None
        train_idx, valid_idx = splits
        # get list of column names to insert into validdf, one-time only. These columns are helpful to post-inference analysis.
        saved_columns_names = CreateSavedColumnList(df)
        validdf = df.loc[valid_idx, saved_columns_names]
        #validdf = pd.merge(df.loc[valid_idx], saved_columns_df.loc[valid_idx][nonduplicate_columns], left_index=True, right_index=True, how='outer')

        poolStartTime = time.time()

        for dep_vars in dep_vars_list:

            dep_var, new_dep_var, suit, direction, double = dep_vars
            assert dep_var in df.columns
            assert dep_var not in validdf.columns
            depVarFile = savedModelsPath.joinpath(
                f"{dep_var}_{max_iter}Iters.pkl")
            if depVarFile.exists():
                continue
            TrainModel(pool, df, train_idx, valid_idx, dep_vars,
                       depVarFile, n_jobs, max_iter, hidden_layer_sizes)

        if not pool is None:
            print(
                f'{time.strftime("%X")}: {__file__}: Train: training {dep_vars} scheduling complete: Elapsed pool seconds: {time.time()-poolStartTime:.2f}')
            pool.close()
            pool.join()
            pool = None

        print(
            f'{time.strftime("%X")}: {__file__}: main: Train: complete: Elapsed seconds: {time.time()-startTime:.2f}')

        probdfs = pd.DataFrame()  # use dict, then df instead?
        predictionsCoefficientsd = {}
        for dep_vars in dep_vars_list:

            dep_var, new_dep_var, suit, direction, double = dep_vars
            assert dep_var not in validdf.columns
            AssignToColumn(validdf, dep_var,
                           df.loc[valid_idx, dep_var], 'int8')
            depVarFile = savedModelsPath.joinpath(
                f"{dep_var}_{max_iter}Iters.pkl")

            with open(depVarFile, 'rb') as f:
                predictionsDict = pickle.load(
                    f)
            predictionsEmbedded = predictionsDict['predictionsEmbedded']
            probabilities = predictionsDict['probabilities']
            predictionsCoefficientsdf = predictionsDict['predictionsCoefficientsdf']

            # add key for prediction coefficients dataframe. Use case is architectural search and debugging
            #predictionsCoefficientsdf = pd.DataFrame(predictionsCoefficients, columns=trainx.columns)
            # assert predictionsCoefficientsdf.shape == (14, len(validdf.columns)) # 14 x feature count
            predictionsCoefficientsd[dep_var] = predictionsCoefficientsdf

            # append probabilities for each trick count
            assert probabilities.shape == (len(validdf), 14)
            AssignToColumn(probdfs, MakeSuitCols('Prob', suit, direction),
                           pd.DataFrame(probabilities, index=validdf.index), 'float')
            assert all(predictionsEmbedded >= 0) and all(predictionsEmbedded <= 13), [
                v for v in predictionsEmbedded if v < 0 or v > 13]

            # create column of predicted tricks
            assert len(predictionsEmbedded) == len(validdf)
            AssignToColumn(validdf, MakeColName('Pred', 'Tricks',
                                                suit, direction), predictionsEmbedded, 'float')

        assert probdfs.shape == (len(validdf), 14*len(dep_vars_list))
        assert len(predictionsCoefficientsd) == len(dep_vars_list)
        # write model results as a dict to 'latest model' filename.
        modelResultsDict = {'validdf': validdf, 'probdfs': probdfs, 'dep_vars_list': dep_vars_list,
                            'predictionsCoefficientsd': predictionsCoefficientsd}
        with open(modelResultsFile, 'wb') as f:
            pickle.dump(modelResultsDict, f)
        # write model results to filename specific to model parameters for archiving.
        depVarResultFile = savedModelsPath.joinpath(
            f"DD_Tricks_{max_iter}Iters.pkl")
        with open(depVarResultFile, 'wb') as f:
            pickle.dump(modelResultsDict, f)

    print(f'{time.strftime("%X")}: {__file__}: Train: Done: clubNumber: {clubNumber}: Total elapsed time: {time.time()-startTime:.2f}')

    return


def main(args):
    print(f'{time.strftime("%X")}: {__file__}: main: mlBridgeTrain: Train and Validate Bridge Deals')
    startTime = time.time()

    options = TheCommonGameOptionsLib.getOptions(args)

    # Initial testing shows multiprocessing causes an 80% speed up on my notebook. Using apply_async() gains another 10%
    # turns on multiprocessing. flip on/off to test performance difference. Default is on.
    useMultiprocessing = options.multiprocess
    useMultiprocessing = False  # temp for debugging

    pool = None
    nCPUs = options.cpus
    # todo: how to specify max_iter, n_jobs and any program specific options from the command line?
    # todo: max_iter not an attribute for LinearRegression
    # max_iter and hidden_layer_sizes (MLPRegressor only) need to be adjusted if regressor issues message informing failure to converage
    max_iter = 200  # 20000
    hidden_layer_sizes = (400, 200, 100)  # MLPClassifier ok (400, 200, 100)
    # Warning message appears when calling from LogisticRegression. Favor multiprocessing or Loky?
    # "UserWarning: Loky-backed parallel loops cannot be called in a multiprocessing, setting n_jobs=1"
    # is n_jobs still relevant?
    n_jobs = 1 if useMultiprocessing else 12 if nCPUs < 12 else nCPUs

    # if nCPUs > 3: # todo: implement max cpu count
    #    # todo: needed to limit nCPUs to avoid out of memory (this module only). Better workaround?
    #    nCPUs = 3

    rootPathStr = '../..'  # todo: options.path
    rootPath = pathlib.Path(rootPathStr)
    clubsPath = rootPath.joinpath('clubs')
    savedModelsPath = rootPath.joinpath('SavedModels')

    clubNumbers = options.clubNumbers
    # temp for debugging ftlbc: 104034 small: 275966 missing 2020 data: 267096
    # todo: temp
    # '108571'  # temp 209080 275966 108571
    clubNumbers = [clubsPath.joinpath('108571')]
    forceRewriteOfOutputFiles = options.forcerewrite
    deleteOutputFiles = options.delete

    # todo: where should display options go? In getOptions?
    # pd.set_option('display.max_columns', 500) # show all rows
    pd.set_option('display.max_rows', 50)
    pd.options.display.max_columns = None
    pd.options.display.width = None

    print(f'{time.strftime("%X")}: {__file__}: main: Processing clubs:{clubNumbers}')

    forceRewriteOfOutputFiles = deleteOutputFiles = False  # temp

    if useMultiprocessing:
        pool = multiprocessing.Pool(processes=nCPUs)

    #clubNumberName = None
    #Train(pool, clubNumberName)
    # assert False  # temp

    for cn in clubNumbers:
        clubNumber = cn  # loc[1]
        clubDir = cn.parent  # loc[2]
        print(
            f"Start processing club {clubNumber.name} in directory {clubDir}")

        clubNumberName = clubNumber.name
        Train(pool, clubNumberName, clubNumber, savedModelsPath,
              n_jobs, max_iter, hidden_layer_sizes)

        print(
            f"Finished processing club {clubNumber.name} in directory {clubDir}")

    print(f'{time.strftime("%X")}: {__file__}: main: Total elapsed time: Elapsed seconds: {time.time()-startTime:.2f}')


if __name__ == "__main__":
    main(None)
