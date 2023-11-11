from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import sklearn
#Function to produce training and targets using full dataset tables

#GLOBAL PATH to data
#DATA_PATH = '~/data/finance'

def produceXYDataSets(ticker, corp, ns_back): 
    df = pd.read_csv('~/data/finance'+'/optionchaindata/all/'+ticker+'_alldata_'+corp+'.csv.zip', parse_dates=['quoteDate','expiryDate'])
    print("Dataframe shape from file",df.shape)
    
    #basic data cleaning, remove lines where the strike price is more than 50 from the stockprice
    df_good = df[ df['strikeDelta'] > -50 ]
    print("After removing deltastrike bigger than -50", df_good.shape)

    df_good = df_good[ df_good['strikeDelta'] < 50 ]
    print("After removing deltastrike less than 50", df_good.shape)

    #All contract names to read through
    contracts = df_good['contractSymbol'].unique()

    x_train = []
    y_train = []

    #Only look at contracts that have at least 1.5 time as many entries as the look back time
    nquotes_min = int(1.5*ns_back)
    good_contracts = []

    for contract in contracts:
        data = df_good[df_good['contractSymbol'] == contract]
        
        if data.shape[0] < nquotes_min:
            continue
        
        good_contracts.append(contract)
        
        data['deltaDays'] = data['quoteDate'].diff()
        data['weekday'] = data['quoteDate'].dt.dayofweek

        ndays = data.shape[0]

        for iday in range(ns_back+1, ndays):
            #X:
            stockPrices = data['stockClose'][iday-ns_back:iday].values
            strike = data['strike'].values[iday]
            openInterest = data['openInterest'].values[iday]
            daysToExpiry = int(data['daysToExpiry'].values[iday].split('days')[0])
            deltaDays = data['deltaDays'].values[iday]/ np.timedelta64(1, 'D')
            weekday = data['weekday'].values[iday]

            features =np.concatenate( [[strike, openInterest, daysToExpiry, deltaDays, weekday], stockPrices] )

            #y:
            ask = data['ask'].values[iday]
            bid = data['bid'].values[iday]

            targets = np.array([bid, ask])

            #print(weekday,ask,bid,daysToExpiry,deltaDays, strike,stock_prices)
            x_train.append(features)
            y_train.append(targets)
    

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    print("Used",len(good_contracts),"contracts total")
    print("Done, made data set with",x_train.shape[0],"samples")
    
    xydata = np.concatenate([x_train,y_train], axis=1)
    DATA_PATH = os.path.expanduser('~/data/')  # Expands the ~ to the full home directory path
    directory = os.path.join(DATA_PATH, 'save/')

    if not os.path.exists(directory):
        os.makedirs(directory)
        
    np.save(directory+ticker+'_'+corp+'_XY.npy',xydata)
    
    return x_train, y_train

   
