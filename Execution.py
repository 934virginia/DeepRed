#!/usr/lib/python3.6

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import time
import datetime
import random, os, sys
import numpy as np
import datetime
import DeepRed


INVERT = False
#leave this in forever because it's never going to stop being funny

try:
    from dataloader import TokenList, pad_to_longest
except: pass

#Initialize everything with the attributes that I was using
dr = DeepRed.RedObj()

#####Basically everything to play with on the model and training side of things.#####
##dr.SEQ_LEN= 30
##dr.FUTURE_PERIOD_PREDICT= 5
##dr.BATCH_SIZE= 32
##dr.INSTRUMENT= 'EUR_USD'
##dr.EPOCHS= 50
##dr.M_SCALE= 1
##dr.DST_OFFSET= 9
##dr.PATH= '/home/DeepRed/'
##dr.AMA= None
##dr.AMAc= None
##dr.GRANULARITY = 'M' + str(dr.M_SCALE)
##dr.RATIO_TO_PREDICT = 'mid'
##dr.X_VALID = None
##dr.X_TRAIN = None

#Minimize training epochs for speed of debugging the pipeline
dr.EPOCHS = 1




if dr.DSTCheck() == True:
    dr.DST_OFFSET -= 1

offset = dr.DST_OFFSET

###############################
##CHECK FOR PID FILES, EXIT IF NECESSARY##
dr.PIDcheck()
##############################
##Write PID file##
PIDx = os.getpid()
dr.PIDwrite(PIDx)

TradeDict = dr.getTradesInst()


#####LOGIC FOR ALREADY OPEN TRADES!#########
if str(TradeDict) != '{}':
    OpenDT = (datetime.datetime.strptime(((TradeDict['0'])['OpenDT'][1:-5]),'%Y-%m-%dT%H:%M:%S.%f')) - datetime.timedelta(hours=offset)
    Open = float(((TradeDict['0'])['Fill']))
    if float((TradeDict['0'])['InitialUnits']) > 0:
         Short = False
         BANow = 1
    else:
         Short = True
         BANow = 3
         
    #Returning this as a list is dumb. Make it a dict or something and skip the vars to declutter namespace.
    NowList = dr.getPrice()
    Spread = float(NowList[3]) - float(NowList[1])
    Buffer = Spread * 2
    Now = float(NowList[BANow])
    NowDT = datetime.datetime.now()

    #FPP-1 BECAUSE WE GET MIDS OF THE PREVIOUS MINUTE#
    if NowDT < OpenDT + datetime.timedelta(minutes=(dr.FUTURE_PERIOD_PREDICT-1) * dr.M_SCALE):
        breakpoint = 0
        print('not 5(x) minutes yet')
        dr.PIDexit(PIDx)
        ##Does trailing SL for initial 5 minutes. Maybe just use this one all the way and forget the 5 minute check?
        #^^Possible, but clearly not ready to mess with SL's yet.
        if Short == True:
            profit = Open-Now
            if profit > breakpoint:
                pass
                #Rework SL logic after base decision making. Currently making all bad trades worse and ensuring that fewer can break even.
                #Possible that stuff like this gets targeted specifically by quants.
            
                #stop_loss.main((TradeDict['0'])['TradeID'][1:-1], round(NowSL - (profit), 5), (TradeDict['0'])['SLOrderID'][1:-1])
            else:
                pass          

        elif Short == False:
            profit = Now-Open
            if profit > breakpoint:
                pass
                #stop_loss.main((TradeDict['0'])['TradeID'][1:-1], round(NowSL + (profit), 5), (TradeDict['0'])['SLOrderID'][1:-1])
            else:
                pass
        #pass
        
        
    else:
        breakpoint = Buffer/2
        print('5 minutes are done')
        dr.CloseMarketOrder(Short)
        
        if Short == True:
            profit = Open-Now
            if profit > breakpoint:
                if round(Now + (profit/10), 5) < NowSL:
                    pass
                    #dr.stop_loss((TradeDict['0'])['TradeID'][1:-1], round(Now + (profit/3), 5), (TradeDict['0'])['SLOrderID'][1:-1])
                else:
                    print("SL moving backward. No dice.")
            else:
                dr.CloseMarketOrder(Short)
                #pass

        elif Short == False:
            profit = Now-Open
            if profit > breakpoint: 
                if round(Now - (profit/10), 5) > NowSL:
                    pass                
                    #dr.stop_loss((TradeDict['0'])['TradeID'][1:-1], round(Now - (profit/3), 5), (TradeDict['0'])['SLOrderID'][1:-1])
                else:
                    print("SL moving backward. No dice.")
            else:
                dr.CloseMarketOrder(Short)

##This is a terrible way to handle these connection errors. Find where it is and import it.##
    while True:        
        try:
            TradeDict = dr.getTradesInst()
            break
        except Exception:
            pass
                        
######################END OF OPEN TRADE LOGIC#####################################

########LOGIC FOR OPENING A TRADE!############
if str(TradeDict) == '{}':

    ###############
    dr.makeModelFull()
    ###############

    ###THIS IS WHERE AMAp GOES-- MAKE SURE IT'S ALREADY ADJUSTED BEFORE WRITING TO FILE.
    ##Don't worry about this ^^ our HP sweep is garbage.

    with open(dr.PATH + 'v20-python-samples/models/' + dr.INSTRUMENT + '/history/HP/' + str(datetime.date.today() - datetime.timedelta(days=1)) + '.csv', 'r') as f:
        HPstring = f.read()[1:-1]
        HPlist = []
        for i in range(1, len(HPstring.split(', '))):
            HPlist.append(HPstring.split(', ')[i])

    AMA = int(HPlist[1])
    AMAc = float(HPlist[2])
    dr.AMA = AMA
    dr.AMAc = AMAc

    datefiles = sorted(os.listdir(dr.PATH + 'v20-python-samples/data/' + dr.INSTRUMENT + '/'), key = str)
    datefiles.pop(-1)

    totalprofitglobal = 100

    #Focusing entirely on mids is probably part of why this thing chugs. Doing lows and highs separately would be the obvious next move to minimize backtracking.
    #Already have all the raw data, would just rework (basically duplicate) directories, run TPipe andd associated checks twice (low/high),
    #and execute predictions on SEQ_LEN data from getPrice the same way, loading model once and just changing weights. Predictions then get spooned into algos for decision making.
    #Lows/Highs also open the door for juicier HyperParameters to sweep on recent backdata.  No more mystery "buffer" BS.

    #################################################################


    spreadProj = dr.getCandlesLive()
    test_data = dr.dfGet('livecandlesOUT.csv', live=True).reshape(-1,1)

    scaler = MinMaxScaler()

    pipeline = Pipeline([
                ('normalization', MinMaxScaler())
            ])

    pipeline = joblib.load(dr.PATH + 'v20-python-samples/models/EUR_USD/history/scaler/'+ datefiles[-1][:-4] +'.joblib') 
    test_data = pipeline.transform(test_data).reshape(-1)

    X_test = []
    y_test = []
    for i in range(dr.SEQ_LEN, len(test_data)- (dr.FUTURE_PERIOD_PREDICT - 1)):
        X_test.append(test_data[i-dr.SEQ_LEN:i])
        y_test.append(test_data[i+(dr.FUTURE_PERIOD_PREDICT-1)])
        
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    multi_head = dr.getModel()


    ########play with this more later to edit the csv instead of overwriting? Or just do in the callback?#####

    predicted_stock_price_multi_head = multi_head.predict(X_test)
    predicted_stock_price_multi_head.shape
    predicted_stock_price_multi_head = np.vstack((np.full(((dr.SEQ_LEN+(dr.FUTURE_PERIOD_PREDICT-1)),1), np.nan), predicted_stock_price_multi_head))


    test_dataREAL = pipeline.inverse_transform(test_data.reshape(-1,1))
    predicted_stock_price_multi_headREAL = pipeline.inverse_transform(predicted_stock_price_multi_head.reshape(-1,1))

    truex = test_dataREAL.tolist()
    predictx = predicted_stock_price_multi_headREAL.tolist()
    import itertools

    true = list(itertools.chain.from_iterable(truex))
    predict = list(itertools.chain.from_iterable(predictx))
    predictadj = []


    for l in range(dr.SEQ_LEN+(dr.FUTURE_PERIOD_PREDICT-1)):
        predictadj.append(np.nan)


    for m in range(dr.SEQ_LEN+(dr.FUTURE_PERIOD_PREDICT-1),len(predict)):       
        predictadj.append(predict[m] - (dr.PredAdjMA(predict,true,m,dr.AMA) * dr.AMAc))



    NowList = dr.getPrice()
    spreadNow = float(NowList[3]) - float(NowList[1])
    NowMid = float(NowList[2])
    spreadProjAdj = (spreadProj/2) + (spreadNow/2)
    #Totally disregarding what we got out of HPSweep, lol
    Buffer = .0005
    NowDT = datetime.datetime.now()
    #NowSL = float((TradeDict['0'])['SLPrice'])
    AMApredict = float(predictadj[-1])

    ################################################################################
    exbuffer = .0005
    print('extra buffer for SL in case we make buffer dynamic again = ' + str(exbuffer))
    ################################################################################

    print('true= ')
    print(true)
    
    print('predict = ')
    print(predict)

    print('predictadj= ')
    print(predictadj)
            

    AcctDict = dr.AcctDetails()
    Balance = float(AcctDict['balance'])
    #Why did I even make this a dict?



    #if OpenTrade == False:
    if abs(NowMid - AMApredict) > (spreadProjAdj + Buffer):
        if AMApredict > NowMid:
            Short = False
            if INVERT == True:
                Short = True
            Open = float(NowList[3])
            SL = round(Open - (Buffer + exbuffer), 5)
            print('open = ' + str(Open))
            print ('AMA = ' + str(AMApredict))
            print('SL = ' + str(SL))
            PositionUnits = int((.99 * Balance) / Open)
            if INVERT == True:
                PositionUnits *= -1
            print('position units = ' + str(PositionUnits))
            dr.MarketOrder(PositionUnits, SL)
            print('trade opened long!')

            
            
        elif AMApredict < NowMid:
            Short = True
            if INVERT == True:
                Short = False
            Open = float(NowList[1])
            SL = round(Open + (Buffer + exbuffer), 5)
            print('open = ' + str(Open))
            print ('AMA = ' + str(AMApredict))
            print('SL = ' + str(SL))
            PositionUnits = int((.99 * Balance) / Open) * -1
            if INVERT == True:
                PositionUnits *= -1
            print('position units = ' + str(PositionUnits))
            dr.MarketOrder(PositionUnits, SL)
            print('trade opened short!')

        else:
            print("whoops")

        
    
    else:
        print('no trade!')      



dr.PIDexit(PIDx)
