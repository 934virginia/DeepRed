import argparse
import common.config
import common.view
import common.args
from account import Account
from args import OrderArguments, add_replace_order_id_argument
from v20.order import MarketOrderRequest
import psutil
import signal
import time
import datetime
import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
from keras.engine.topology import Layer
import random, os, sys
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import statistics
import math



try:
    from dataloader import TokenList, pad_to_longest
    # for transformer
except: pass

class RedObj:
    def __init__(self):
        self.SEQ_LEN= 30
        self.FUTURE_PERIOD_PREDICT= 5
        self.BATCH_SIZE= 32
        self.INSTRUMENT= 'EUR_USD'
        self.EPOCHS= 50
        self.M_SCALE= 1
        self.DST_OFFSET= 9
        self.PATH= '/home/DeepRed/'
        self.AMA= None
        self.AMAc= None
        self.GRANULARITY = 'M' + str(self.M_SCALE)
        self.RATIO_TO_PREDICT = 'mid'
        self.X_VALID = None
        self.X_TRAIN = None

#####Some data grabbing stuff modified from included API examples.#####

    class CandleWriter(object):
        def __init__(self):
            self.width = {
                'time' : 19,
                'type' : 4,
                'price' : 8,
                'volume' : 6,
            }
            # setattr(self.width, "time", 19)
            self.time_width = 19

        def write_header(self, OutFile):
            with open(OutFile, 'w') as f:
                f.write("Date;Type;Open;High;Low;Close;Volume\n")
            
        def write_candle(self, candle, OutFile):
            try:
                time = str(
                    datetime.datetime.strptime(
                        candle.time,
                        "%Y-%m-%dT%H:%M:%S.000000000Z"
                    )
                )
            except:
                time = candle.time.split(".")[0]

            volume = candle.volume

            for price in ["mid", "bid", "ask"]:
                c = getattr(candle, price, None)

                if c is None:
                    continue

                with open(OutFile, 'a') as f:
                    f.write(time + ";" + str(price) + ";" + str(c.o) + ";" + str(c.h) + ";" + str(c.l) + ";" + str(c.c) + ";" + str(volume) + "\n")

                volume = ""
                time = ""
    def print_order_create_response_transactions(self, response):
        """
        Print out the transactions found in the order create response
        """

        common.view.print_response_entity(
            response, None,
            "Order Create",
            "orderCreateTransaction"
        )

        common.view.print_response_entity(
            response, None,
            "Long Order Create",
            "longOrderCreateTransaction"
        )

        common.view.print_response_entity(
            response, None,
            "Short Order Create",
            "shortOrderCreateTransaction"
        )

        common.view.print_response_entity(
            response, None,
            "Order Fill",
            "orderFillTransaction"
        )

        common.view.print_response_entity(
            response, None,
            "Long Order Fill",
            "longOrderFillTransaction"
        )

        common.view.print_response_entity(
            response, None,
            "Short Order Fill",
            "shortOrderFillTransaction"
        )

        common.view.print_response_entity(
            response, None,
            "Order Cancel",
            "orderCancelTransaction"
        )

        common.view.print_response_entity(
            response, None,
            "Long Order Cancel",
            "longOrderCancelTransaction"
        )

        common.view.print_response_entity(
            response, None,
            "Short Order Cancel",
            "shortOrderCancelTransaction"
        )

        common.view.print_response_entity(
            response, None,
            "Order Reissue",
            "orderReissueTransaction"
        )

        common.view.print_response_entity(
            response, None,
            "Order Reject",
            "orderRejectTransaction"
        )

        common.view.print_response_entity(
            response, None,
            "Order Reissue Reject",
            "orderReissueRejectTransaction"
        )

        common.view.print_response_entity(
            response, None,
            "Replacing Order Cancel", 
            "replacingOrderCancelTransaction"
        )

    def price_to_string(self, price):
        return "{} ({}) {}/{}".format(
            price.instrument,
            price.time,
            price.bids[0].price,
            price.asks[0].price
        )
#######################################

        

    def dfGet(self, DATE, HL=None, live=False):
        #sets up dataframes from csv's. filepaths are stupid and should probably be reworked.
        
        def classify(current, future):
            if float(future) > float(current):
                return 1
            else:
                return 0
        if live == True:
            df = pd.read_csv(self.PATH + 'v20-python-samples/dailies/' + self.INSTRUMENT + '/' + DATE,delimiter=';',usecols=['Date','Type','Open','High','Low','Close','Volume'])
        else:        
            df = pd.read_csv(self.PATH + 'v20-python-samples/data/' + self.INSTRUMENT + '/' + DATE,delimiter=';',usecols=['Date','Type','Open','High','Low','Close','Volume'])

        # Sort DataFrame by date
        df = df.sort_values('Date')

        if HL == None:
            df['mid'] = (df['Low']+df['High'])/2.0
        elif HL == 'H':
            df['mid'] = df['High']
        elif HL == 'L':
            df['mid'] = df['Low']

        

        ## Step 2 - Data preprocessing 

        df['mid'] = (df['Low']+df['High'])/2.0

        df['future'] = df[self.RATIO_TO_PREDICT].shift(-self.FUTURE_PERIOD_PREDICT)

        df['target'] = list(map(classify, df[self.RATIO_TO_PREDICT], df['future']))

        times = sorted(df.index.values)  # get the times

        last_100pct = sorted(df.index.values)[-int(1*len(times))]

        data_df = df[(df.index >= last_100pct)] ##Makes a dataframe array out of the test data


        data_df.drop(columns=["Date", "Type", "future", 'Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

        return_data = data_df[self.RATIO_TO_PREDICT].values

        return return_data

    def PredAdjMA(self, predictData, trueData, T, period):
        #function for adjusting predictions based on how bad they've been recently. Basically the core of the janky Hyperparameter system.
        
        diff = 0
        for n in range(period):
            
            diffX = (predictData[T-(self.FUTURE_PERIOD_PREDICT+(period-n))] - trueData[T-(self.FUTURE_PERIOD_PREDICT+(period-n))])
            if np.isnan(diffX) == True:
                diffX = 0
            diff += diffX
        try:
            return (diff/period) * 1
        except ZeroDivisionError:
            return 0

    def PIDexit(self, PIDx):
        os.remove(self.PATH + 'v20-python-samples/PID/' + self.INSTRUMENT + '/' + str(PIDx) + '.csv')
        exit()

    PID = os.getpid()

    def PIDwrite(self, PIDx):
        #Process ID functions to make sure only one instance of the script is running; lets me fob everything off on Cron fairly reliably.
        with open(self.PATH + 'v20-python-samples/PID/' + self.INSTRUMENT + '/' + str(PIDx) + '.csv', 'w') as f:
            f.write(str(datetime.datetime.now()))


    def PIDcheck(self):
        PIDfiles = sorted(os.listdir(self.PATH + 'v20-python-samples/PID/' + self.INSTRUMENT + '/'), key = str)

        for i in range(len(PIDfiles)):
            with open(self.PATH + 'v20-python-samples/PID/' + self.INSTRUMENT + '/' + str(PIDfiles[i]), 'r') as g:
                PIDtime = datetime.datetime.strptime(g.read(), '%Y-%m-%d %H:%M:%S.%f')
                PIDNOW = int(str(PIDfiles[i][:-4]))
                if psutil.pid_exists(PIDNOW):
                    print("a process with pid %d exists" % PIDNOW)
                    if PIDtime < (datetime.datetime.now() - datetime.timedelta(hours=4)):
                        print('Process over 4h old; killing process and deleting PID file')
                        #lol 4 hours
                        os.kill(PIDNOW,signal.SIGSTOP)
                        os.remove(self.PATH + 'v20-python-samples/PID/' + self.INSTRUMENTx + '/' + str(PIDfiles[i]))
                    else:
                        print('Process not 4h old; exiting program.')
                        exit()
                else:
                    print("a process with pid %d does not exist, removing PID file" % PIDNOW)
                    os.remove(self.PATH + 'v20-python-samples/PID/' + self.INSTRUMENT + '/' + str(PIDfiles[i]))


    def MarketOrder(self, UNITSx, SLx=0):
        #Place a market order with an optional stop loss order attached. Should abandon altogether and do limit orders with layered execution of entrances.
        
        UNITS = str(UNITSx)
        #TSL = str(TSLx)
        SL = str(SLx)    

        """
        Create a Market Order in an Account based on the provided command-line
        arguments.
        """

        parser = argparse.ArgumentParser()

        ###############################

        ##############################

        #
        # Add the command line argument to parse to the v20 config
        #
        common.config.add_argument(parser)
        #
        # Add the command line arguments required for a Market Order
        #

        marketOrderArgs = OrderArguments(parser)
        marketOrderArgs.add_instrument()
        '''
        parser.add_argument(
                "self.INSTRUMENT",
                type=common.args.self.INSTRUMENT,
                #default='EUR_USD',
                help="The self.INSTRUMENT to place the Order for"
            )
        '''
        marketOrderArgs.add_units()
        '''
        parser.add_argument(
                "units",
                #type=int,
                #default=UNITS,
                help=(
                    "The number of units for the Order. "
                    "Negative values indicate sell, Positive values indicate buy"
                )
            )
        '''
        marketOrderArgs.add_time_in_force(["FOK", "IOC"])
        marketOrderArgs.add_price_bound()
        #marketOrderArgs.add_distance()
        marketOrderArgs.add_position_fill()
        marketOrderArgs.add_take_profit_on_fill()
        marketOrderArgs.add_stop_loss_on_fill()
        marketOrderArgs.add_trailing_stop_loss_on_fill()
        marketOrderArgs.add_client_order_extensions()
        marketOrderArgs.add_client_trade_extensions()


        args = parser.parse_args(args=[self.INSTRUMENT, UNITS])


        #
        # Create the api context based on the contents of the
        # v20 config file
        #
        api = args.config.create_context()

        #
        # Extract the Market order parameters from the parsed arguments
        #
        marketOrderArgs.parse_arguments(args)

        #
        # Submit the request to create the Market Order
        #
        response = api.order.market(
            args.config.active_account,
            **marketOrderArgs.parsed_args
        )

        print("Response: {} ({})".format(response.status, response.reason))
        print("")

        self.print_order_create_response_transactions(response)

    def getPrice(self):
        #returns current prices in the form of a list, for some reason. List populates as [MID,BID,ASK,SPREAD]
        """
        Get the prices for a list of self.INSTRUMENTs for the active Account.
        Repeatedly poll for newer prices if requested.
        """

        parser = argparse.ArgumentParser()

        common.config.add_argument(parser)

        parser.add_argument(
            '--instrument', "-i",
            type=common.args.instrument,
            required=True,
            action="append",
            help="instrument to get prices for"
        )

        parser.add_argument(
            '--poll', "-p",
            action="store_true",
            default=False,
            help="Flag used to poll repeatedly for price updates"
        )

        parser.add_argument(
            '--poll-interval',
            type=float,
            default=2,
            help="The interval between polls. Only relevant polling is enabled"
        )

        args = parser.parse_args(args=[str('-i=' + self.INSTRUMENT)])

        account_id = args.config.active_account

        api = args.config.create_context()

        latest_price_time = None 

        def poll(latest_price_time):
            """
            Fetch and display all prices since than the latest price time

            Args:
                latest_price_time: The time of the newest Price that has been seen

            Returns:
                The updated latest price time
            """

            response = api.pricing.get(
                account_id,
                instruments=",".join(args.instrument),
                since=latest_price_time,
                includeUnitsAvailable=False
            )

            #
            # Print out all prices newer than the lastest time 
            # seen in a price
            #
            for price in response.get("prices", 200):
                if latest_price_time is None or price.time > latest_price_time:
                    retVal = []
                    retVal.append(abs(float((self.price_to_string(price).partition(') ')[2]).partition('/')[2]) - float((self.price_to_string(price).partition(') ')[2]).partition('/')[0])))
                    retVal.append(float((self.price_to_string(price).partition(') ')[2]).partition('/')[0]))
                    retVal.append((float((self.price_to_string(price).partition(') ')[2]).partition('/')[0]) + float((self.price_to_string(price).partition(') ')[2]).partition('/')[2])) / 2)
                    retVal.append(float((self.price_to_string(price).partition(') ')[2]).partition('/')[2]))

                    print(retVal)

                    return retVal
                    ##Do this as a dict later?##
                                  
                    '''
                    if MBAS == 'M':
                        retVal = (float((self.price_to_string(price).partition(') ')[2]).partition('/')[0]) + float((self.price_to_string(price).partition(') ')[2]).partition('/')[2])) / 2
                        print(retVal)
                        return retVal
                    elif MBAS == 'B':
                        retVal = float((self.price_to_string(price).partition(') ')[2]).partition('/')[0])
                        print(retVal)
                        return retVal
                    elif MBAS == 'A':
                        retVal = float((self.price_to_string(price).partition(') ')[2]).partition('/')[2])
                        print(retVal)
                        return retVal
                    elif MBAS == 'S':
                        retVal = abs(float((self.price_to_string(price).partition(') ')[2]).partition('/')[2]) - float((self.price_to_string(price).partition(') ')[2]).partition('/')[0]))
                        print(retVal)
                        return retVal
                    '''
                    


            #
            # Stash and return the current latest price time
            #
            for price in response.get("prices", 200):
                if latest_price_time is None or price.time > latest_price_time:
                    latest_price_time = price.time

            #return latest_price_time

        #
        # Fetch the current snapshot of prices
        #
        latest_price_time = poll(latest_price_time)
        
        return latest_price_time

        #
        # Poll for of prices
        #
        while args.poll:
            time.sleep(args.poll_interval)
            latest_price_time = poll(latest_price_time)

    def getTradesInst(self):
        #Spits out a dict of dicts for each open position. I actually don't hate this.
        """
        Get details of a specific Trade or all open Trades in an Account,
        then returns them in a dict
        """

        parser = argparse.ArgumentParser()

        #
        # Add the command line argument to parse to the v20 config
        #
        common.config.add_argument(parser)

        parser.add_argument(
            "--trade-id", "-t",
            help=(
                "The ID of the Trade to get. If prepended "
                "with an '@', this will be interpreted as a client Trade ID"
            )
        )

        parser.add_argument(
            "--all", "-a",
            action="store_true",
            default=False,
            help="Flag to get all open Trades in the Account"
        )

        parser.add_argument(
            "--summary", "-s",
            dest="summary",
            action="store_true",
            help="Print Trade summary instead of full details",
            default=True
        )

        parser.add_argument(
            "--verbose", "-v",
            dest="summary",
            help="Print Trade details instead of summary",
            action="store_false"
        )

        args = parser.parse_args(args=['-a','-v'])

        if args.trade_id is None and not args.all:
            parser.error("Must provide --trade-id or --all")

        account_id = args.config.active_account

        #
        # Create the api context based on the contents of the
        # v20 config file
        #
        api = args.config.create_context()
        content = []

        if args.all:
            response = api.trade.list_open(account_id)

            if not args.summary:
                pass
                
            for trade in reversed(response.get("trades", 200)):
                if args.summary:
                    content.append(trade.title())
                else:
                    content.append(trade.yaml(True))


        elif args.trade_id:
            response = api.trade.get(account_id, args.trade_id)

            trade = response.get("trade", 200)

            if args.summary:
                content.append(trade.title())

            else:
                content.append(trade.yaml(True))


        tradestring =  str(content)
        '''
        with open('tradestring', 'r') as f:
            tradestring = f.read()
            print(tradestring)
            tradestrings = tradestring.split('\\n')
            print(len(tradestrings))
        '''


        tradestrings = tradestring.split('\\n')
        print(tradestring)

        DictCount = 0

        DictKeys = [
            'TradeID',
            'self.INSTRUMENT',
            'Fill',
            'OpenDT',
            'Status',
            'InitialUnits',
            'InitialMargin',
            'CurrentUnits',
            'RealizedPL',
            'UnrealizedPL',
            'MarginUsed',
            'Financing',
            #'StopLossOrder',
            #'SLOrderID',
            #'SLOpenDT',
            #'SLStatus',
            #'SLType',
            #'SLID2',
            #'SLPrice',
            #'TimeInForce',
            #'Trigger',
        ]
        
        TradeDict = {}

        for i in range(len(tradestrings)):
            if tradestrings[i].partition(": ")[2] == self.INSTRUMENT:
                k = {}
                for j in range(len(DictKeys)):
                    k[DictKeys[j]] = tradestrings[(i-1)+j].partition(": ")[2]
                TradeDict = {str(DictCount) : k
                }
                DictCount +=1

        return TradeDict


        ##MAKE A DICTIONARY-- DO IT FOR GETP AS WELL##
        ##Start with the one trade,
        ###Make scalable so can be grabbed and handled by self.INSTRUMENT?

    def CloseMarketOrder(self, longshortx):
        #Closes an open position entirely. Rework this to allow for layered exits, and then just make it generate longshortx dynamically by calling getTradesInst
        if longshortx == False:
            longshort = '--long-units=ALL'
        elif longshortx == True:
            longshort = '--short-units=ALL'
        """
        Close an open Trade in an Account
        """

        #view.print_positions()
        
        parser = argparse.ArgumentParser()

        #
        # Add the command line argument to parse to the v20 config
        #
        common.config.add_argument(parser)

        parser.add_argument(
            "self.INSTRUMENT",
            type=common.args.instrument,
            help=(
                "The instrument of the Position to close. If prepended "
                "with an '@', this will be interpreted as a client Trade ID"
            )
        )

        parser.add_argument(
            "--long-units",
            default=None,
            help=(
                "The amount of the long Position to close. Either the string 'ALL' "
                "indicating a full Position close, the string 'NONE', or the "
                "number of units of the Position to close"
            )
        )

        parser.add_argument(
            "--short-units",
            default=None,
            help=(
                "The amount of the short Position to close. Either the string "
                "'ALL' indicating a full Position close, the string 'NONE', or the "
                "number of units of the Position to close"
            )
        )

        #DID YOU LONG OR SHORT IT?
        args = parser.parse_args(args=[self.INSTRUMENT, longshort])

        account_id = args.config.active_account

        #
        # Create the api context based on the contents of the
        # v20 config file
        #
        api = args.config.create_context()

        if args.long_units is not None and args.short_units is not None:
            response = api.position.close(
                account_id,
                args.instrument,
                longUnits=args.long_units,
                shortUnits=args.short_units
            )
        elif args.long_units is not None:
            response = api.position.close(
                account_id,
                args.instrument,
                longUnits=args.long_units
            )
        elif args.short_units is not None:
            response = api.position.close(
                account_id,
                args.instrument,
                shortUnits=args.short_units
            )
        else:
            print("No units have been provided");
            return

        print(
            "Response: {} ({})\n".format(
                response.status,
                response.reason
            )
        )

        self.print_order_create_response_transactions(response)

    def DSTCheck(self):
        x = time.localtime()

        #print(x)
        if x.tm_isdst == 1:
            return True
        else:
            return False

    def AcctDetails(self):
        #straightforward dump of account info. Used to determine position sizes when opening positions.
        """
        Create an API context, and use it to fetch and display the state of an
        Account.

        The configuration for the context and Account to fetch is parsed from the
        config file provided as an argument.
        """

        parser = argparse.ArgumentParser()

        #
        # The config object is initialized by the argument parser, and contains
        # the REST APID host, port, accountID, etc.
        #
        common.config.add_argument(parser)

        args = parser.parse_args()

        account_id = args.config.active_account

        #
        # The v20 config object creates the v20.Context for us based on the
        # contents of the config file.
        #
        api = args.config.create_context()

        #
        # Fetch the details of the Account found in the config file
        #
        response = api.account.get(account_id)

        #
        # Extract the Account representation from the response.
        #
        account = Account(
            response.get("account", "200")
        )

        rawstring = str(account.dump1()).split('\n')
        AcctDict = {}
        #xxz = []

        for i in range(len(rawstring)):
            label = str(rawstring[i].partition(': ')[0])
            balance = str(rawstring[i].partition(': ')[2])
            AcctDict[label] = balance
            
        
        print(AcctDict)
        return AcctDict
        #Srsly? Why make this a dict, though?

    '''
    if __name__ == "__main__":
        AcctDict = main()
        print(AcctDict['balance'])
    '''


    def getModel(self, datefiles = None, train=False):
        #train or rebuild the Keras model. Should maybe wrap the profit testing in a subclass since it shows up elsewhere and is kind of useful for experimenting.
        #This is also where the lines blur between code I've written and the original model that I snatched from Shujian on Kaggle and started messing with.
        #LINK: https://www.kaggle.com/shujian/transformer-with-lstm
        
        if datefiles == None:
            datefiles = sorted(os.listdir(self.PATH + 'v20-python-samples/data/' + self.INSTRUMENT + '/'), key = str)
            datefiles.pop(-1)

        TRAINING_DAYS = self.M_SCALE * 10

        totalprofitglobal = 100



        ###################################################


        #################################################################

        #print(train_data)
        data_parts = []
        dataindex = 1
        blanks = 0
        while dataindex <= TRAINING_DAYS+1:
            if (os.stat(self.PATH + 'v20-python-samples/data/' + self.INSTRUMENT + '/' + datefiles[(-dataindex - blanks)]).st_size) < 200: #<40 originally
                blanks+=1
            else:
                data_parts.append(self.dfGet(datefiles[-dataindex - blanks]))
                dataindex+=1

        print(len(data_parts))
        data_parts.reverse()

        data_parts[0]
        pjoined = []

        for i in range(TRAINING_DAYS):
            pjoined = np.concatenate((pjoined, data_parts[i]), axis=0)

        train_data = pjoined.reshape(-1,1)

        print('---')
        print(data_parts[-1])
        print('---')

        valid_data = data_parts[-1].reshape(-1,1)
        test_data = data_parts[-1].reshape(-1,1)

        scaler = MinMaxScaler()

        pipeline = Pipeline([
                    ('normalization', MinMaxScaler())
                ])

        # Train the Scaler with training data and smooth data
        smoothing_window_size = int(len(train_data))
        chunk_sum = int(len(train_data) / smoothing_window_size) * smoothing_window_size

        for di in range(0,chunk_sum,smoothing_window_size):
            pipeline.fit(train_data[di:di+smoothing_window_size,:])
            train_data[di:di+smoothing_window_size,:] = pipeline.transform(train_data[di:di+smoothing_window_size,:])
            

        joblib.dump(pipeline, self.PATH + 'v20-python-samples/models/EUR_USD/history/scaler/'+ datefiles[-1][:-4] +'.joblib')
        
        # Reshape both train and test data
        train_data = train_data.reshape(-1)

        # Normalize test data and validation data based on scaling of train data
        valid_data = pipeline.transform(valid_data).reshape(-1)
        test_dataRAW = test_data
        test_data = pipeline.transform(test_data).reshape(-1)

        # Now perform exponential moving average smoothing
        # So the data will have a smoother curve than the original ragged data
        EMA = 0.0
        gamma = 0.1
        for ti in range(len(train_data)):
            EMA = gamma*train_data[ti] + (1-gamma)*EMA
            train_data[ti] = EMA

        # Used for visualization and test purposes
        all_mid_data = np.concatenate([train_data,valid_data, test_data],axis=0)

        X_train = []
        y_train = []

        for i in range(self.SEQ_LEN, len(train_data)- (self.FUTURE_PERIOD_PREDICT - 1)):
            X_train.append(train_data[i-self.SEQ_LEN:i])
            y_train.append(train_data[i + (self.FUTURE_PERIOD_PREDICT-1)])

        X_train, y_train = np.array(X_train), np.array(y_train)


        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


        X_valid = []
        y_valid = []
        for i in range(self.SEQ_LEN, len(valid_data)- (self.FUTURE_PERIOD_PREDICT - 1)):
            X_valid.append(valid_data[i-self.SEQ_LEN:i])
            y_valid.append(valid_data[i+(self.FUTURE_PERIOD_PREDICT-1)])
        X_valid, y_valid = np.array(X_valid), np.array(y_valid)

        X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
        self.X_VALID = X_valid

        X_test = []
        y_test = []
        for i in range(self.SEQ_LEN, len(test_data)- (self.FUTURE_PERIOD_PREDICT - 1)):
            X_test.append(test_data[i-self.SEQ_LEN:i])
            y_test.append(test_data[i+(self.FUTURE_PERIOD_PREDICT-1)])
            
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        self.X_TEST = X_test

        y_train.shape

        y_valid.shape

        X_train_2 = []
        y_train_2 = []
        for i in range(self.SEQ_LEN, len(train_data)- (self.FUTURE_PERIOD_PREDICT - 1)):
            X_train_2.append(train_data[i-self.SEQ_LEN:i])
            y_train_2.append(train_data[i + (self.FUTURE_PERIOD_PREDICT-1)])
        X_train_2, y_train_2 = np.array(X_train_2), np.array(y_train_2)

        X_train_2 = np.reshape(X_train_2, (X_train_2.shape[0], X_train_2.shape[1], 1))

        X_train_3, y_train_3 = X_train, y_train
        X_train, y_train = shuffle(X_train, y_train)


        NAME = "{self.SEQ_LEN}-SEQ-{self.FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"  # a unique name for the model

        #embed size = 60
        embed_size = self.SEQ_LEN

        class LayerNormalization(Layer):
            def __init__(self, eps=1e-6, **kwargs):
                self.eps = eps
                super(LayerNormalization, self).__init__(**kwargs)
            def build(self, input_shape):
                self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                             initializer=Ones(), trainable=True)
                self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                            initializer=Zeros(), trainable=True)
                super(LayerNormalization, self).build(input_shape)
            def call(self, x):
                mean = K.mean(x, axis=-1, keepdims=True)
                std = K.std(x, axis=-1, keepdims=True)
                return self.gamma * (x - mean) / (std + self.eps) + self.beta
            def compute_output_shape(self, input_shape):
                return input_shape

        class ScaledDotProductAttention():
            def __init__(self, d_model, attn_dropout=0.1):
                self.temper = np.sqrt(d_model)
                self.dropout = Dropout(attn_dropout)
            def __call__(self, q, k, v, mask):
                attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
                if mask is not None:
                    mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
                    attn = Add()([attn, mmask])
                attn = Activation('softmax')(attn)
                attn = self.dropout(attn)
                output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
                return output, attn

        class MultiHeadAttention():
            # mode 0 - big matrixes, faster; mode 1 - more clear implementation
            def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
                self.mode = mode
                self.n_head = n_head
                self.d_k = d_k
                self.d_v = d_v
                self.dropout = dropout
                if mode == 0:
                    self.qs_layer = Dense(n_head*d_k, use_bias=False)
                    self.ks_layer = Dense(n_head*d_k, use_bias=False)
                    self.vs_layer = Dense(n_head*d_v, use_bias=False)
                elif mode == 1:
                    self.qs_layers = []
                    self.ks_layers = []
                    self.vs_layers = []
                    for _ in range(n_head):
                        self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                        self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                        self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
                self.attention = ScaledDotProductAttention(d_model)
                self.layer_norm = LayerNormalization() if use_norm else None
                self.w_o = TimeDistributed(Dense(d_model))

            def __call__(self, q, k, v, mask=None):
                d_k, d_v = self.d_k, self.d_v
                n_head = self.n_head

                if self.mode == 0:
                    qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
                    ks = self.ks_layer(k)
                    vs = self.vs_layer(v)

                    def reshape1(x):
                        s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
                        x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                        x = tf.transpose(x, [2, 0, 1, 3])  
                        x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                        return x
                    qs = Lambda(reshape1)(qs)
                    ks = Lambda(reshape1)(ks)
                    vs = Lambda(reshape1)(vs)

                    if mask is not None:
                        mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
                    head, attn = self.attention(qs, ks, vs, mask=mask)  
                        
                    def reshape2(x):
                        s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
                        x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
                        x = tf.transpose(x, [1, 2, 0, 3])
                        x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
                        return x
                    head = Lambda(reshape2)(head)
                elif self.mode == 1:
                    heads = []; attns = []
                    for i in range(n_head):
                        qs = self.qs_layers[i](q)   
                        ks = self.ks_layers[i](k) 
                        vs = self.vs_layers[i](v) 
                        head, attn = self.attention(qs, ks, vs, mask)
                        heads.append(head); attns.append(attn)
                    head = Concatenate()(heads) if n_head > 1 else heads[0]
                    attn = Concatenate()(attns) if n_head > 1 else attns[0]

                outputs = self.w_o(head)
                outputs = Dropout(self.dropout)(outputs)
                if not self.layer_norm: return outputs, attn
                # outputs = Add()([outputs, q]) # sl: fix
                ##Still no idea what this is.##
                return self.layer_norm(outputs), attn

        class PositionwiseFeedForward():
            def __init__(self, d_hid, d_inner_hid, dropout=0.1):
                self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
                self.w_2 = Conv1D(d_hid, 1)
                self.layer_norm = LayerNormalization()
                self.dropout = Dropout(dropout)
            def __call__(self, x):
                output = self.w_1(x) 
                output = self.w_2(output)
                output = self.dropout(output)
                output = Add()([output, x])
                return self.layer_norm(output)

        class EncoderLayer():
            def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
                self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
                self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
            def __call__(self, enc_input, mask=None):
                output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
                output = self.pos_ffn_layer(output)
                return output, slf_attn


        def GetPosEncodingMatrix(max_len, d_emb):
            pos_enc = np.array([
                [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
                if pos != 0 else np.zeros(d_emb) 
                    for pos in range(max_len)
                    ])
            pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
            pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
            return pos_enc

        def GetPadMask(q, k):
            ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
            mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
            mask = K.batch_dot(ones, mask, axes=[2,1])
            return mask

        def GetSubMask(s):
            len_s = tf.shape(s)[1]
            bs = tf.shape(s)[:1]
            mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
            return mask

        class Transformer():
            def __init__(self, len_limit, embedding_matrix, d_model=embed_size, \
                      d_inner_hid=512, n_head=10, d_k=64, d_v=64, layers=2, dropout=0.1, \
                      share_word_emb=False, **kwargs):
                self.name = 'Transformer'
                self.len_limit = len_limit
                self.src_loc_info = False # True # sl: fix later
                self.d_model = d_model
                self.decode_model = None
                d_emb = d_model

                pos_emb = Embedding(len_limit, d_emb, trainable=False, \
                                    weights=[GetPosEncodingMatrix(len_limit, d_emb)])

                i_word_emb = Embedding(max_features, d_emb, weights=[embedding_matrix]) # Add Kaggle provided embedding here

                self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, \
                                       word_emb=i_word_emb, pos_emb=pos_emb)

                
            def get_pos_seq(self, x):
                mask = K.cast(K.not_equal(x, 0), 'int32')
                pos = K.cumsum(K.ones_like(x, 'int32'), 1)
                return pos * mask

            def compile(self, active_layers=999):
                src_seq_input = Input(shape=(None, ))
                x = Embedding(max_features, embed_size, weights=[embedding_matrix])(src_seq_input)
                
                # LSTM before attention layers
                x = Bidirectional(LSTM(128, return_sequences=True))(x)
                x = Bidirectional(LSTM(64, return_sequences=True))(x) 
                
                x, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.1)(x, x, x)
                
                avg_pool = GlobalAveragePooling1D()(x)
                max_pool = GlobalMaxPooling1D()(x)
                conc = concatenate([avg_pool, max_pool])
                conc = Dense(64, activation="relu")(conc)
                ##linear works better than sigmoid for price breakouts##
                #x = Dense(1, activation="sigmoid")(conc)
                x = Dense(1, activation="linear")(conc)  
                
                
                self.model = Model(inputs=src_seq_input, outputs=x)
                self.model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

        def build_model():
            inp = Input(shape = (self.SEQ_LEN, 1))
            
            # LSTM before attention layers
            x = Bidirectional(LSTM(128, return_sequences=True))(inp)
            x = Bidirectional(LSTM(64, return_sequences=True))(x) 
                
            x, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.1)(x, x, x)
                
            avg_pool = GlobalAveragePooling1D()(x)
            max_pool = GlobalMaxPooling1D()(x)
            conc = concatenate([avg_pool, max_pool])
            conc = Dense(64, activation="relu")(conc)
            ##More sigmoid//linear nonsense##
            #x = Dense(1, activation="sigmoid")(conc)
            x = Dense(1, activation="linear")(conc) 

            model = Model(inputs = inp, outputs = x)
            model.compile(
                loss = "mean_squared_error", 
                #optimizer = Adam(lr = config["lr"], decay = config["lr_d"]), 
                optimizer = "adam")
            
            ##This doesn't work. Can't pickle lambda layers. Save the weights instead, load weights for each currency pair.##
            #model.save('my_model.h5')

            model.save_weights(self.PATH + 'v20-python-samples/models/' + self.INSTRUMENT + '/' + (datefiles[-1])[:-4] + '.h5')
            
            return model

        def rebuild_model(fname):
            inp = Input(shape = (self.SEQ_LEN, 1))
            
            # LSTM before attention layers
            x = Bidirectional(LSTM(128, return_sequences=True))(inp)
            x = Bidirectional(LSTM(64, return_sequences=True))(x) 
                
            x, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.1)(x, x, x)
                
            avg_pool = GlobalAveragePooling1D()(x)
            max_pool = GlobalMaxPooling1D()(x)
            conc = concatenate([avg_pool, max_pool])
            conc = Dense(64, activation="relu")(conc)
            ##More sigmoid//linear nonsense##
            #x = Dense(1, activation="sigmoid")(conc)
            x = Dense(1, activation="linear")(conc) 

            model = Model(inputs = inp, outputs = x)
            model.compile(
                loss = "mean_squared_error", 
                optimizer = "adam")

            #If you ever work out how to pickle the whole thing with lambda layers, load the entire h5 instead of defining structure and loading weights.
            model.load_weights(fname)
            
            return model

        #######BUILD IT################
        multi_head = build_model()
        ##REBUILDS LAST SAVED MODEL AFTER TRAINING FINISHES TO AVOID OVERFITTING-- MAKES DATA REPRODUCIBLE!##############
        multi_head = rebuild_model(self.PATH + 'v20-python-samples/models/' + self.INSTRUMENT + '/' + (datefiles[-1])[:-4] + '.h5')
        #################################################################################################################
        if train == False:
            
            #This is hideous. Just grab the AMA values in the execution script, I guess. These AMA hyperparams are all going on the chopping block anyway.
            '''
            try:
                with open(self.PATH + 'v20-python-samples/models/' + self.INSTRUMENT + '/history/HP/' + str(datetime.date.today() - datetime.timedelta(days=1)) + '.csv', 'r') as jjj:
                    HPstring = jjj.read()[1:-1]
                    HPlist = []
                    for i in range(1, len(HPstring.split(', '))):
                        HPlist.append(HPstring.split(', ')[i])

                    self.AMA = int(HPlist[1])
                    self.AMAc = float(HPlist[2])
                    print('AMA = ' + str(self.AMA))
                    print('AMAc = ' + str(self.AMAc))
            except FileNotFoundError:
                pass
            '''
            
            return multi_head
        else:         
            ##PRINTS A BUNCH OF TECHNICAL INFO ABOUT THE MODEL##
            #multi_head.summary()

            checkpoint = ModelCheckpoint(self.PATH + 'v20-python-samples/models/' + self.INSTRUMENT + '/' + (datefiles[-1])[:-4] + '.h5',
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=True,
                                            mode='auto',
                                            period=1
                                            )

            lr_reduce = ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.1,
                                          patience=10,
                                          verbose=1,
                                          mode='auto',
                                          min_delta=0.0001,
                                          cooldown=0,
                                          min_lr=0)

            EarlyStop = EarlyStopping(monitor='val_loss',
                                                    min_delta=0,
                                                    patience=20,
                                                    verbose=0,
                                                    mode='auto',
                                                    baseline=None,
                                                    restore_best_weights=False)

            def getHist():
                history = multi_head.fit(X_train, y_train,
                                            batch_size=self.BATCH_SIZE,
                                            epochs=self.EPOCHS,
                                            validation_data=(X_valid, y_valid),
                                            callbacks = [checkpoint],
                                            ###These callbacks are legit, but made accuracy a bit less consistent.
                                            #callbacks = [checkpoint , lr_reduce],
                                            #callbacks = [checkpoint , EarlyStop],
                                            verbose = 2,
                                         )
                return history

            ##UNCOMMENT THIS TO TRAIN IT.##                              )
            history = getHist()

            ########play with this more later to edit the csv instead of overwriting? Or just do in the callback?#####

            def writehistory(History):
                hist_df = pd.DataFrame(History.history) 

                hist_csv_file = self.PATH + 'v20-python-samples/models/'+ self.INSTRUMENT + '/history/' + datefiles[-1]
                with open(hist_csv_file, mode='w+') as f:
                    hist_df.to_csv(f)
                    f.seek(0)
                    g = f.readlines()
                    g[0] = str("epoch" + g[0])
                    f.seek(0)
                    f.truncate()
                    f.writelines(g)
                            
            ##############################################################################################
                    
            writehistory(history)
            
            

            ##Super basic visualization stuff. Seems like these same methods are littered everywhere; explore making it a function.

            
            predicted_stock_price_multi_head = multi_head.predict(X_test)
            predicted_stock_price_multi_head.shape

            predicted_stock_price_multi_head = np.vstack((np.full((self.SEQ_LEN,1), np.nan), predicted_stock_price_multi_head))

            test_dataREAL = pipeline.inverse_transform(test_data.reshape(-1,1))
            predicted_stock_price_multi_headREAL = pipeline.inverse_transform(predicted_stock_price_multi_head.reshape(-1,1))

            
            plt.figure(figsize=(15, 5))

            plt.plot(np.arange(y_train_2.shape[0]), y_train_2, color='blue', label='train target')

            plt.plot(np.arange(y_train_2.shape[0], y_train_2.shape[0]+y_valid.shape[0]), y_valid,
                     color='gray', label='valid target')

            plt.plot(np.arange(y_train_2.shape[0]+y_valid.shape[0],
                               y_train_2.shape[0]+y_valid.shape[0]+y_test.shape[0]),
                     y_test, color='black', label='test target')


            plt.title('Separation des donnees')
            plt.xlabel('time [days]')
            plt.ylabel('normalized price')
            plt.legend(loc='best');

            plt.figure(figsize = (18,9))
            plt.plot(train_data, color = 'black', label = 'GE Stock Price')
            plt.title('GE Mid Price Prediction Doot', fontsize=30)

            plt.xlabel('Date')
            plt.ylabel('GE Mid Price')
            plt.legend(fontsize=18)



            plt.figure(figsize = (18,9))
            plt.plot(test_data, color = 'black', label = 'Normalized test Data')
            plt.plot(predicted_stock_price_multi_head, color = 'blue', label = 'Normalized Predictions')
            plt.title('GE Mid Price Prediction', fontsize=30)
            plt.xlabel('Time')
            plt.ylabel('Normalized GE Mid Price')
            plt.legend(fontsize=18)


            truex = test_dataREAL.tolist()
            predictx = predicted_stock_price_multi_headREAL.tolist()
            import itertools

            true = list(itertools.chain.from_iterable(truex))
            predict = list(itertools.chain.from_iterable(predictx))
            predictadj = []

            for l in range(self.SEQ_LEN):
                predictadj.append(np.nan)

            for m in range(self.SEQ_LEN,len(predict)):       
                predictadj.append(predict[m] - self.PredAdjMA(predict,true,m,3))
                #predictadj.append(predict[m])



            plt.figure(figsize = (18,9))
            #plt.plot(test_dataRAW, color = 'black', label = 'Raw')
            plt.plot(test_dataREAL, color = 'black', label = 'Real price')
            plt.plot(predicted_stock_price_multi_headREAL, color = 'blue', label = 'Predicted')
            plt.plot(predictadj, color = 'Purple', label = 'PredictedAdj')
            plt.title('Test Real vs Predict', fontsize=30)
            #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
            plt.xlabel('Time')
            plt.ylabel('GE Mid Price')
            plt.legend(fontsize=18)

            ##Just make this simulation garbage a subclass so it's consistent.


            profit = 0
            realizedprofit = 0
            profitpercent = 0
            totalprofit = 100
            spread = .00015
            buffer = spread * 2
            totaltrades = 0
            goodtrades = 0
            badtrades = 0
            rawgood = []
            rawbad = []
            makegood = []
            makebad = []
            stoploss = []
            i = self.SEQ_LEN
            ding = 0
            leverage = 1
            goodlist = []
            badlist = []

            for l in range(self.SEQ_LEN):
                goodlist.append(np.nan)
                badlist.append(np.nan)

            def GoodAdd(TOpen, TClose):
                for k in range(TClose - TOpen):
                    goodlist.append(true[TOpen + k])
                    badlist.append(np.nan)
            def BadAdd(TOpen, TClose):
                for k in range(TClose - TOpen):
                    badlist.append(true[TOpen + k])
                    goodlist.append(np.nan)
            def NaNAdd(T):
                goodlist.append(np.nan)
                badlist.append(np.nan)



            while i < (len(true) - self.FUTURE_PERIOD_PREDICT):

                
                try:
                    if (abs(predictadj[i + self.FUTURE_PERIOD_PREDICT] - true[i]) - (spread+buffer)) > 0:
                        totaltrades +=1
                        #print("2")
                        iopen = i
                        exitcheck = 0
                        while i < (iopen + self.FUTURE_PERIOD_PREDICT) and exitcheck != 1:
                            i +=1
                            profit = (abs(true[iopen + (i - iopen)] - true[iopen]) - spread)
                            realizedprofit = profit
                            if realizedprofit <= (-1 * spread):
                                profitpercent = realizedprofit / true[iopen]
                                profitlev = ((totalprofit * leverage) * (1 + profitpercent)) - (totalprofit * leverage)
                                stoploss.append(profitlev / totalprofit)
                                totalprofit += profitlev
                                profitlevglobal = ((totalprofitglobal * leverage) * (1 + profitpercent)) - (totalprofitglobal * leverage)
                                totalprofitglobal += profitlevglobal
                                #totaltrades +=1
                                badtrades += 1
                                exitcheck = 1
                                GoodAdd(iopen,i)
                            
                                
                        if exitcheck == 1:
                            pass
                        if i == (iopen + self.FUTURE_PgRIOD_PREDICT) and exitcheck != 1:
                            if realizedprofit <= 0:
                                profitpercent = realizedprofit / true[iopen]
                                profitlev = ((totalprofit * leverage) * (1 + profitpercent)) - (totalprofit * leverage)
                                rawbad.append(profitlev / totalprofit)
                                totalprofit += profitlev
                                profitlevglobal = ((totalprofitglobal * leverage) * (1 + profitpercent)) - (totalprofitglobal * leverage)
                                totalprofitglobal += profitlevglobal
                                badtrades += 1
                                BadAdd(iopen,i)
                                
                            elif realizedprofit > 0:
                                if realizedprofit > (buffer/2):
                                    while realizedprofit > (profit * .9):
                                        ding += 1
                                        i += 1
                                        realizedprofitminus = realizedprofit
                                        realizedprofit = (abs(true[iopen + (i - iopen)] - true[iopen]) - spread)
                                        if realizedprofit >= realizedprofitminus:
                                            profit = realizedprofit
                                    if realizedprofit > 0:
                                        profitpercent = realizedprofit / true[iopen]
                                        profitlev = ((totalprofit * leverage) * (1 + profitpercent)) - (totalprofit * leverage)
                                        makegood.append(profitlev / totalprofit)
                                        totalprofit += profitlev
                                        profitlevglobal = ((totalprofitglobal * leverage) * (1 + profitpercent)) - (totalprofitglobal * leverage)
                                        totalprofitglobal += profitlevglobal
                                        goodtrades += 1
                                        GoodAdd(iopen,i)
                                    elif realizedprofit <= 0:
                                        profitpercent = realizedprofit / true[iopen]
                                        profitlev = ((totalprofit * leverage) * (1 + profitpercent)) - (totalprofit * leverage)
                                        makebad.append(profitlev / totalprofit)
                                        totalprofit += profitlev
                                        profitlevglobal = ((totalprofitglobal * leverage) * (1 + profitpercent)) - (totalprofitglobal * leverage)
                                        totalprofitglobal += profitlevglobal
                                        badtrades += 1
                                        BadAdd(iopen,i)
                                else:
                                    profitpercent = realizedprofit / true[iopen]
                                    profitlev = ((totalprofit * leverage) * (1 + profitpercent)) - (totalprofit * leverage)
                                    rawgood.append(profitlev / totalprofit)
                                    totalprofit += profitlev
                                    profitlevglobal = ((totalprofitglobal * leverage) * (1 + profitpercent)) - (totalprofitglobal * leverage)
                                    totalprofitglobal += profitlevglobal
                                    goodtrades += 1
                                    GoodAdd(iopen,i)
                    else:
                        NaNAdd(i)
                        i+=1
                        
                except IndexError:
                    NaNAdd(i)
                    i +=1
                    pass

            rawgoodavg = 0
            rawbadavg = 0
            makegoodavg = 0
            makebadavg = 0
            stoplossavg = 0

            for i in range(len(stoploss)):
                stoplossavg += stoploss[i]
            try:
                stoplossavg /= len(stoploss)
            except ZeroDivisionError:
                pass

            for i in range(len(rawgood)):
                rawgoodavg += rawgood[i]
            try:
                rawgoodavg /= len(rawgood)
            except ZeroDivisionError:
                pass

            for i in range(len(rawbad)):
                rawbadavg += rawbad[i]
            try:
                rawbadavg /= len(rawbad)
            except ZeroDivisionError:
                pass

            for i in range(len(makegood)):
                makegoodavg += makegood[i]
            try:
                makegoodavg /= len(makegood)
            except ZeroDivisionError:
                pass

            for i in range(len(makebad)):
                makebadavg += makebad[i]
            try:
                makebadavg /= len(makebad)
            except ZeroDivisionError:
                pass

            #Good idea to run a sim after training, but this needs to be rebuilt. The graphing isn't bad, though.
            '''
            print("stoploss total = " + str(len(stoploss)))
            print("stoploss avg = " + str(stoplossavg))
            print("rawgood total = " + str(len(rawgood)))
            print("rawgood avg = " + str(rawgoodavg))
            print("rawbad total = " + str(len(rawbad)))
            print("rawbad avg = " + str(rawbadavg))
            print("makegood total = " + str(len(makegood)))
            print("makegood avg = " + str(makegoodavg))
            print("makebad total = " + str(len(makebad)))
            print("makebad avg= " + str(makebadavg))

            print("-------------------------")
            '''


            plt.plot(goodlist, color = 'green', label = 'Good')
            plt.plot(badlist, color = 'red', label = 'Bad')
            plt.savefig(self.PATH + 'v20-python-samples/models/'+ self.INSTRUMENT + '/history/vis/' + (datefiles[-1])[:-4] + 'testvis.png')
            
            #print(true)
            #print(predict)
    def getData(self, outputF, MBA=None, queryDate=None, count=None):

        if MBA == None:
            MBA = self.RATIO_TO_PREDICT
        """
        Create an API context, and use it to fetch candles for an self.INSTRUMENT.

        The configuration for the context is parsed from the config file provided
        as an argumentV
        """

        parser = argparse.ArgumentParser()

        #
        # The config object is initialized by the argument parser, and contains
        # the REST APID host, port, accountID, etc.
        #
        common.config.add_argument(parser)
        '''
        parser.add_argument(
            "instrument",
            type=common.args.instrument,
            help="The self.INSTRUMENT to get candles for"
        )

        parser.add_argument(
            "--output",
            type=str,
            help="output directory and filename"
        )

        parser.add_argument(
            "--mid", 
            action='store_true',
            help="Get midpoint-based candles"
        )

        parser.add_argument(
            "--bid", 
            action='store_true',
            help="Get bid-based candles"
        )

        parser.add_argument(
            "--ask", 
            action='store_true',
            help="Get ask-based candles"
        )

        parser.add_argument(
            "--smooth", 
            action='store_true',
            help="'Smooth' the candles"
        )

        parser.set_defaults(mid=False, bid=False, ask=False)

        parser.add_argument(
            "--granularity",
            default=None,
            help="The candles granularity to fetch"
        )

        parser.add_argument(
            "--count",
            default=None,
            help="The number of candles to fetch"
        )

        date_format = "%Y-%m-%d %H:%M:%S"

        parser.add_argument(
            "--from-time",
            default=None,
            type=common.args.date_time(),
            help="The start date for the candles to be fetched. Format is 'YYYY-MM-DD HH:MM:SS'"
        )

        parser.add_argument(
            "--to-time",
            default=None,
            type=common.args.date_time(),
            help="The end date for the candles to be fetched. Format is 'YYYY-MM-DD HH:MM:SS'"
        )

        parser.add_argument(
            "--alignment-timezone",
            default=None,
            help="The timezone to used for aligning daily candles"
        )
        '''
        args = parser.parse_args()
        

        account_id = args.config.active_account

        granularity = self.GRANULARITY
        output = outputF
        smooth = None
        #count = None
        #from_time = (queryDate) - datetime.timedelta(hours=offset)
        #to_time = (queryDate + datetime.timedelta(days=1)) - datetime.timedelta(hours=offset)
        #from_time = (datetime.datetime.combine(queryDate, datetime.datetime.min.time())) - datetime.timedelta(hours=offset)
        #to_time = from_time + datetime.timedelta(days=1)
        #alignment_timezone = 'America/Sao_Paulo'
        alignment_timezone = None
        mid = True
        bid = False
        ask = False

        #
        # The v20 config object creates the v20.Context for us based on the
        # contents of the config file.
        #
        api = args.config.create_context()

        kwargs = {}

        if granularity is not None:
            kwargs["granularity"] = self.GRANULARITY

        if output is not None:
            kwargs["output"] = output

        if smooth is not None:
            kwargs["smooth"] = smooth

        if count is not None:
            kwargs["count"] = count

        if queryDate is not None:
            from_time = (datetime.datetime.combine(queryDate, datetime.datetime.min.time())) - datetime.timedelta(hours=self.DST_OFFSET)
            to_time = from_time + datetime.timedelta(days=1)
            kwargs["fromTime"] = api.datetime_to_str(from_time)
            kwargs["toTime"] = api.datetime_to_str(to_time)

        if alignment_timezone is not None:
            kwargs["alignmentTimezone"] = alignment_timezone

        #price = "mid"

        if MBA == "mid":
            kwargs["price"] = "M" + kwargs.get("price", "")
            price = "mid"

        if MBA == "bid":
            kwargs["price"] = "B" + kwargs.get("price", "")
            price = "bid"

        if MBA == "ask":
            kwargs["price"] = "A" + kwargs.get("price", "")
            price = "ask"

        if MBA == "BA":
            kwargs["price"] = "B" + kwargs.get("price", "")
            price = "bid"
            kwargs["price"] = "A" + kwargs.get("price", "")
            price = "ask"

        #
        # Fetch the candles
        #
        #try:
        response = api.instrument.candles(self.INSTRUMENT, **kwargs)

        if response.status != 200:
            print(response)
            print(response.body)
            return

        #print("Instrument: {}".format(response.get("instrument", 200)))
        #print("Granularity: {}".format(response.get("granularity", 200)))

        writer = self.CandleWriter()

        writer.write_header(str(output))


        candles = response.get("candles", 200)

        for candle in response.get("candles", 200):
            writer.write_candle(candle, str(output))

    def getCandlesLive(self):
        #figure out how to import v20ConnectionError for these kinds of exceptions instead of this.
        
        while True:
            try:
                self.getData(str(self.PATH + 'v20-python-samples/dailies/' + self.INSTRUMENT + '/' + 'livecandles.csv'), MBA='BA', count=(self.SEQ_LEN+self.FUTURE_PERIOD_PREDICT+self.AMA))
                break
            except Exception:
                pass
        
        
        #self.getData(str(self.PATH + 'v20-python-samples/dailies/' + self.INSTRUMENT + '/' + 'livecandles.csv'), MBA='BA', count=(self.SEQ_LEN+self.FUTURE_PERIOD_PREDICT+self.AMA))

        #I think this is an AMA thing?
        spreadPeriod = 5

        with open(self.PATH + 'v20-python-samples/dailies/' + self.INSTRUMENT + '/' + 'livecandles.csv','r') as f:
            flist = f.readlines()
            
        with open(self.PATH + 'v20-python-samples/dailies/' + self.INSTRUMENT + '/' + 'livecandlesOUT.csv','w+') as g:
            g.write(flist[0])
            for i in range(1,len(flist),2):
                flinebid = flist[i]
                flineask = flist[i+1]
                g.write(flinebid.split(';')[0] + ';' + 'mid' + ';')
                for j in range(2,len(flinebid.split(';'))-1):
                    flinemid = (float(flinebid.split(';')[j]) + float(flineask.split(';')[j])) / 2
                    g.write(str(flinemid) + ';')
                g.write(flinebid.split(';')[6])
            for zz in range(self.FUTURE_PERIOD_PREDICT):
                g.write('NaN;NaN;NaN;NaN;NaN;NaN;NaN\n')
        spreadAvg = []
        spreadTrj = 0
        for k in range(len(flist)-((spreadPeriod+1) * 2),len(flist), 2):
            flinebid = flist[k]
            flineask = flist[k+1]
            spreadx = 0
            for l in range(2,len(flinebid.split(';'))-1):
                spreadx += abs(float(flineask.split(';')[l]) - float(flinebid.split(';')[l]))
                
            spreadAvg.append(spreadx/4)
        spreadNow = spreadAvg[0]

        for m in range(1,len(spreadAvg)):
            spreadTrj+= (spreadAvg[m] - spreadAvg[m-1])
            
        spreadTrj /= (len(spreadAvg)-1)

        spreadProj = spreadAvg[-1] + (spreadTrj * self.FUTURE_PERIOD_PREDICT)
        return spreadProj

    def stop_loss(self, TradeIDx, Pricex, SLIDx):
        #add a stop loss to an open trade
        TradeID = str(TradeIDx)
        SLID = str(SLIDx)
        Price = str(Pricex)

        parser = argparse.ArgumentParser()

        #
        # Add the command line argument to parse to the v20 config
        #
        common.config.add_argument(parser)

        #
        # Add the argument to support replacing an existing argument
        #
        add_replace_order_id_argument(parser)

        #
        # Add the command line arguments required for a Limit Order
        #
        orderArgs = OrderArguments(parser)
        orderArgs.add_trade_id()
        orderArgs.add_price()
        orderArgs.add_time_in_force(["GTD", "GFD", "GTC"])
        orderArgs.add_client_order_extensions()

        args = parser.parse_args(args=[TradeID, Price, str('-r'+ SLID)])
        #, str('-r='+TradeID)
        #
        # Create the api context based on the contents of the
        # v20 config file
        #
        api = args.config.create_context()

        #
        # Extract the Limit Order parameters from the parsed arguments
        #
        orderArgs.parse_arguments(args)

        if args.replace_order_id is not None:
            #
            # Submit the request to cancel and replace a Stop Loss Order
            #
            response = api.order.stop_loss_replace(
                args.config.active_account,
                args.replace_order_id,
                **orderArgs.parsed_args
            )
        else:
            #
            # Submit the request to create a Stop Loss Order
            #
            response = api.order.stop_loss(
                args.config.active_account,
                **orderArgs.parsed_args
            )

        print("Response: {} ({})".format(response.status, response.reason))
        print("")

        self.print_order_create_response_transactions(response)

    def HPTrain(self, deploy=False, datefilesx=None):
        #This is pure spaghetti. Ended up working, but because the hyperparameters are stupid, it makes very little difference.
        def HPgetLists(datefiles=datefilesx):
            #Prep all the lists for HPSweep to do its thing. Actually returns halfway coherent dicts.
            TRAINING_DAYS = self.M_SCALE * 10

            if datefiles == None:
                datefiles = sorted(os.listdir(self.PATH + 'v20-python-samples/data/' + self.INSTRUMENT + '/'), key = str)
                datefiles.pop(-1)


            data_parts = []
            BID_parts = []
            ASK_parts = []
            HIGH_parts = []
            LOW_parts = []
            dataindex = 1
            blanks = 0
            #+2 for valid and test!#
            while dataindex <= TRAINING_DAYS+2:
                if (os.stat('/home/johnny/v20-python-samples/data/' + self.INSTRUMENT + '/' + datefiles[(-dataindex - blanks)]).st_size) < 200: #<40 originally
                    blanks+=1
                else:
                    data_parts.append(self.dfGet(datefiles[-dataindex - blanks]))
                    BID_parts.append(self.dfGet('raw/bid/' + datefiles[-dataindex - blanks]))
                    ASK_parts.append(self.dfGet('raw/ask/' + datefiles[-dataindex - blanks]))
                    HIGH_parts.append(self.dfGet(datefiles[-dataindex - blanks], 'H'))
                    LOW_parts.append(self.dfGet(datefiles[-dataindex - blanks], 'L'))
                    dataindex+=1

            data_parts.reverse()
            BID_parts.reverse()
            ASK_parts.reverse()
            HIGH_parts.reverse()
            LOW_parts.reverse()

            data_parts[0]
            BID_parts[0]
            ASK_parts[0]
            HIGH_parts[0]
            LOW_parts[0]
            pjoined = []

            for i in range(TRAINING_DAYS):
                pjoined = np.concatenate((pjoined, data_parts[i]), axis=0)

            train_data = pjoined.reshape(-1,1)

            if deploy == True:
                VDataPart = -1
            else:
                #This is mostly for backtesting and HP sweeping
                VDataPart = -2
                
            valid_data = data_parts[VDataPart].reshape(-1,1)
            test_data = data_parts[-1].reshape(-1,1)
            BID_valid_data = BID_parts[VDataPart].reshape(-1,1)
            BID_test_data = BID_parts[-1].reshape(-1,1)
            ASK_valid_data = ASK_parts[VDataPart].reshape(-1,1)
            ASK_test_data = ASK_parts[-1].reshape(-1,1)
            HIGH_valid_data = HIGH_parts[VDataPart].reshape(-1,1)
            HIGH_test_data = HIGH_parts[-1].reshape(-1,1)
            LOW_valid_data = LOW_parts[VDataPart].reshape(-1,1)
            LOW_test_data = LOW_parts[-1].reshape(-1,1)

            multi_head = self.getModel()

            predicted_stock_price_multi_head = multi_head.predict(self.X_TEST)
            predicted_stock_price_multi_heady = multi_head.predict(self.X_VALID)

            predicted_stock_price_multi_head.shape
            predicted_stock_price_multi_heady.shape
            predicted_stock_price_multi_head = np.vstack((np.full((self.SEQ_LEN,1), np.nan), predicted_stock_price_multi_head))
            predicted_stock_price_multi_heady = np.vstack((np.full((self.SEQ_LEN,1), np.nan), predicted_stock_price_multi_heady))

            scaler = MinMaxScaler()

            pipeline = Pipeline([
                ('normalization', MinMaxScaler())
                ])
        
            pipeline = joblib.load(self.PATH + 'v20-python-samples/models/'+ self.INSTRUMENT + '/history/scaler/'+ datefiles[-1][:-4] +'.joblib') 

            valid_data = pipeline.transform(valid_data).reshape(-1)
            test_dataRAW = test_data
            test_data = pipeline.transform(test_data).reshape(-1)
            
            valid_dataREAL = pipeline.inverse_transform(valid_data.reshape(-1,1))
            test_dataREAL = pipeline.inverse_transform(test_data.reshape(-1,1))
            predicted_stock_price_multi_headREAL = pipeline.inverse_transform(predicted_stock_price_multi_head.reshape(-1,1))
            predicted_stock_price_multi_headREALy = pipeline.inverse_transform(predicted_stock_price_multi_heady.reshape(-1,1))

            bidx = BID_test_data.tolist()
            askx = ASK_test_data.tolist()
            highx = HIGH_test_data.tolist()
            lowx = LOW_test_data.tolist()
            bidy = BID_valid_data.tolist()
            asky = ASK_valid_data.tolist()
            highy = HIGH_valid_data.tolist()
            lowy = LOW_valid_data.tolist()
            truex = test_dataREAL.tolist()
            truey = valid_dataREAL.tolist()
            predictx = predicted_stock_price_multi_headREAL.tolist()
            predicty = predicted_stock_price_multi_headREALy.tolist()
            spreadx = []
            spready = []
            
            #No idea why this is imported here or how big it is. Relic of the original model.
            import itertools

            ValidLists = []
            TestLists = []
            
            def getVdiff(high, low):
                retList = []
                for i in range(len(low)):
                    retList.append(abs(high[i]-low[i]))
                return retList
            
            def getVscale(Vdiff):
                Vmin = 9999
                Vmax = 0
                for i in range(len(Vdiff)):
                    if Vdiff[i] > Vmax:
                        Vmax = Vdiff[i]
                    if Vdiff[i] < Vmin:
                        Vmin = Vdiff[i]
                return [Vmin, Vmax]

            def getVol(Vdiff, Vscale):
                Vol = []
                for i in range(len(Vdiff)):
                    try:
                        Vol.append((Vdiff[i] - Vscale[0]) / (Vscale[1] - Vscale[0]))
                    except ZeroDivisionError:
                        Vol.append(0)
                return Vol                   
            
            highyy =list(itertools.chain.from_iterable(highy))
            lowyy = list(itertools.chain.from_iterable(lowy))

            ValidLists.append(list(itertools.chain.from_iterable(truey)))
            ValidLists.append(list(itertools.chain.from_iterable(predicty)))
            ValidLists.append(list(itertools.chain.from_iterable(bidy)))
            ValidLists.append(list(itertools.chain.from_iterable(asky)))
            ValidLists.append(highyy)
            ValidLists.append(lowyy)
            
            Vdiffy = getVdiff(highyy, lowyy)
            ValidLists.append(getVol(Vdiffy, getVscale(Vdiffy)))
            
            for yyy in range(len(truey)):
                spready.append(abs(list(itertools.chain.from_iterable(asky))[yyy] - list(itertools.chain.from_iterable(bidy))[yyy]))

            ValidLists.append(spready)
            
            highxx =list(itertools.chain.from_iterable(highx))
            lowxx = list(itertools.chain.from_iterable(lowx))

            TestLists.append(list(itertools.chain.from_iterable(truex)))
            TestLists.append(list(itertools.chain.from_iterable(predictx)))
            TestLists.append(list(itertools.chain.from_iterable(bidx)))
            TestLists.append(list(itertools.chain.from_iterable(askx)))
            TestLists.append(highxx)
            TestLists.append(lowxx)
            
            Vdiffx = getVdiff(highxx, lowxx)
            #SCALE HAS TO BE Vdiffy#
            TestLists.append(getVol(Vdiffx, getVscale(Vdiffy)))
            
            for yyy in range(len(truex)):
                spreadx.append(abs(list(itertools.chain.from_iterable(askx))[yyy] - list(itertools.chain.from_iterable(bidx))[yyy]))

            TestLists.append(spreadx)

            if deploy == True:
                ValidLists = TestLists
            
            DictKeys = [
                'true',
                'predict',
                'bid',
                'ask',
                'high',
                'low',
                'vol',
                'spread',
            ]

            ListDictValid = {}
            ListDictTest = {}

            for i in range(len(DictKeys)):
                ListDictValid[DictKeys[i]] = ValidLists[i]
                ListDictTest[DictKeys[i]] = TestLists[i]

            MasterDict = {}

            MasterDict['Valid'] = ListDictValid
            MasterDict['Test'] = ListDictTest
            
            return MasterDict
        

        def HPSweep(MDict, recursion=True, VT='Valid', HPList=None, Rdepth=0, RDR=1, ret=False):
            #Executes a grid search for optimal hyperparameter configurations to be used IN CONJUNCTION with the trained Transformer for decision making purposes.
            #HP's were:
            #-number of trailing candles to count and average differences between predicted and true prices (AMAp)
            #-coefficient applied to adjustment derived from predAdjAMA and applied to live prediction (AMAc)
            #-Extra buffer added to predicted profit of a potential trade before execution of a MarketOrder can take place (buffer)
            
            #Basically tested model+HP configs sequentially along the validation data, using RNG seeding of execution prices within actual candle data to simulate real world variance.
            #Badly implemented recursion to narrow the scope by a factor of 10 each layer. No proof it did anything substantial other than make this look more like spaghetti.
            #Amazingly ended up doing exactly what I wanted, but grid searches are slow/wack and this ended up being pretty much incomprehensible.


            true = MDict[VT]['true']
            predict = MDict[VT]['predict']
            bid = MDict[VT]['bid']
            ask = MDict[VT]['ask']
            high = MDict[VT]['high']
            low = MDict[VT]['low']
            spread = MDict[VT]['spread']
            vol = MDict[VT]['vol']

            if VT == 'Valid':          
                print('Starting sweep')
            elif VT == 'Test':
                print('Testing')


            if recursion == True:
                wrange = 10
                vrange = 20
                urange = 10
            elif recursion == False:
                wrange = 1
                vrange = 1
                urange = 1
            bestsweep = []
            BOAT = []
            BOATprofit = 0
     
            for xxx in range(1):
                #Change the range to do it all on more RNG seeds and get best averages. Leave it at 1 for speed.
                RNGseed = xxx
                random.seed(RNGseed)
                RNGraw = []
                bestprofit = 0

                for zzz in range(len(true)):
                    try:
                        RNGraw.append((random.randrange(round((low[zzz]) * 100000), round((high[zzz]) * 100000), 1)) / 100000)
                    except ValueError:
                        RNGraw.append(low[zzz])            

                for www in range(wrange):               
                    if recursion == True:
                        if Rdepth > 0:
                            if www < (wrange/2):
                                buffer = ((HPList[0] - ((wrange/2)* (pow(.1,Rdepth)))) + (www*(pow(.1,Rdepth)))) / 2
                            elif www >= (wrange/2):
                                buffer = (HPList[0] + (((www - wrange/2)+1)*(pow(.1,Rdepth)))) / 2
                        else:    
                            buffer = (www / 2)
                    elif recursion == False:
                        buffer = (HPList[0] / 2)                    
                        
                    for vvv in range(vrange):
                        
                        for uuu in range(urange):
                            predictadj = []
                            for l in range(self.SEQ_LEN):
                                predictadj.append(np.nan)

                            for m in range(self.SEQ_LEN,len(predict)):

                                if recursion == True:
                                    if Rdepth > 0:
                                        if vvv < (vrange/2):
                                            AMAp = round(((HPList[1] - ((vrange/2)* (pow(.1,Rdepth)))) + (vvv*(pow(.1,Rdepth)))))
                                        
                                        elif vvv >= (vrange/2):
                                            AMAp = round(HPList[1] + (((vvv - vrange/2)+1)*(pow(.1,Rdepth))))

                                        if uuu < (urange/2):
                                            AMAc = ((HPList[2] - ((urange/2)* (pow(.1,Rdepth)))) + (uuu*(pow(.1,Rdepth))))
                                        elif uuu >= (urange/2):
                                            AMAc = (HPList[2] + (((uuu - urange/2)+1)*(pow(.1,Rdepth))))
                                            
                                    else:
                                        AMAp = round(vvv)
                                        
                                        AMAc = uuu
                                        
                                elif recursion == False:
                                    AMAp = round(HPList[1])
                                    
                                    AMAc = HPList[2]
                                    
                                predictadj.append(predict[m] - (self.PredAdjMA(predict,true,m,(AMAp + 1)) * ((AMAc + 1)/10) ))

                            profit = 0
                            realizedprofit = 0
                            profitpercent = 0
                            totalprofit = 100
                            #spread = .00015
                            #buffer = (.05 * www)
                            totaltrades = 0
                            goodtrades = 0
                            badtrades = 0
                            rawgood = []
                            rawbad = []
                            #makegood = []
                            #makebad = []
                            #stoploss = []
                            i = self.SEQ_LEN
                            ding = 0
                            leverage = 1
                            goodlist = []
                            badlist = []

                            #####This sidesteps the whole AMA adjusment deal#####
                            
                            #predictadj = predict



                            for l in range(self.SEQ_LEN):
                                goodlist.append(np.nan)
                                badlist.append(np.nan)

                            def GoodAdd(TOpen, TClose):
                                for k in range(TClose - TOpen):
                                    goodlist.append(true[TOpen + k])
                                    badlist.append(np.nan)
                            def BadAdd(TOpen, TClose):
                                for k in range(TClose - TOpen):
                                    badlist.append(true[TOpen + k])
                                    goodlist.append(np.nan)
                            def NaNAdd(T):
                                goodlist.append(np.nan)
                                badlist.append(np.nan)

                            while i < (len(true) - self.FUTURE_PERIOD_PREDICT):

                                try:
                                    if (abs(predictadj[i + (self.FUTURE_PERIOD_PREDICT-1)] - RNGraw[i]) - (spread[i] + (spread[i] * buffer))) > 0:
                                        if predictadj[i + (self.FUTURE_PERIOD_PREDICT-1)] < RNGraw[i]:
                                            Short = True
                                        elif predictadj[i + (self.FUTURE_PERIOD_PREDICT-1)] > RNGraw[i]:
                                            Short = False
                                        totaltrades +=1
                                        iopen = i
                                        exitcheck = 0

                                        while i < (iopen + (self.FUTURE_PERIOD_PREDICT-1)) and exitcheck != 1:
                                            i +=1
                                            if Short == True:
                                                profit = ((RNGraw[iopen] - RNGraw[iopen + (i - iopen)])) - ((spread[i]/2) + (spread[iopen]/2))
                                            elif Short == False:
                                                profit = ((RNGraw[iopen + (i - iopen)] - RNGraw[iopen]) - ((spread[i]/2) + (spread[iopen]/2)))
                                                
                                            realizedprofit = profit
                                        if i == (iopen + (self.FUTURE_PERIOD_PREDICT-1)) and exitcheck != 1:
                                            if realizedprofit <= 0:
                                                profitpercent = realizedprofit / RNGraw[iopen]
                                                profitlev = ((totalprofit * leverage) * (1 + profitpercent)) - (totalprofit * leverage)
                                                rawbad.append(profitlev / totalprofit)
                                                totalprofit += profitlev
                                                badtrades += 1
                                                BadAdd(iopen,i)

                                            else:
                                                profitpercent = realizedprofit / RNGraw[iopen]
                                                profitlev = ((totalprofit * leverage) * (1 + profitpercent)) - (totalprofit * leverage)
                                                rawgood.append(profitlev / totalprofit)
                                                totalprofit += profitlev
                                                goodtrades += 1
                                                GoodAdd(iopen,i)
                                    else:
                                        NaNAdd(i)
                                        i+=1

                                except IndexError:
                                    NaNAdd(i)
                                    i +=1
                                    pass

                            if totalprofit > bestprofit:
                                try:
                                    bestsweep[xxx] = (buffer,AMAp,AMAc,totalprofit)
                                except IndexError:
                                    bestsweep.append((buffer,AMAp,AMAc,totalprofit))
                                bestprofit = totalprofit
                                if totalprofit > BOATprofit:
                                    try:
                                        BOAT[0] = (buffer,AMAp,AMAc,totalprofit)
                                    except IndexError:
                                        BOAT.append((buffer,AMAp,AMAc,totalprofit))
                                    BOATprofit = totalprofit

            bestbufferyL = []
            bestAMAperyL = []
            bestAMAcoeffyL = []
            avgsweepprofityL = []
            

            bestbuffery = 0
            bestAMApery = 0
            bestAMAcoeffy = 0
            avgsweepprofity = 0

            for qqq in range(len(bestsweep)):
                bestbufferyL.append(bestsweep[qqq][0])
                bestAMAperyL.append(bestsweep[qqq][1])
                bestAMAcoeffyL.append(bestsweep[qqq][2])
                avgsweepprofityL.append(bestsweep[qqq][3])



            if recursion == True:
                Mean = True
                Median = True
                Mode = True
                BotB = True

                bestsweeptest = []
                for r in range(len(bestsweep)):
                    bestsweeptest.append([HPSweep(MDict, recursion=False, VT='Test', HPList=bestsweep[r], ret=True), bestsweep[r][0],bestsweep[r][1],bestsweep[r][2]])

                bestsweeptest.sort(key=lambda k: (k[0]))
                print('worst')
                print(bestsweeptest[0])
                print('best')
                print(bestsweeptest[-1])

                #None of this is ever getting used.
                '''
                if Mean != None:
                    print("Mean")
                    bestbufferyMean = statistics.mean(bestbufferyL)
                    bestAMAperyMean = statistics.mean(bestAMAperyL)
                    bestAMAcoeffyMean = statistics.mean(bestAMAcoeffyL)
                    avgsweepprofityMean = statistics.mean(avgsweepprofityL)
                    print('best buffer = ' + str(bestbufferyMean/2))
                    print('best AMA period = ' + str(bestAMAperyMean + 1))
                    print('best AMA coefficient = ' + str((bestAMAcoeffyMean + 1) / 10))
                    print('average profit in sweep = ' + str(avgsweepprofityMean))
                    meanList = [bestbufferyMean, bestAMAperyMean, bestAMAcoeffyMean]
                    meanRes = HPSweep(MDict, recursion=False, VT='Test', HPList=meanList, ret=True)
                    meanListout = [meanRes, (bestbufferyMean/2), (bestAMAperyMean + 1), ((bestAMAcoeffyMean + 1) / 10)]

                if BotB != None:
                    print("BotB")
                    bestbufferyBotB = BOAT[0][0]
                    bestAMAperyBotB = BOAT[0][1]
                    bestAMAcoeffyBotB = BOAT[0][2]
                    avgsweepprofityBotB = BOAT[0][3]
                    print('best buffer = ' + str(bestbufferyBotB/2))
                    print('best AMA period = ' + str(bestAMAperyBotB + 1))
                    print('best AMA coefficient = ' + str((bestAMAcoeffyBotB + 1) / 10))
                    print('average profit in sweep = ' + str(avgsweepprofityBotB))
                    BotBList = [bestbufferyBotB, bestAMAperyBotB, bestAMAcoeffyBotB]
                    BotBRes = HPSweep(MDict, recursion=False, VT='Test', HPList=BotBList, ret=True)
                    BotBListout = [BotBRes, (bestbufferyBotB/2), (bestAMAperyBotB + 1), ((bestAMAcoeffyBotB + 1) / 10)]
                   
                if Median != None:
                    print("Median")
                    bestbufferyMedian = statistics.median(bestbufferyL)
                    bestAMAperyMedian = statistics.median(bestAMAperyL)
                    bestAMAcoeffyMedian = statistics.median(bestAMAcoeffyL)
                    avgsweepprofityMedian = statistics.median(avgsweepprofityL)
                    print('best buffer = ' + str(bestbufferyMedian/2))
                    print('best AMA period = ' + str(bestAMAperyMedian + 1))
                    print('best AMA coefficient = ' + str((bestAMAcoeffyMedian + 1) / 10))
                    print('average profit in sweep = ' + str(avgsweepprofityMedian))
                    medianList = [bestbufferyMedian, bestAMAperyMedian, bestAMAcoeffyMedian]
                    medianRes = HPSweep(MDict, recursion=False, VT='Test', HPList=medianList, ret=True)
                    medianListout = [medianRes, (bestbufferyMedian/2), (bestAMAperyMedian + 1), ((bestAMAcoeffyMedian + 1) / 10)]

                    
                    
                if Mode != None:
                    try:
                        print("Mode")
                        bestbufferyMode = statistics.mode(bestbufferyL)
                        bestAMAperyMode = statistics.mode(bestAMAperyL)
                        bestAMAcoeffyMode = statistics.mode(bestAMAcoeffyL)
                        #avgsweepprofity = statistics.mode(avgsweepprofityL)
                        print('best buffer = ' + str(bestbufferyMode/2))
                        print('best AMA period = ' + str(bestAMAperyMode + 1))
                        print('best AMA coefficient = ' + str((bestAMAcoeffyMode + 1) / 10))
                        print('average profit in sweep = ' + str(avgsweepprofity))
                        modeList = [bestbufferyMode, bestAMAperyMode, bestAMAcoeffyMode]
                        modeRes = HPSweep(MDict, recursion=False, VT='Test', HPList=modeList, ret=True)
                        modeListout = [modeRes, (bestbufferyMode/2), (bestAMAperyMode + 1), ((bestAMAcoeffyMode + 1) / 10)]
                    except Exception:
                        modeListout = medianListout
                '''
                        
                    
                


                Rdepth +=1
                if Rdepth < RDR:
                    HPSweep(MDict, recursion=True, VT='Valid', HPList=BotBList, Rdepth=Rdepth, RDR=RDR)

                with open(self.PATH + 'v20-python-samples/models/EUR_USD/history/HP/' + str(datetime.date.today() - datetime.timedelta(days=1)) + '.csv', 'w') as jj:
                    jj.write(str(bestsweeptest[-1]))

                #literally just stole this from TradeExec rather than trying to decipher HPSweep to assign the attributes... Yikes.
                with open(self.PATH + 'v20-python-samples/models/EUR_USD/history/HP/' + str(datetime.date.today() - datetime.timedelta(days=1)) + '.csv', 'r') as jjj:
                    HPstring = jjj.read()[1:-1]
                    HPlist = []
                    for i in range(1, len(HPstring.split(', '))):
                        HPlist.append(HPstring.split(', ')[i])

                    self.AMA = int(HPlist[1])
                    self.AMAc = float(HPlist[2])
                    
                    

            elif recursion == False:

                print('---')

                rawgoodavg = 0
                rawbadavg = 0
                makegoodavg = 0
                makebadavg = 0
                stoplossavg = 0


                for i in range(len(rawgood)):
                    rawgoodavg += rawgood[i]
                try:
                    rawgoodavg /= len(rawgood)
                except ZeroDivisionError:
                    pass

                for i in range(len(rawbad)):
                    rawbadavg += rawbad[i]
                try:
                    rawbadavg /= len(rawbad)
                except ZeroDivisionError:
                    pass
    
                retval = statistics.mean(avgsweepprofityL)
                print(retval)
                if ret == True:
                    return retval

        HPSweep(HPgetLists())

        def showvis():
            ###Not currently used, apparently?
            #Also mostly redundant, but might be the bones for making a function for all the visualization stuff?
            
            rawgoodavg = 0
            rawbadavg = 0
            makegoodavg = 0
            makebadavg = 0
            stoplossavg = 0


            for i in range(len(rawgood)):
                rawgoodavg += rawgood[i]
            try:
                rawgoodavg /= len(rawgood)
                #break
            except ZeroDivisionError:
                pass

            for i in range(len(rawbad)):
                rawbadavg += rawbad[i]
            try:
                rawbadavg /= len(rawbad)
                #break
            except ZeroDivisionError:
                pass


            #print(rawbad)



            print("rawgood total = " + str(len(rawgood)))
            print("rawgood avg = " + str(rawgoodavg))
            print("rawbad total = " + str(len(rawbad)))
            print("rawbad avg = " + str(rawbadavg))

            print("-------------------------")

            plt.figure(figsize = (18,9))
            plt.plot(test_dataREAL, color = 'black', label = 'Real price')
            plt.plot(predicted_stock_price_multi_headREAL, color = 'blue', label = 'Predicted')
            plt.plot(predictadj, color = 'Purple', label = 'PredictedAdj')
            plt.plot(HIGH_test_data, color = 'orange', label = 'High')
            plt.plot(LOW_test_data, color = 'pink', label = 'Low')
            plt.title('Test Real vs Predict', fontsize=30)
            plt.xlabel('Time')
            plt.ylabel('GE Mid Price')
            plt.legend(fontsize=18)

            plt.plot(goodlist, color = 'green', label = 'Good')
            plt.plot(badlist, color = 'red', label = 'Bad')
            plt.savefig(self.PATH + 'HPProto/' + (datefiles[-1])[:-4] + 'testvis.png')

    def makeModelFull(self):
        #This is basically the whole training pipeline. It mostly uses datestamped files in specific directories to sort itself out.
        #built to be scalable with more instruments, just needs scripting to create the directory trees under the new currency pairs.
        
        #Checks if it has data, requests it if it doesn't exist. Does the same thing for training the transformer model.
        #Checks for a completed model by looking for the history file that gets spit out at the end of training.
        #Checks for completed HP file last.
        
        today = (datetime.datetime.now()).date()
        ##########################################
        TDays = self.M_SCALE * 10
        TDaysplus = TDays * 2
        ##########################################
        try: 
            datefiles = sorted(os.listdir(self.PATH + 'v20-python-samples/data/' + self.INSTRUMENT + '/raw/ask/'), key = str)
            ####BECAUSE RAW FOLDER###########
            datefiles.pop(-1)
            #################################
            lastdate = datefiles[-1]
            testoDT = (datetime.datetime.strptime(lastdate, '%Y-%m-%d.csv')).date()
        except IndexError:
            testoDT = today - datetime.timedelta(days=TDaysplus)

        testoDTfilename = testoDT.strftime('%Y-%m-%d.csv')
        yesterday = (today - datetime.timedelta(days=1))

        while testoDT != yesterday:
            #calc difference in days, grab that many days worth of data
            testodiff = abs((testoDT - yesterday).days)
            if testodiff > TDaysplus:
                testodiff = TDaysplus
            for i in range(testodiff):
                testoDT = yesterday - datetime.timedelta(days=(testodiff-(i+1)))
                testoDTfilename = testoDT.strftime('%Y-%m-%d.csv')
                
                while True:
                    try:
                        self.getData(str(self.PATH + 'v20-python-samples/data/' + self.INSTRUMENT + '/' + testoDTfilename), queryDate=testoDT)
                        break
                    except Exception:
                        print("Error getting mid candles.")
                        pass

                while True:
                    try:
                        self.getData(str(self.PATH + 'v20-python-samples/data/' + self.INSTRUMENT + '/raw/bid/' + testoDTfilename), MBA='bid', queryDate=testoDT)
                        break
                    except Exception:
                        print("Error getting bid candles.")
                        pass
                while True:
                    try:
                        self.getData(str(self.PATH + 'v20-python-samples/data/' + self.INSTRUMENT + '/raw/ask/' + testoDTfilename), MBA='ask', queryDate=testoDT)
                        break
                    except Exception:
                        print("Error getting ask candles.")
                        pass

                        
        print("All done getting data!")
                
        try: 
            modelfiles = sorted(os.listdir(self.PATH + 'v20-python-samples/models/' + self.INSTRUMENT + '/'), key = str)
            modelfiles.pop(-1)
            lastmodel = modelfiles[-1]
            modelDT = (datetime.datetime.strptime(lastmodel, '%Y-%m-%d.h5')).date()
        except IndexError:
            modelDT = testoDT - datetime.timedelta(days=1)

        if modelDT != testoDT:
            print("Training today's model!")
            self.getModel(train=True)
            self.HPTrain(deploy=True)
        else:
            print("Current model already exists; checking history.")
            histfiles = sorted(os.listdir(self.PATH + 'v20-python-samples/models/' + self.INSTRUMENT + '/history/'), key = str)
            histfiles.pop(-1)
            histfiles.pop(-1)
            histfiles.pop(-1)
            try:
                lasthist = histfiles[-1]
                histDT = (datetime.datetime.strptime(lasthist, '%Y-%m-%d.csv')).date()
            except IndexError:
                histDT = None
            if histDT != testoDT:
                print('history file not up to date; retraining incomplete model.')
                self.getModel(train=True)
                self.HPTrain(deploy=True)
            else:
                print('history file is up to date. Model is current and complete. Checking Hyperparameters.')
                HPfiles = sorted(os.listdir(self.PATH + 'v20-python-samples/models/' + self.INSTRUMENT + '/history/HP/'), key = str)
                try:
                    lastHP = HPfiles[-1]
                    HPDT = (datetime.datetime.strptime(lastHP, '%Y-%m-%d.csv')).date()
                except IndexError:
                    HPDT = None
                if HPDT != testoDT:
                    print('HP file not up to date; Sweeping HPs.')
                    self.HPTrain(deploy=True)
                else:
                    print('HP file is up to date.')
                

        print("All done training!")

