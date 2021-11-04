# DeepRed

DeepRed is an autonomous neural network training and execution pipeline for algo-assisted foreign exchange trading. It was written in Python and interfaces with OandA's REST-V20 API (https://github.com/oanda/v20-python).

## DISCLAIMER
This is prototype code built as a proof of concept, and is intended for educational purposes only, and IS NOT INTENDED FOR USE, CONSULTATION, OR CONSIDERATION WITH REAL MONETARY ASSETS AT STAKE. IF DEPLOYED, THIS SOFTWARE CAN AND WILL LOSE MONEY. BY DOWNLOADING THIS CODE, THE USER AGREES THAT THE AUTHORS AND ALL AFFILIATES ARE NOT LEGALLY RESPONSIBLE FOR ANY DAMAGES, MONETARY OR OTHERWISE, SUSTAINED THROUGH THE IMPLEMENTATION OR MODIFICATION OF THIS CODE. For more information, please review the LICENSE file included in this repository.

## Attribution
The base Transformer model and various chunks of preprocessing code were based on/modified from a Kaggle Project by Shujian titled "Transformer with LSTM" (https://www.kaggle.com/shujian/transformer-with-lstm). Several class functions interfacing with teh REST API are modified directly from example functions included with OandA's REST-V20 API Samples repository (https://github.com/oanda/v20-python-samples).

## Basic Explanation
The entirety of the trading bot's functionality is automated through a single execution script, that is scheduled to run every minute as a Cron job.
Note: This repository intentionally does not include configuration scripts or any setup of directory trees in order to prevent live deployment of the software.

1) The script begins by checking a process ID dump directory to make sure another instance of the script isn't still running. If it is, the second script quits. If it isn't, It logs its own PID and moves on.

2) It makes a class object with all the attributes necessary for the rest of the operations. By default, these are set to the values that I was actually using.

3) It checks your connected OandA account for trades that are already open. For the purposes of limiting variables, position exits are currently limited to simply executing 5 minutes after open to determine a baseline of how often the predictions turn out to be profitable. This is not intended as a serious long term strategy, but it enables us to close positions out without/before loading or running models at all. If the trade has not been open for 5 minutes, it deletes its PID and exits the script.

4) If/once there are no open trades, it runs a function to check the training status of the model and all associated data. These checks are done using datestamped filenames in a directory tree. If the relevant historical price data does not exist, or the history files of the model or algo hyperparameters are not in their associated directories, it runs the associated functions to fully train those systems and generate those files.

5) If/when everything exists, then the script loads the algo hyperparameters, grabs enough live price data to make a prediction, and then adjusts its prediction according to its hyperparameters.

6) Once it has its adjusted prediction, it checks the current price and determines if the projected price 5 minutes from now will be profitable based on their difference minus the projected spread between bid/ask prices and a safety buffer.

7) If the absolute result is greater than 0, it checks its account balance to generate position size, opens the trade either long or short based on whether the result of Step 6 is positive or negative, then wipes its PID and exits the script.
