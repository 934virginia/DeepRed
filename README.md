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

## Notes on the Model

-You might notice that the model's structure is built in script both when it's trained and when it's deployed, rather than being loaded from file. This is due to issues pickling Lambda layers in Keras. The time scale upon which the model operates doesn't really require intense optimization of the script, so I opted to simply save all of the weights for my trained model and load them as needed after building rather than start from scratch redesigning those Keras layers.

-If you want to test on a greater time scale, you'll need to play with how you train the scaler in data preprocessing, especially if the training data shows a strong protracted uptrend or downtrend. Even though the activation function is linear, the model doesn't do great making predictions at the extremes of its training data range. 

-A lot of the static hyperparameters probably seem arbitrary. Things like batch size, total epochs, and preceeding sequence length were optimized for the default prediction length and currency peir through the use of a grid search. There are definitely smarter ways to do this, especially when all the training took place on a GTX 940M.

-In the end, the validation MSE tended to be in the lower range of Xe-5 after a 50 epoch training schedule, and uncommonly ended up cracking Xe-6. Considering that a lot of similar models at the time were more or less confined to the Xe-4 range for N+1 predictions (on sigmoidal activation, no less) and mine was N+5 (linear), I'm not mad about that. However, the reduced range of my training data resulting from the smaller time frame probably gave me this model a serious advantage in that regard.

-There's probably a lot more to say about the ML side of this, but this isn't a dev journal. Feel free to reach out with any questions.
