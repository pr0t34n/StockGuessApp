# StockGuessApp
A Flask-based webapp for improving self-accuracy of good stock picks

# APP IS IN-PROGRESS

Requires... lots of stuff. The app was generated in an environment where Anaconda was installed, so requirements.txt might be a little bloated.

## This app is not for production use, the webserver launches in debug mode and should only be accessed locally!

The app requires a dataset consisting of a dict in the form dict[stock ticker symbol][df] where [df] is a pandas dataframe. This dataset can be generated by the program by specifying an argument at launch.

# Usage
Download webApp folder, then install all needed reqs with:
```pip install -r requirements.txt```

Then, on your first launch, make sure to update the master dictionary file with: ```python app.py -u```. This will take up about 1 GB space, so please ensure you've got the room for it.

Run with ```python app.py -tX``` where X is your default time display window (3, 6, or 12 [months]). No option defaults to a 6 month view

'Save and Reset' button will save the current session data (datetime, current funds, number of wins, return pct, etc.) to a csv file in the local directory

# Known Issues
* Stoplosses must be placed in the following order: Buy, set stoploss value, check Stoploss checkbox
* The graphs do not display dates. This is **by design**. I wanted to not be influenced by known bear/bull market trends
* There's some weirdness with selling after a stoploss has been executed, diagnosing
