# StockGuessApp
A Flask-based webapp for improving self-accuracy of good stock picks

# APP IS IN-PROGRESS

Requires... lots of stuff. The app was generated in an environment where Anaconda was installed, so requirements.txt might be a little bloated.

## This app is not for production use, the webserver launches in debug mode and should only be accessed locally!

The app requires a dataset consisting of a dict in the form dict[stock ticker symbol][df] where [df] is a pandas dataframe. This dataset can be generated by the program by specifying an argument.
