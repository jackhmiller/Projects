# Event-Based Trade Prediction Model
An event-driven strategy is a type of investment strategy that attempts to take advantage of temporary stock mispricing or behavoral patterns, which can occur before or after a corporate event takes place, or at fixed time intervals such as month end.

This model is configurable for different types of financial assets and instruments, and only requires two data inputs. The first is the returns of the financial instrument that is being traded. The second is a list of event dates, both historical and future around which the strategy will be implemented.

In accordance with the efficient market hypothesis, opportunities to exploit arbitrage or behavoral patters are increasingly sparse. As a result, the model is configured to identify trades within the window of 5 days before and 5 days after the event of interest. This can be changed in the configuration file. The model assumes data of daily granularity, but includes more complexe deep learning architectures to account for data per minute.

The model as seen here is configured to trade around US monthly treasury auctions, specifically 5 year treasury note auctions. Sample data is provided. The code entry point is the run.py file in the Treasury_Auction folder.

Hyperopt is used to optimize hyperparamters of the boosting classifier. 
