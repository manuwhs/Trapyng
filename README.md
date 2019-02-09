# Trapyng

This is a personal project that aims to build an automatic trading system in python, which allows for easy development, testing and real-world use of ideas. In the semi-automated trading mode, the platform will inform the trader of possible entry
point via sending emails. Backtesting tools are implemented in order to validate a strategy before using
it. Many different kinds of analysis are also ready to use. The platform also focues on proper visualization
graphs and understanding of the underlaying methods, promoting the generation of new ideas for future
trading strategies. All in all, the tool provides capabilities for:

 - **Downloading and preprocessing market data:** By means of functions that automatically read
information from different sources: internet, .csv and .hst files, MT4... and automatically load
them into the models. There are also many options for preprocessing the RAW data such as filling
the gaps or timezone alignment.

 - **Enhanced Visualization:** Visualization is the most important tool of a Trader, it is its main
source of information to perform a trade. We developed a graphical library, gl, in order to advise
the trader properly and also, be able to create complex graphs easily to test indicators and get new
ideas.

 - **Backtesting:** The whole system is actually more intended for Backtesting of ideas than for an
efficient real-time implementation, it can be used for real-time tradyng, but it is not optimized for
that. Once a good strategy is found and tested in backtest, the system provides functionalities that
make it easy to implement the strategy into real-time mode.

 - **Real-time Trading mode:** We can create a Real-time trading system. It can be used just as a
guide to plot graphs or as a recomender system.

 - **Trading indicators:** The system provides with the implementation of the most common trading
indicators, their explanation is covered in their corresponding document. New more advanced
indicators are also crafted and the whole environment promotes the easy implementaiton of new
ideas.

 - **Machine learning and Signal Processing:** We provide with mechanisms and ways to use ML
and SP in trading. Not as if they were a magical tool that will solve anything, but to use them
in local specific task where they are good at. The preprocessing layer takes care that the data is
suitable for these algorithms.

 - **Email library:** Trusting a new automatic trading system is just crazy. The way to go is first at
least perfrom trading in a semi-automatic way. We created an email library to easily send quality
reports, including images and explanations to the trader, so that this can asses the potential trade
better.
