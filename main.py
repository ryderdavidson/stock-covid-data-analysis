"""
Project: COVID Stock Data Analysis
Author: Ryder Davidson, Kevin Pham

This project seeks to articulate (in a general sense) the relationship between COVID-19 and the
S&P 500. COVID-19 is examined with respect to new cases per 1,000,000 (NCPM) viz. with respect to
global metrics (as well the United States of America and China). The results provided herein examine
which companies saw an increase in stock value during the presence of an increase in NCPM.
"""

import pandas as pd
import matplotlib.pyplot as plt

### TICKER API
#         ticker api calls (lines 15-22) commented out.
#         used on first run to generate 'ticker_data.csv'

# import yfinance as yf
# from stocksymbol import StockSymbol
# api_key = 'e8c87ea7-8992-463f-bf99-280b24e25a9f'
# ss = StockSymbol(api_key)
# ticker_list = ss.get_symbol_list(index='SPX', symbols_only=True)
# tdf = yf.download(ticker_list, start="2020-02-28", end="2022-03-02", interval="1d")['Adj Close']
# tdf.head()
# tdf.to_csv('ticker_data.csv')

def min_max_scaling(series):
    """Min-Max data normalization helper function for graph output"""
    return (series - series.min()) / (series.max() - series.min())

# --- CONSTANTS
QUANTILES = [0.5, 0.90, 0.995]
WRT_NATION = ['World', 'United States', 'China']
LRG_POP_NATIONS = ['United States', 'World', 'United Kingdom', 'Syria', 'Italy', 'China', 'India', 'Saudi Arabia',
                   'Brazil', 'Iraq']

# --- TICKER DATAFRAMES
tdf = pd.read_csv('ticker_data.csv')

print(tdf)
tdf = tdf.loc[:, tdf.isna().sum() <= .1 * len(tdf)]     # remove tickers that only have values for 10% of dates in range
tdf_shift = (tdf.loc[:, tdf.columns != 'Date'] - tdf.loc[:, tdf.columns != 'Date'].shift(1)).fillna(0)
tdf_shift.drop(tdf_shift[(tdf_shift == 0).all(axis=1)].index, inplace=True)
tdf_shift.insert(loc=0, column='Date', value=tdf.loc[1:, 'Date'])
tdf_shift['Date'] = pd.to_datetime(tdf_shift['Date'], format='%Y/%m/%d')
tdf['Date'] = pd.to_datetime(tdf['Date'], format='%Y/%m/%d')
tdf.drop(tdf.tail(1).index, inplace=True)

# --- COVID19 DATAFRAMES
covid_df = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv',
                       usecols=['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases_per_million',
                                'total_cases_per_million'])
covid_df['date'] = pd.to_datetime(covid_df['date'], format='%Y/%m/%d')
covid_df = covid_df[
    (covid_df['date'] >= pd.Timestamp(2020, 3, 2, 0)) & (covid_df['date'] <= pd.Timestamp(2022, 3, 1, 0))]
covid_df.rename(columns={'date': 'Date'}, inplace=True)

# --- CLEAN & MERGE DFs
tickers = list(tdf_shift.columns)[1:]                                   # list of tickers used
nations = covid_df.drop_duplicates(subset=['location'])['location']     # list of nations used

for nation in nations:  # left merge data - uses date index relative to the ticker dataframe
    temp1 = pd.DataFrame(covid_df[['new_cases_per_million', 'Date']][covid_df['location'] == nation])
    temp2 = pd.DataFrame(covid_df[['total_cases_per_million', 'Date']][covid_df['location'] == nation])
    tdf_shift = pd.merge(tdf_shift, temp1, on='Date', how='left')
    tdf_shift.rename(columns={'new_cases_per_million': nation}, inplace=True)
    tdf = pd.merge(tdf, temp2, on='Date', how='left')
    tdf.rename(columns={'total_cases_per_million': nation}, inplace=True)
tdf_shift.fillna(0, inplace=True)
tdf.fillna(0, inplace=True)

master_results = {}

# --- CONSOLE PRINT VALUES

# display head for both dataframes utilized viz. tdf and tdf_shift
print("Ticker Dataframe Absolute Price LEFT MERGE Total Covid Case Per Million:\n", tdf.head(10), "\n\n")
print("Ticker Dataframe Net Change Value LEFT MERGE New Covid Case Per Million:\n", tdf_shift.head(10), "\n\n")
print('tickers:\n', list(tickers))
print('\nnations:\n', list(nations))

# methodology: using constant QUANTILES, iterate through values in nested for loop
#              count the number of times each ticker >= QUANTILE when covid_case >= QUANTILE
#              (higher count indicates closer positive correlation between ticker and case)

# top 10 results (sorted DESC) are stored in master_results dictionary
# master_results key schema = str(WRT_NATIONS[index] + quantile[i] + 'x' + quantile[j])
# e.g. to query tickers that increase >= mean when world covid cases increase >= mean:
#      the master_results key would be: 'World0.5x0.5' (0.5 percentile => mean)

print("\n*********************************"
      "\nMETRICS GIVEN GLOBAL COVID CASES"
      "\n*********************************\n")
for i in QUANTILES:
    for j in QUANTILES:
        output = {}
        for ticker in tickers:
            count = len(tdf_shift.loc[
                            (tdf_shift[WRT_NATION[0]] >= tdf_shift[WRT_NATION[0]].quantile(q=i)) & (
                                        tdf_shift[ticker] >= tdf_shift[ticker].quantile(q=j)), [
                                ticker, WRT_NATION[0]]])
            if count > 0:
                output[ticker] = count
        sorted_output = sorted(output.items(), key=lambda kv: kv[1], reverse=True)
        master_results[WRT_NATION[0] + str(i) + 'x' + str(j)] = sorted_output[:10]
        print("TOP 10 of", len(sorted_output), "-> covid cases >= ", i * 100, "quantile; ticker increase >= ", j * 100,
              "quantile")
        print(sorted_output[:10], "\n")

print("\n*********************************"
      "\nMETRICS GIVEN U.S. COVID CASES"
      "\n*********************************\n")
for i in QUANTILES:
    for j in QUANTILES:
        output = {}
        for ticker in tickers:
            count = len(tdf_shift.loc[
                            (tdf_shift[WRT_NATION[1]] >= tdf_shift[WRT_NATION[1]].quantile(q=i)) & (
                                        tdf_shift[ticker] >= tdf_shift[ticker].quantile(q=j)), [
                                ticker, WRT_NATION[1]]])
            if count > 0:
                output[ticker] = count
        sorted_output = sorted(output.items(), key=lambda kv: kv[1], reverse=True)
        master_results[WRT_NATION[1] + str(i) + 'x' + str(j)] = sorted_output[:10]
        print("TOP 10 of", len(sorted_output), "-> covid cases >= ", i * 100, "quantile; ticker increase >= ", j * 100,
              "quantile")
        print(sorted_output[:10], "\n")

print("\n*********************************"
      "\nMETRICS GIVEN CHINA COVID CASES"
      "\n*********************************\n")
for i in QUANTILES:
    for j in QUANTILES:
        output = {}
        for ticker in tickers:
            count = len(tdf_shift.loc[
                            (tdf_shift[WRT_NATION[2]] >= tdf_shift[WRT_NATION[2]].quantile(q=i)) & (
                                        tdf_shift[ticker] >= tdf_shift[ticker].quantile(q=j)), [
                                ticker, WRT_NATION[2]]])
            if count > 0:
                output[ticker] = count
        sorted_output = sorted(output.items(), key=lambda kv: kv[1], reverse=True)
        master_results[WRT_NATION[2] + str(i) + 'x' + str(j)] = sorted_output[:10]
        print("TOP 10 of", len(sorted_output), "-> covid cases >= ", i * 100, "quantile; ticker increase >= ", j * 100,
              "quantile")
        print(sorted_output[:10], "\n")

sorted_covid_cases_lrgpop = sorted(tdf_shift[LRG_POP_NATIONS].mean().items(), key=lambda kv: kv[1], reverse=True)[:10]
sorted_covid_cases = sorted(tdf_shift[nations].mean().items(), key=lambda kv: kv[1], reverse=True)[:10]
sorted_tickers = sorted(tdf_shift[tickers].mean().items(), key=lambda kv: kv[1], reverse=True)[:10]

# --- GRAPH OUTPUTS

# methodology: for each nation in WRT_NATION, create line graph for (only) *top three* tickers (sorted)
#              when q1 = 0.5 & q2 = 0.5 and when q1 = 0.995 & q2 = 0.995, using master_results dict
#              i.e. keys: 'World0.5x0.5', 'WORLD0.995x0.995', 'United States0.5x0.5', 'United States0.995x0.995' ...

# Graphs: WRT World Covid Cases

plt.figure(figsize=(20, 5))
plt.plot(tdf['Date'], min_max_scaling(tdf[WRT_NATION[0]]), label='Covid')
three_ticks = [x[0] for x in master_results['World0.5x0.5'][:3]]
for tick in three_ticks:
    plt.plot(tdf['Date'], min_max_scaling(tdf[tick]), label=str(tick))
plt.suptitle("Stock Value Increase vs. Covid Case Increase (World)", y=1, fontsize=15)
plt.title("TICKER >= 50th percentile, NCPM >= 50th percentile", fontsize=12)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Normalized Values")
plt.show()

plt.figure(figsize=(20, 5))
plt.plot(tdf['Date'], min_max_scaling(tdf[WRT_NATION[0]]), label='Covid')
three_ticks = [x[0] for x in master_results['World0.995x0.995'][:3]]
for tick in three_ticks:
    plt.plot(tdf['Date'], min_max_scaling(tdf[tick]), label=str(tick))
plt.suptitle("Stock Value Increase vs. Covid Case Increase (World)", y=1, fontsize=15)
plt.title("TICKER >= 99.5th percentile, NCPM >= 99.5th percentile", fontsize=12)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Normalized Values")
plt.show()

# Graphs: WRT U.S. Covid Cases

plt.figure(figsize=(20, 5))
plt.plot(tdf['Date'], min_max_scaling(tdf[WRT_NATION[0]]), label='Covid')
three_ticks = [x[0] for x in master_results['United States0.5x0.5'][:3]]
for tick in three_ticks:
    plt.plot(tdf['Date'], min_max_scaling(tdf[tick]), label=str(tick))
plt.suptitle("Stock Value Increase vs. Covid Case Increase (U.S.)", y=1, fontsize=15)
plt.title("TICKER >= 50th percentile, NCPM >= 50th percentile", fontsize=12)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Normalized Values")
plt.show()

plt.figure(figsize=(20, 5))
plt.plot(tdf['Date'], min_max_scaling(tdf[WRT_NATION[0]]), label='Covid')
three_ticks = [x[0] for x in master_results['United States0.995x0.995'][:3]]
for tick in three_ticks:
    plt.plot(tdf['Date'], min_max_scaling(tdf[tick]), label=str(tick))
plt.suptitle("Stock Value Increase vs. Covid Case Increase (U.S.)", y=1, fontsize=15)
plt.title("TICKER >= 99.5th percentile, NCPM >= 99.5th percentile", fontsize=12)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Normalized Values")
plt.show()

# Graphs: WRT China Covid Cases

plt.figure(figsize=(20, 5))
plt.plot(tdf['Date'], min_max_scaling(tdf[WRT_NATION[0]]), label='Covid')
three_ticks = [x[0] for x in master_results['China0.5x0.5'][:3]]
for tick in three_ticks:
    plt.plot(tdf['Date'], min_max_scaling(tdf[tick]), label=str(tick))
plt.suptitle("Stock Value Increase vs. Covid Case Increase (China)", y=1, fontsize=15)
plt.title("TICKER >= 50th percentile, NCPM >= 50th percentile", fontsize=12)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Normalized Values")
plt.show()

plt.figure(figsize=(20, 5))
plt.plot(tdf['Date'], min_max_scaling(tdf[WRT_NATION[0]]), label='Covid')
three_ticks = [x[0] for x in master_results['China0.995x0.995'][:3]]
for tick in three_ticks:
    plt.plot(tdf['Date'], min_max_scaling(tdf[tick]), label=str(tick))
plt.suptitle("Stock Value Increase vs. Covid Case Increase (China)", y=1, fontsize=15)
plt.title("TICKER >= 99.5th percentile, NCPM >= 99.5th percentile", fontsize=12)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Normalized Values")
plt.show()

# Graphs: BAR GRAPHS

plt.figure(figsize=(20, 5))
ten_ticks = [x[0] for x in master_results['World0.5x0.5']]
values = [x[1] for x in master_results['World0.5x0.5']]
plt.bar(ten_ticks, values)
plt.suptitle("Number of Stock-Covid Correlations (World)", y=1, fontsize=15)
plt.title("TICKER >= 50th percentile, NCPM >= 50th percentile", fontsize=12)
plt.xlabel("Ticker")
plt.ylabel("Number of Correlations")
plt.show()

print("\n\nSelected Nations NCPM:\n", sorted_covid_cases_lrgpop)
print("\nTop Average NCPM Nations:\n", sorted_covid_cases)
print("\nTop Average Net Change Tickers:\n", sorted_tickers)

fig, ax = plt.subplots(figsize=(20, 5))
mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
ten_ticks = [x[0] for x in sorted_covid_cases_lrgpop]
values = [x[1] for x in sorted_covid_cases_lrgpop]
plt.bar(ten_ticks, values)
plt.suptitle("Cross-Section of Nations", x=mid, fontsize=15)
plt.title("Average Covid Case Density (NCPM)", fontsize=12)
plt.xlabel("Nation")
plt.ylabel("Average NCPM")
plt.show()

plt.figure(figsize=(20, 5))
ten_ticks = [x[0] for x in sorted_covid_cases]
values = [x[1] for x in sorted_covid_cases]
plt.bar(ten_ticks, values)
plt.suptitle("Top 10 Nations", x=mid, fontsize=15)
plt.title("Highest Average Case Density (NCPM)", fontsize=12)
plt.xlabel("Nation")
plt.ylabel("Average NCPM")
plt.show()

plt.figure(figsize=(20, 5))
ten_ticks = [x[0] for x in sorted_tickers]
values = [x[1] for x in sorted_tickers]
plt.bar(ten_ticks, values)
plt.suptitle("Top 10 Companies", x=mid, fontsize=15)
plt.title("Highest Average Net Change ($)", fontsize=12)
plt.xlabel("Ticker")
plt.ylabel("Average Net Change")
plt.show()

# Graphs: TREND LINES

plt.figure(figsize=(20, 5))
five_ticks = [x[0] for x in sorted_tickers[:5]]
for tick in five_ticks:
    print(tick)
    print(tdf_shift[tick])
    plt.plot(tdf['Date'], tdf[tick], label=str(tick))
plt.suptitle("Top 5 Companies", y=1, x=mid, fontsize=15)
plt.title("Daily Stock Price Per Share", fontsize=12)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price Per Share")
plt.show()
