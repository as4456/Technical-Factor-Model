# -*- coding: utf-8 -*-
"""A single unified technical factor based model """

import numpy as np
import pandas as pd
import datetime as dt
from pandas.tseries.offsets import BDay
import random
import statsmodels.api as sm
import operator
from yahoo_finance import Share
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import pickle


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
filename='factor_model'
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('log_'+filename+'.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)



"""
CONSTANTS
"""
NUMBER_OF_STOCKS = 350 # change to 500
TOP_STOCKS = 10 # change to 10


"""
Headers
"""
CLOSE_STR = 'Close'
VOLUME_STR = 'Volume'
OPEN_STR = 'Open'
DATE_STR = 'Date'
LOW_STR = 'Low'
HIGH_STR = 'High'
ADJ_CLOSE_STR = 'Adj_Close'
SYMBOL_STR = 'Symbol'
STOCKS_FILE = 'sp500_names.csv'


"""
Helper functions
"""
insample_start_year=2010
insample_end_year=2014
outsample_start_year=2014
outsample_end_year=2015

def dates_insample_monthly():
    """In-sample dates"""
    dates = []
    for y in range(insample_start_year,insample_end_year):
        for m in range(1,13):
            date = str(dt.date(y,m,1))
            dates.append(date)
    return dates

def dates_outsample_monthly():
    """Out-sample dates"""
    dates = []
    for y in range(outsample_start_year,outsample_end_year):
        for m in range(1,13):
            date = str(dt.date(y,m,1))
            dates.append(date)
    return dates


def bd_range(start_date, end_date):
    """Business day range"""
    return pd.bdate_range(start_date, end_date)


def cut_dataframe(df,start_date=None,end_date=None):
    """Cut dataframe"""
    all_dates = df.index.values
    if not end_date:
        end_date = all_dates.max()
    if not start_date:
        start_date = all_dates.min()
    cut_df = df.ix[df.index.searchsorted(start_date):(1+df.index.searchsorted(end_date))]
    return cut_df


"""
Read financial data
"""


def read_stock_names():
    """Read stock names from input file (S&P500 stocks)"""
    names_df = pd.read_csv(STOCKS_FILE)
    return list(names_df.ix[:,1].values)


def get_stock_df(ticker,start_date,end_date):
    """Get stock dataframe"""
    share = Share(ticker)
    share_hist = share.get_historical(start_date,end_date)
    len_share_hist = len(share_hist)
    dates = ['']*len_share_hist
    open = [0.]*len_share_hist
    close = [0.]*len_share_hist
    high = [0.]*len_share_hist
    low = [0.]*len_share_hist
    volume = [0.]*len_share_hist
    adj_close = [0.]*len_share_hist
    for i in range(len_share_hist):
        dates[i] = share_hist[i][DATE_STR]
        open[i] = float(share_hist[i][OPEN_STR])
        close[i] = float(share_hist[i][CLOSE_STR])
        adj_close[i] = float(share_hist[i][ADJ_CLOSE_STR])
        high[i] = float(share_hist[i][HIGH_STR])
        low[i] = float(share_hist[i][LOW_STR])
        volume[i] = float(share_hist[i][VOLUME_STR])
    df = pd.DataFrame(open, index = pd.to_datetime(dates), columns=[OPEN_STR])
    df[CLOSE_STR] = close
    df[ADJ_CLOSE_STR] = adj_close
    df[HIGH_STR] = high
    df[LOW_STR] = low
    df[VOLUME_STR] = volume
    df.index.name = DATE_STR
    return df.sort_index()
    
    

"""
Financial analysis functions
"""

def sharpe_ratio_snp500(returns,date,column_name):
    """Sharpe ratio for a month ended with date"""
    day = pd.to_datetime(date)
    day_minus_month = day - dt.timedelta(days=30)
    day = str(day)[:10] # conversion to string
    day_minus_month = str(day_minus_month)[:10] # conversion to string
    snp500_df = get_stock_df('^GSPC',day_minus_month,day)
    price_snp500 = snp500_df[column_name].values
    returns_snp500 = price_snp500[1:] / price_snp500[:-1]
    len_returns = len(returns)
    len_snp500 = len(returns_snp500)
    len_min = min(len_returns,len_snp500)
    returns = returns[:len_min]
    returns_snp500 = returns_snp500[:len_min]
    return (returns - returns_snp500).mean() / (returns - returns_snp500).std()

def sharpe_ratio(returns1,returns2):
    """Sharpe ratio for two datasets"""
    len1 = len(returns1)
    len2 = len(returns2)
    len_min = min(len1,len2)
    returns1 = returns1[:len_min]
    returns2 = returns2[:len_min]
    return (returns1 - returns2).mean() / (returns1 - returns2).std()


def portfolio_return_monthly(ticker_list,weight_dict,date,column_name):
    """Monthly portfolio return"""
    # weight_dict = {'AAPL': 0.05, ...}
    day = pd.to_datetime(date)
    day_minus_month = day - dt.timedelta(days=30)
    day = str(day)[:10] # conversion to string
    day_minus_month = str(day_minus_month)[:10] # conversion to string
    portfolio_returns_list = []
    for ticker in ticker_list:
        print (ticker)
        df = get_stock_df(ticker,day_minus_month,day)
        prices = df[column_name].values
        returns = (prices[1:] / prices[:-1] - 1)
        portfolio_returns_list.append(returns * weight_dict[ticker])
    len_pr_array = np.zeros([len(ticker_list)])
    for i in range(len(portfolio_returns_list)): # cleaning and correcting length
        len_pr_array[i] = len(portfolio_returns_list[i])
    min_len_pr = int(len_pr_array[len_pr_array>0].min())
    for i in range(len(portfolio_returns_list)):
        if portfolio_returns_list[i].size > 0:
            portfolio_returns_list[i] = portfolio_returns_list[i][:min_len_pr]
        else:
            portfolio_returns_list[i] = np.zeros([min_len_pr])
    return sum(portfolio_returns_list)



"""
Mock Data
"""


def generate_mock_df(start_date,end_date):
    """Generate mock data"""
    dates = bd_range(start_date, end_date)
    prices = np.array([random.uniform(50,150) for n in range(len(dates))])
    volume = np.array([random.randint(100,1000) for n in range(len(dates))])
    close = prices * random.uniform(0.9,1.1)
    open = prices * random.uniform(0.9,1.1)
    low = prices * 0.8
    high = prices * 1.2
    adj_close = close*1.05
    df = pd.DataFrame(open, index = dates, columns=[OPEN_STR])
    df[CLOSE_STR] = close
    df[ADJ_CLOSE_STR] = adj_close
    df[HIGH_STR] = high
    df[LOW_STR] = low
    df[VOLUME_STR] = volume
    df.index.name = DATE_STR
    return df


"""
Factors
"""

def slopeWeekly(df,date,column_name):
    """Price slope of 10 Week Exponential Moving Average (EMA) over 5 weeks"""
    day = df.index.asof(pd.to_datetime(date))
    cut_df = cut_dataframe(df,end_date=day)
    ema = pd.ewma(cut_df[column_name], span=50).ix[-25:]
    ema_dates = ema.index.values
    ema_time_delta_days = pd.Timedelta(ema_dates.max()-ema_dates.min()).total_seconds()/3600/24
    slope = (ema.values[-1] - ema.values[0]) / ema_time_delta_days
    return slope

def volumentumWeekly(df,date,column_name):
    """[Price (End of this week) - Price (End of last week)]*[Avg Week Volume / Avg 6 Mo Volume]"""
    day = df.index.asof(pd.to_datetime(date))
    friday1 = df.index.asof(day - dt.timedelta(days=(day.weekday() - 4) % 7, weeks=0))
    friday2 = df.index.asof(day - dt.timedelta(days=(day.weekday() - 4) % 7, weeks=1))
    one_week = [day - dt.timedelta(i) for i in range(7)]
    six_months = [day - dt.timedelta(i) for i in range(180)]
    avg_week_volume = df.loc[one_week].dropna()[column_name].mean()
    avg_six_months_volume = df.loc[six_months].dropna()[column_name].mean()
    return (df[column_name].loc[friday1] - df[column_name].loc[friday2]) * avg_week_volume / avg_six_months_volume

def volumentumMonthly(df,date,column_name):
    """[Price (End of this month) - Price (End of last month)]*[Avg Monthly Volume / Avg 12 Mo Volume]"""
    day = df.index.asof(pd.to_datetime(date))
    end_of_month1 = pd.to_datetime(dt.date(day.year,day.month,1)) - dt.timedelta(days=1)
    end_of_month1 = df.index.asof(pd.to_datetime(end_of_month1))
    end_of_month2 = pd.to_datetime(dt.date(end_of_month1.year,end_of_month1.month,1)) - dt.timedelta(days=1)
    end_of_month2 = df.index.asof(pd.to_datetime(end_of_month2))
    one_month = [df.index.asof(day - dt.timedelta(i)) for i in range(30)]
    twelve_months = [df.index.asof(day - dt.timedelta(i)) for i in range(360)]
    avg_monthly_volume = df.loc[one_month].dropna()[column_name].mean()
    avg_twelve_months_volume = df.loc[twelve_months].dropna()[column_name].mean()
    return (df[column_name].loc[end_of_month1] - df[column_name].loc[end_of_month2]) \
           * avg_monthly_volume / avg_twelve_months_volume

def momentumNMo(df,date,column_name,number_of_months):
    """Avg of daily returns of last N months"""
    end_date = df.index.asof(pd.to_datetime(date))
    start_date = end_date - dt.timedelta(days=30*number_of_months)
    start_date = df.index.asof(start_date)
    cut_df = cut_dataframe(df,start_date=start_date,end_date=date)[column_name]
    cut_df_minus1 = cut_dataframe(df,start_date=start_date-BDay(1),end_date=end_date-BDay(1))[column_name]
    len_df = len(cut_df)
    len_df_minus1 = len(cut_df_minus1)
    len_min = min(len_df,len_df_minus1)
    daily_returns = cut_df.values[:len_min] / cut_df_minus1.values[:len_min]
    return daily_returns.mean()

def meanReversion(df,date,column_name,n_days,N_days):
    """(Price Avg for n Days - Price Avg for N Days)/Price Avg for N Days"""
    day = df.index.asof(pd.to_datetime(date))
    n_day_range = [df.index.asof(day - dt.timedelta(i)) for i in range(n_days)]
    N_day_range = [df.index.asof(day - dt.timedelta(i)) for i in range(N_days)]
    avg_n_days = df.loc[n_day_range].dropna()[column_name].mean()
    avg_N_days = df.loc[N_day_range].dropna()[column_name].mean()
    return avg_n_days / avg_N_days - 1.0

def highLowRange(df,date,column_name):
    """(Current price ‐ 52 week price low)/(52 Week High – 52 Week Low)"""
    day = df.index.asof(pd.to_datetime(date))
    fifty_two_day_range = [df.index.asof(day - dt.timedelta(i)) for i in range(52*5)]
    fifty_two_week_low = df.loc[fifty_two_day_range].dropna()[LOW_STR].min()
    fifty_two_week_high = df.loc[fifty_two_day_range].dropna()[HIGH_STR].max()
    current_price = df.loc[day][column_name]
    return (current_price - fifty_two_week_low) / (fifty_two_week_high - fifty_two_week_low)

def moneyFlow(df,date):
    """Money Flow = (((Close‐Low) ‐ (High‐Close)) / (High‐Low)) * Volume"""
    day = df.index.asof(pd.to_datetime(date))
    close = df.loc[day][CLOSE_STR]
    low = df.loc[day][LOW_STR]
    high = df.loc[day][HIGH_STR]
    volume = df.loc[day][VOLUME_STR]
    return (((close - low) - (high - close)) / (high - low)) * volume

def moneyFlowPersistency(df,date,number_of_months):
    """No of days when Money Flow was positive in N months / Number of Days in N months"""
    day = df.index.asof(pd.to_datetime(date))
    day_range = [df.index.asof(day - dt.timedelta(i)) for i in range(number_of_months*30)]
    money_flows = np.array([moneyFlow(df,day1 - BDay(1)) for day1 in day_range])
    signs_of_money_flows = np.sign(money_flows)
    return (signs_of_money_flows[signs_of_money_flows>0]).sum() / (number_of_months*30)

def slopeDaily(df,date,column_name):
    """Price slope of 10 Day Exponential Moving Average (EMA) over 5 days"""
    day = df.index.asof(pd.to_datetime(date))
    cut_df = cut_dataframe(df,end_date=day)
    ema = pd.ewma(cut_df[column_name], span=10).ix[-5:]
    ema_dates = ema.index.values
    ema_time_delta_days = pd.Timedelta(ema_dates.max()-ema_dates.min()).total_seconds()/3600/24
    slope = (ema.values[-1] - ema.values[0]) / ema_time_delta_days
    return slope

def slopeMonthly(df,date,column_name):
    """Price slope of 10 Month Exponential Moving Average (EMA) over 5 months"""
    day = df.index.asof(pd.to_datetime(date))
    cut_df = cut_dataframe(df,end_date=day)
    ema = pd.ewma(cut_df[column_name], span=300).ix[-150:]
    ema_dates = ema.index.values
    ema_time_delta_days = pd.Timedelta(ema_dates.max()-ema_dates.min()).total_seconds()/3600/24
    slope = (ema.values[-1] - ema.values[0]) / ema_time_delta_days
    return slope

def pxRet(df,date,column_name, number_of_days):
    """Price return in percentage in N days"""
    day = df.index.asof(pd.to_datetime(date))
    day_minus_n_days = df.index.asof(day - dt.timedelta(days=number_of_days))
    cut_df = cut_dataframe(df,end_date=day,start_date=day_minus_n_days)[column_name]
    return cut_df.values[-1] / cut_df.values[0]

def currPxRet(df,date,column_name):
    """(Current Price – Moving Avg of Last 3 yrs Price) / Current Price"""
    day = df.index.asof(pd.to_datetime(date))
    price = df.loc[day][column_name]
    day_start = df.index.asof(day - dt.timedelta(days=3*360))
    price_mean = cut_dataframe(df,start_date=day_start, end_date=day)[column_name].mean()
    return 1.0 - price_mean / price

def nDayADR(df,date,column_name,number_of_days):
    """Avg of daily returns of last N days"""
    day = df.index.asof(pd.to_datetime(date))
    day_minus_n_days = df.index.asof(day - dt.timedelta(days=number_of_days))
    cut_df = cut_dataframe(df,end_date=day,start_date=day_minus_n_days)[column_name]
    cut_df_minus1 = cut_dataframe(df,end_date=day,start_date=day_minus_n_days)[column_name]
    return (cut_df.values / cut_df_minus1.values).mean()

def nDayADP(df,date,column_name,number_of_days):
    """Avg of daily price change of last N days"""
    day = df.index.asof(pd.to_datetime(date))
    day_minus_n_days = df.index.asof(day - dt.timedelta(days=number_of_days))
    cut_df = cut_dataframe(df,end_date=day,start_date=day_minus_n_days)[column_name]
    cut_df_minus1 = cut_dataframe(df,end_date=day,start_date=day_minus_n_days)[column_name]
    return (cut_df.values - cut_df_minus1.values).mean()

def pxRet2(df,date,column_name,N_days,n_days):
    """-50% of N days price return + 50% of n days price return"""
    return -0.5 * pxRet(df,date,column_name,N_days) + 0.5 * pxRet(df,date,column_name,n_days)

def currPxRetSlope(df,date,column_name):
    """-50% of 3YrCurrPxRet + 50% of SlopeWeekly"""
    return -0.5 * currPxRet(df,date,column_name) + 0.5 * slopeWeekly(df,date,column_name)


def allFactors(df_list,date,column_name):
    """All Factors as an array"""
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    f5 = []
    f6 = []
    f7 = []
    f8 = []
    f9 = []
    f10 = []
    f11 = []
    f12 = []
    f13 = []
    f14 = []
    f15 = []
    f16 = []
    f17 = []
    f18 = []
    f19 = []
    f20 = []
    f21 = []
    f22 = []
    f23 = []
    f24 = []
    f25 = []
    f26 =[]
    f27 = []
    f28 = []
    for df in df_list:
        if not df.empty:
            try:
                f1.append(slopeWeekly(df,date,column_name))
            except:
                f1.append(0.)
            try:
                f2.append(volumentumWeekly(df,date,column_name))
            except:
                f2.append(0.)
            try:
                f3.append(volumentumMonthly(df,date,column_name))
            except:
                f3.append(0.)
            try:
                f4.append(momentumNMo(df,date,column_name,3))
            except:
                f4.append(0.)
            try:
                f5.append(momentumNMo(df,date,column_name,6))
            except:
                f5.append(0.)
            try:
                f6.append(momentumNMo(df,date,column_name,9))
            except:
                f6.append(0.)
            try:
                f7.append(meanReversion(df,date,column_name,5,250))
            except:
                f7.append(0.)
            try:
                f8.append(meanReversion(df,date,column_name,5,500))
            except:
                f8.append(0.)
            try:
                f9.append(meanReversion(df,date,column_name,5,1000))
            except:
                f9.append(0.)
            try:
                f10.append(highLowRange(df,date,column_name))
            except:
                f10.append(0.)
            try:
                f11.append(moneyFlow(df,date))
            except:
                f11.append(0.)
            try:
                f12.append(moneyFlowPersistency(df,date,1))
            except:
                f12.append(0.)
            try:
                f13.append(moneyFlowPersistency(df,date,3))
            except:
                f13.append(0.)
            try:
                f14.append(moneyFlowPersistency(df,date,6))
            except:
                f14.append(0.)
            try:
                f15.append(slopeDaily(df,date,column_name))
            except:
                f15.append(0.)
            try:
                f16.append(slopeMonthly(df,date,column_name))
            except:
                f16.append(0.)
            try:
                f17.append(pxRet(df,date,column_name,360*3))
            except:
                f17.append(0.)
            try:
                f18.append(pxRet(df,date,column_name,30))
            except:
                f18.append(0.)
            try:
                f19.append(pxRet(df,date,column_name,60) )
            except:
                f19.append(0.)
            try:
                f20.append(pxRet(df,date,column_name,90))
            except:
                f20.append(0.)
            try:
                f21.append(currPxRet(df,date,column_name))
            except:
                f21.append(0.)
            try:
                f22.append(nDayADR(df,date,column_name,90))
            except:
                f22.append(0.)
            try:
                f23.append(nDayADP(df,date,column_name,60))
            except:
                f23.append(0.)
            try:
                f24.append(nDayADP(df,date,column_name,90))
            except:
                f24.append(0.)
            try:
                f25.append(pxRet2(df,date,column_name,360*3,30))
            except:
                f25.append(0.)
            try:
                f26.append(pxRet2(df,date,column_name,360*3,60))
            except:
                f26.append(0.)
            try:
                f27.append(pxRet2(df,date,column_name,360*3,90))
            except:
                f27.append(0.)
            try:
                f28.append(currPxRetSlope(df,date,column_name))
            except:
                f28.append(0.)
        else:
            f1.append(0.)
            f2.append(0.)
            f3.append(0.)
            f4.append(0.)
            f5.append(0.)
            f6.append(0.)
            f7.append(0.)
            f8.append(0.)
            f9.append(0.)
            f10.append(0.)
            f11.append(0.)
            f12.append(0.)
            f13.append(0.)
            f14.append(0.)
            f15.append(0.)
            f16.append(0.)
            f17.append(0.)
            f18.append(0.)
            f19.append(0.)
            f20.append(0.)
            f21.append(0.)
            f22.append(0.)
            f23.append(0.)
            f24.append(0.)
            f25.append(0.)
            f26.append(0.)
            f27.append(0.)
            f28.append(0.)
    return [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28]


def returnsMonthly(dfs, column_name):
    """Monthly returns for all stocks for given dates"""
    d1 = dates_insample_monthly()
    d2 = dates_outsample_monthly()
    d = d1+ d2
    zero_date = pd.to_datetime(d[0]) - dt.timedelta(days=30)
    zero_date = str(zero_date)[:10]
    d.insert(0,zero_date)
    slice = {}
    ret_df_dict = {}
    for key in list(dfs.keys()):
        df = dfs[key]
        dates_asof = []
        for date in d:
            dates_asof.append(df.index.asof(date))
        dates_asof = pd.DatetimeIndex(dates_asof)
        try:
            slice[key] = df.loc[dates_asof]
            ret = slice[key][column_name].values[1:] / slice[key][column_name].values[:-1] - 1
            ret_df = pd.DataFrame(ret,index=slice[key].index.values[1:],columns=['Return'])
        except:
            ret_df = pd.DataFrame(np.zeros([len(d)-1]),index=d[1:],columns=['Return'])
        ret_df_dict[key] = ret_df
    return ret_df_dict


class Factors:
    """Z-scores and Alpha generating factors"""
    def __init__(self,data_yahoo=False):
        self.column_name = CLOSE_STR
        tickers = read_stock_names()
        self.ticker_sample = tickers[:NUMBER_OF_STOCKS]
        date = dates_insample_monthly()[0]
        start_date = pd.to_datetime(date) - dt.timedelta(days=366*3)
        start_date = str(start_date)[:10]
        end_date = dates_outsample_monthly()[-1]
        self.dfs = {}
        if data_yahoo:
            for ticker in self.ticker_sample:
                print (ticker)
                df = get_stock_df(ticker,start_date,end_date)
                self.dfs[ticker] = df
                print ('Data retrieved for', ticker)
        else:
            self.dfs=pickle.load(open("stock_data.p","rb"))
            print "Data loaded!"
        self.returns = returnsMonthly(self.dfs,self.column_name)
        print "Monthly returns calculated"

    def z_scoring(self,date):
        tickers = list(self.dfs.keys())
        all_factors = allFactors(list(self.dfs.values()),date,self.column_name)
        z_factors = []
        zf_dict = {}
        for factor in all_factors:
            z_factor = (factor - np.mean(factor)) / np.std(factor)
            z_factors.append(z_factor)
        z_factors = np.array(z_factors)
        where_are_nans = np.isnan(z_factors)
        z_factors[where_are_nans] = 0.
        for i in range(len(tickers)):
            zf_dict[tickers[i]] = z_factors.T[i]
        return zf_dict

    def data_for_regression(self,year):
        """Data for regression for a given year"""
        rs = []
        zs = []
        ds_dict = {}
        z_dict = {}
        for key in self.ticker_sample:
            try:
                r = self.returns[key].loc[dt.date(year,1,1):dt.date(year,12,31)]
                rs.append(r['Return'].values)
                ds = r.index.values
            except:
                rs.append(np.zeros([12]))
                ds = np.array([dt.date(year,i,1) for i in range(1,13)])
            ds_dict[key] = ds
        ds_dates = ds_dict.values()
        ds_dates = [val for sublist in ds_dates for val in sublist]
        ds_dates = list(set(ds_dates))
        for d in ds_dates:
            z_dict[d] = self.z_scoring(d)
        for key in self.ticker_sample:
            for d in ds_dict[key]:
                zs.append(z_dict[d][key])
        rs = np.concatenate(rs)
        zs = np.array(zs)
        return rs, zs

    def data_for_regression_all(self):
        """Data for regression for all years"""
        dates = dates_insample_monthly()
        year1 = pd.to_datetime(dates[0]).year
        year2 = pd.to_datetime(dates[-1]).year
        dr = {}
        for y in range(year1,year2+1):
            dr[y] = self.data_for_regression(y)
            print ('Regression data loaded for year', y)
        self.data_reg = dr
        return dr

    def regression(self):
        """Regression"""
        years = list(self.data_reg.keys())
        tvals = {}
        X_all_years = []
        for year in years:
            X = self.data_reg[year][1]
            y = self.data_reg[year][0]
            est = sm.OLS(y,X).fit()
            tvals[year] = est.tvalues
            where_are_nans = np.isnan(tvals[year])
            tvals[year][where_are_nans] = 0.
            if year == years[0]:
                X_all_years = X
            else:
                X_all_years = np.concatenate((X_all_years,X))
        mean_t = np.array([np.array(list(tvals.values())).T[i].mean() for i in range(28)])
        std_t = np.array([np.array(list(tvals.values())).T[i].std() for i in range(28)])
        t = mean_t / std_t
        self.t = t
        corr = np.corrcoef(X_all_years.T)
        where_are_nans = np.isnan(corr)
        corr[where_are_nans] = 0.
        correlated = []
        for i in range(28):
            for j in range(i+1,28):
                if abs(corr[i,j]) > 0.9:
                    correlated.append(i)
        correlated = set(correlated)         
        return t, correlated

    def number_of_factors(self):
        """We leave several factors"""
        _, correlated = self.regression()
        
        n = 5
        while True:
            n+=1
            ii = self.t.argsort()[-n:][::-1]
            # Exclude correlated columns
            ii = list(set(ii) - correlated)
            if len(ii)>=6:
                break
        
        self.ii = ii


    def monte_carlo(self,t,ii):
        """Monte-Carlo simulations"""
        dates = dates_insample_monthly()
        year1 = pd.to_datetime(dates[0]).year - 1
        year2 = pd.to_datetime(dates[-1]).year + 1
        years = []
        for y in range(year1,year2):
            year = str(dt.date(y,12,31))
            years.append(year)
        f = {} # returns and factors in dict
        r = {}
        f_list = []
        r_list = []
        factors = {}
        for i in range(1,len(years)):
            factors[years[i]] = self.z_scoring(years[i])
        
        for ticker in self.ticker_sample:
            for i in range(1,len(years)):
                try:
                    date1 = self.returns[ticker].index.asof(pd.to_datetime(years[i-1]))
                    date2 = self.returns[ticker].index.asof(pd.to_datetime(years[i]))
                    f_list.append(factors[years[i]][ticker][self.ii])
                    r_list.append((self.returns[ticker].loc[date2]['Return'] + 1) / (self.returns[ticker].loc[date1]['Return'] + 1) - 1)
                except:
                    f_list.append(np.zeros([len(self.ii)]))
                    r_list.append(0.)
            f[ticker] = np.array([np.mean(x) for x in np.array(f_list).T]) / np.array([np.std(x) for x in np.array(f_list).T])
            where_are_nans = np.isnan(f[ticker])
            f[ticker][where_are_nans] = 0.
            r[ticker] = np.mean(r_list) / np.std(r_list)
        sorted_r = sorted(r.items(), key=operator.itemgetter(1))
        r_top = dict(sorted_r[-TOP_STOCKS:])
        top_tickers = list(r_top.keys())
        stock_averaged_top_factors = np.array([np.mean(x) for x in np.array([f[tt] for tt in top_tickers]).T])

        # Monte-Carlo part
        max_factor = 0.
        for k in range(1000):
            w = np.array([random.randint(-2,2) * 0.25 for n in range(len(self.ii))])
            mf = sum(stock_averaged_top_factors * w)
            if mf > max_factor:
                max_factor = mf
        self.top_stocks = top_tickers
        self.weights = w


    def output(self):
        """Results and graphs"""
        dates_in = dates_insample_monthly()
        dates_out = dates_outsample_monthly()
        dates = dates_in + dates_out
        year1 = pd.to_datetime(dates[0]).year
        year2 = pd.to_datetime(dates[-1]).year
        years = []
        for y in range(year1,year2):
            year = str(dt.date(y,12,31))
            years.append(year)
        snp500_df = get_stock_df('^GSPC',years[0],years[-1])
        returns_snp500 = returnsMonthly({'^GSPC':snp500_df},self.column_name)['^GSPC']
        returns_snp500 = returns_snp500.groupby(returns_snp500.index).first()
        returns_snp500 = returns_snp500.loc[[returns_snp500.index.asof(pd.to_datetime(date)) for date in years]]
        returns_snp500.dropna(inplace=True)

        top_returns = []
        for date in years:
            raw_factors = np.array(allFactors(list(self.dfs.values()),date,self.column_name))[self.ii].T
            where_are_nans = np.isnan(raw_factors)
            raw_factors[where_are_nans] = 0.
            # Final unified factor
            fufs = np.array([x.sum() for x in (raw_factors * self.weights)])
            top_stocks_idx = fufs.argsort()[-TOP_STOCKS:][::-1]
            top_stocks = np.array(self.ticker_sample)[top_stocks_idx].tolist()
            top_dfs = {k:self.dfs[k] for k in top_stocks if k in self.dfs} # sub-dictionary of dfs
            top_returns_monthly = returnsMonthly(top_dfs,self.column_name)
            tr =0.
            for s in top_stocks:
                try:
                    tr = top_returns_monthly[s].loc[top_returns_monthly[s].index.asof(pd.to_datetime(date))]['Return']
                except:
                    tr = 0.
                tr += tr
                tr /= TOP_STOCKS
            top_returns.append(tr)
            print ('Output generated as of', date[:4])
        returns_stocks = pd.DataFrame(top_returns,index=years,columns=['Return'])

        # Graphs
        index_snp500 = [str(i)[:4] for i in returns_snp500.index]
        returns_snp500.index = index_snp500
        returns_snp500.columns = ['S&P500']
        index_stocks = [str(i)[:4] for i in returns_stocks.index]
        returns_stocks.index = index_stocks
        returns_stocks.columns = ['Top Stocks']
        df = pd.concat([returns_snp500,returns_stocks],axis = 1).fillna(0.)
        matplotlib.style.use('ggplot')
        df.plot(kind='bar',alpha=0.5)
        plt.xticks(rotation='horizontal')
        plt.ylabel('Return')
        plt.title("YoY performance of the Top decile stocks Vs. S&P Index")
        plt.show()
        plt.savefig("Return.pdf")
        df2 = df.copy()
        #df2.index = df2.index.values + 1
        r0 = df.ix[:,0].values
        r1 = df.ix[:,1].values
        df2.ix[:,0] = np.cumprod(r0 + 1) * 100
        df2.ix[:,1] = np.cumprod(r1 + 1) * 100
        df2.plot(alpha=0.5)
        #plt.ticklabel_format(useOffset=False)
        plt.ylabel('$')
        plt.title("Technical Factor Basket Vs. S&P Index Return")
        plt.savefig("Value.pdf")


# MAIN PART

if __name__ == "__main__":
    #dates = dates_insample_monthly()
    print ('Simulations STARTED...')
    Z = Factors() # class initialisation
    print "Assembling data for regression"
    Z.data_for_regression_all() # assembling data for regression
    print "Reducing number of factors"
    Z.number_of_factors() # we leave several factors and doing regression
    print "Doing simulations"
    Z.monte_carlo(Z.t,Z.ii) # monte-carlo simulations
    print "Preparing output"
    Z.output() # output
    print ('Simulations DONE.')