from statsmodels.regression.rolling import RollingOLS
#import pandas_datareader.data as web
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import datetime as dt
import yfinance as yf
import requests
from io import StringIO
import os
import pandas_ta

wikiurl="https://en.wikipedia.org/wiki/NIFTY_50"
CACHE_FILE= "nifty50data.parquet"

if os.path.exists(CACHE_FILE):
    df=pd.read_parquet(CACHE_FILE)
else:
    headers={
        "User-Agent":"Mozilla/5.0"
    }
    response=requests.get(wikiurl,headers=headers)
    tables=pd.read_html(StringIO(response.text))
    nifty50=tables[1]
    nifty50.columns = ['Company Name','Symbol','Sector','Date added']
    symbolslist = nifty50["Symbol"].tolist()
    symbolslist=[s+'.NS' for s in symbolslist]
    symbolslist=symbolslist[:10] #for testing lets take only 10 stocks
    enddate=pd.Timestamp.today()
    startdate=enddate-pd.DateOffset(years=5)
    df=yf.download(tickers=symbolslist,start=startdate,end=enddate).stack(future_stack=True)
    df.index.names=['date','ticker']
    df.columns=df.columns.str.lower()
    df.to_parquet(CACHE_FILE)

df['garman_klass_vol']=((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['close'])-np.log(df['open']))**2)
df['rsi']=df.groupby(level=1)['close'].transform(lambda x:pandas_ta.rsi(close=x,length=20))
df['bb_low']=df.groupby(level=1)['close'].transform(lambda  x: pandas_ta.bbands(close=np.log1p(x),length=20).iloc[:,0])
df['bb_mid']=df.groupby(level=1)['close'].transform(lambda  x: pandas_ta.bbands(close=np.log1p(x),length=20).iloc[:,1])
df['bb_high']=df.groupby(level=1)['close'].transform(lambda  x: pandas_ta.bbands(close=np.log1p(x),length=20).iloc[:,2])
def compute_atr(data):
    atr=pandas_ta.atr(high=data['high'],low=data['low'],close=data['close'],length=14)
    return atr.sub(atr.mean()).div(atr.std())
df['atr']=df.groupby(level=1,group_keys=False).apply(compute_atr)
def compute_macd(close):
    macd=pandas_ta.macd(close=close).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())
df['macd']=df.groupby(level=1,group_keys=False)['close'].apply(compute_macd)
df['rs_vol']=(df['volume']*df['close'])/1e6
print(df)
