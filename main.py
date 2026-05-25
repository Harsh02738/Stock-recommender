from statsmodels.regression.rolling import RollingOLS
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models,expected_returns
import warnings
warnings.filterwarnings('ignore')

wikiurl="https://en.wikipedia.org/wiki/NIFTY_500"
CACHE_FILE= "nifty500data.parquet"
PRICE_CACHE = "pricedatadaily.parquet"
outlier_cutoff=0.005

if os.path.exists(CACHE_FILE):
    df=pd.read_parquet(CACHE_FILE)
else:
    headers={
        "User-Agent":"Mozilla/5.0"
    }
    response=requests.get(wikiurl,headers=headers)
    tables=pd.read_html(StringIO(response.text))
    nifty500=tables[4]
    print(nifty500.columns.tolist())
    nifty500.columns = ['Slno','Company Name','Industry','Symbol' ,'Series' ,'ISIN Code']
    symbolslist = nifty500["Symbol"].tolist()
    symbolslist=[s+'.NS' for s in symbolslist]
    symbolslist=symbolslist[1:]
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
    high = data['high'].astype(float)
    low = data['low'].astype(float)
    close = data['close'].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    return atr.sub(atr.mean()).div(atr.std())
df['atr']=df.groupby(level=1,group_keys=False).apply(compute_atr)
def compute_macd(close):
    macd=pandas_ta.macd(close=close).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())
df['macd']=df.groupby(level=1,group_keys=False)['close'].apply(compute_macd)
df['rs_vol']=(df['volume']*df['close'])/1e6

lastcols=[c for c in df.columns.unique(0) if c not in ['rs_vol', 'open','volume','high','low','adj close']]
features=df.unstack()[lastcols].resample('ME').last().stack('ticker')
vol=df.unstack('ticker')['rs_vol'].resample('ME').mean().stack('ticker').to_frame('rs_vol')
data=pd.concat([features,vol],axis=1).dropna()

data['rs_vol']=data['rs_vol'].unstack('ticker').rolling(5*12).mean().stack()
data=data.drop(['rs_vol'],axis=1)

def compute_returns(df):
    lags=[1,2,3,6,9,12]
    for lag in lags:
        df[f'return_{lag}m']=df['close'].pct_change(lag).pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),upper=x.quantile(1-outlier_cutoff))).add(1).pow(1/lag).sub(1)
    return df

data=data.groupby(level=1,group_keys=False).apply(compute_returns).dropna()
data=data.drop('close', axis=1) 

factordata=pd.read_csv("FFdata.csv",parse_dates=['Date']).drop('RF',axis=1)
factordata = factordata.set_index('Date')
factordata=factordata.resample('ME').last().div(100)
factordata.index.name='date'
factordata=factordata.join(data['return_1m']).sort_index()
obs=(factordata.groupby(level=1).size())
valid=obs[obs>=10]
factordata=factordata[factordata.index.get_level_values('ticker').isin(valid.index)]

betas=(factordata.groupby(level=1,group_keys=False).apply(lambda x:RollingOLS(endog=x['return_1m'],
       exog=sm.add_constant(x.drop('return_1m',axis=1)),window=min(24,x.shape[0]),
       min_nobs=len(x.columns)+1).fit(params_only=True).params.drop('const',axis=1)))

data=data.join(betas.groupby('ticker').shift())
factors=['SMB','HML','WML','MF']
data.loc[:,factors]=data.groupby('ticker',group_keys=False)[factors].apply(lambda x:x.fillna(x.mean()))

def get_clusters(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    df['cluster'] = KMeans(n_clusters=4, random_state=0, init='random').fit(scaled).labels_
    return df
data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)
highest_rsi_cluster = data.groupby(level=0).apply(
    lambda x: x.groupby('cluster')['rsi'].mean().idxmax()
).rename('best_cluster')
data = data.join(highest_rsi_cluster, on='date')
data['is_best'] = data['cluster'] == data['best_cluster']

'''
def plotclusters(data):
    cluster0=data[data['cluster']==0]
    cluster1=data[data['cluster']==1]
    cluster2=data[data['cluster']==2]
    cluster3=data[data['cluster']==3]
    plt.scatter(cluster0.iloc[:,0],cluster0.iloc[:,1],color='red',label="cluster 0")
    plt.scatter(cluster1.iloc[:,0],cluster1.iloc[:,1],color='green',label="cluster 1")
    plt.scatter(cluster2.iloc[:,0],cluster2.iloc[:,1],color='blue',label="cluster 2")
    plt.scatter(cluster3.iloc[:,0],cluster3.iloc[:,1],color='black',label="cluster 3")
    plt.legend()
    plt.show()
    return
plt.style.use('ggplot')
for i in data.index.get_level_values('date').unique().to_list():
    g=data.xs(i,level=0)
    plt.title(f" date{i}")
    plotclusters(g)
'''
filterdf=data[data['is_best']].copy()

filterdf=filterdf.reset_index(level=1)
filterdf.index=filterdf.index+pd.DateOffset(1)
filterdf=filterdf.reset_index().set_index(['date','ticker'])
dates=filterdf.index.get_level_values('date').unique().tolist()
fixeddates={}
for date in dates:
    fixeddates[date]=filterdf.xs(date,level=0).index.to_list()

def optimizeweights(prices):
    returns=expected_returns.mean_historical_return(prices=prices,frequency=252)
    cov=risk_models.sample_cov(prices=prices,frequency=252)
    ef=EfficientFrontier(expected_returns=returns,cov_matrix=cov,weight_bounds=(0,.1),solver='SCS')
    ef.max_sharpe()
    return ef.clean_weights()


if os.path.exists(PRICE_CACHE):
    newdf = pd.read_parquet(PRICE_CACHE)
else:
    stocks = data.index.get_level_values('ticker').unique().to_list()
    newdf = yf.download(tickers=stocks, start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12))
    newdf.to_parquet(PRICE_CACHE)

returnsdf=np.log(newdf['Close']).diff()
portfoliodf=pd.DataFrame()

for startdate in fixeddates.keys():
    enddate=pd.to_datetime(startdate)+pd.offsets.MonthEnd(0)
    cols=fixeddates[startdate]
    optmizationstartdate=pd.to_datetime(startdate)-pd.DateOffset(months=12)
    optmizationenddate=pd.to_datetime(startdate)-pd.DateOffset(days=1)
    optdf=newdf[optmizationstartdate:optmizationenddate]['Close'][cols]
    threshold = int(0.8 * len(optdf))
    optdf = optdf.dropna(axis=1, thresh=threshold)
    optdf = optdf.ffill().bfill()
    if optdf.shape[1] < 2:
        continue
    weights=optimizeweights(prices=optdf)
    weights=pd.DataFrame(weights,index=pd.Series(0))
    
    tempdf=returnsdf[startdate:enddate]
    tempdf=tempdf.stack().to_frame('return').reset_index(level=0)
    tempdf.index.name='ticker'
    tempdf=tempdf.rename(columns={'Date':'date'})
    w=weights.stack().to_frame('weight')
    w.index=w.index.droplevel(0)
    w.index.name='ticker'
    tempdf=tempdf.join(w)
    tempdf=tempdf.reset_index().set_index(['date','ticker'])
    tempdf['weighted_return']=tempdf['return']*tempdf['weight']
    
    portfoliodf=pd.concat([portfoliodf,tempdf],axis=0)

portfoliodf = portfoliodf.dropna()
portfolioreturns = portfoliodf.groupby(level=0)['weighted_return'].sum().to_frame('strategy_return')

nifty500index = yf.download('^CRSLDX', start=portfolioreturns.index.min(), end=portfolioreturns.index.max())
nifty500returns = np.log(nifty500index['Close']).diff().dropna()
nifty500returns.columns = ['nifty500_return']

latest = portfoliodf.dropna().index.get_level_values('date').max()
print(portfoliodf.loc[latest].dropna()[['weight']])

plt.figure(figsize=(14,6))
plt.xticks(rotation=45)
ax=plt.gca()
ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%Y'))
plt.plot((portfolioreturns['strategy_return'].add(1).cumprod() - 1)*100, label='strategy')
plt.plot((nifty500returns['nifty500_return'].add(1).cumprod() - 1)*100, label='NIFTY 500')
plt.legend()
plt.title('strategy vs NIFTY 500')
plt.ylabel('Cumulative Return %')
plt.show()