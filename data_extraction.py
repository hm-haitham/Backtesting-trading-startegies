import dask
import pandas as pd
import numpy as np
import glob

dask.config.set(scheduler="processes")

@dask.delayed
def load_TRTH_trade(filename,
             tz_exchange="America/New_York",
             only_non_special_trades=False,
             only_regular_trading_hours=False,
             include_trade_volume=False,
             open_time="09:30:00",
             close_time="16:00:00",
             merge_sub_trades=True):
   
    try:
        DF = pd.read_csv(filename)
    except:
        return None
    
    if DF.shape[0] ==0:
        return None
    
    
    
    if only_non_special_trades:
        DF = DF[DF["trade-stringflag"]=="uncategorized"]
    if not include_trade_volume:
        DF.drop(columns=["trade-volume"],axis=1,inplace=True)
    DF.drop(columns=["trade-rawflag","trade-stringflag"],axis=1,inplace=True)
    
    DF.index = pd.to_datetime(DF["xltime"],unit="d",origin="1899-12-30",utc=True)
    DF.index = DF.index.tz_convert(tz_exchange)  # .P stands for Arca, which is based at New York
    DF.drop(columns="xltime",inplace=True)
    
    if only_regular_trading_hours:
        DF=DF.between_time(open_time,close_time)    # warning: ever heard e.g. about Thanksgivings?
    
    if merge_sub_trades:
        if include_trade_volume:
            DF=DF.groupby(DF.index).agg(trade_price=pd.NamedAgg(column='trade-price', aggfunc='mean'),
                                       trade_volume=pd.NamedAgg(column='trade-volume', aggfunc='sum'))
        else:
            DF=DF.groupby(DF.index).agg(trade_price=pd.NamedAgg(column='trade-price', aggfunc='mean'))
    
    return DF


@dask.delayed
def load_TRTH_bbo(filename,
             tz_exchange="America/New_York",
             open_time="09:30:00",
             close_time="16:00:00",
             only_regular_trading_hours=True):
    
        
    try:
        DF = pd.read_csv(filename)
    except:
        return None
    
    if DF.shape[0] ==0:
        return None
    
    DF.index = pd.to_datetime(DF["xltime"],unit="d",origin="1899-12-30",utc=True)
    DF.index = DF.index.tz_convert(tz_exchange)  # .P stands for Arca, which is based at New York
    DF.drop(columns="xltime",inplace=True)
    
    if only_regular_trading_hours:
        DF=DF.between_time(open_time,close_time)    # warning: ever heard e.g. about Thanksgivings?
        
    return DF

def extract_trade_df(path="../data/extracted_trade/*"):
    extracted_trade_gz = glob.glob(path)
    tickers = []
    for f in extracted_trade_gz:
        ticker = f.split("-")[3]
        ticker = "-" + ticker
        tickers.append(ticker)

    tickers = list(set(tickers))
    first_ticker = [f for f in extracted_trade_gz if tickers[0] in f]
    promises = [load_TRTH_trade(fn) for fn in first_ticker]
    first_trade = dask.compute(promises)[0]
    first_trade = pd.concat(first_trade)
    first_trade.columns = [str(tickers[0][1:-2])]
    first_trade = first_trade.resample("Min").first()

    for ticker in tickers[1:]:
        ticker_files = [f for f in extracted_trade_gz if ticker in f]

        promises = [load_TRTH_trade(fn) for fn in ticker_files]
        trade = dask.compute(promises)[0]
        trade = pd.concat(trade)
        trade.columns = [str(ticker[1:-2])]
        trade = trade.resample("Min").first()
        trade = trade.dropna()
        first_trade = first_trade.merge(trade, how="outer", left_index=True, right_index=True)

    return first_trade