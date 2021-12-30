import seaborn
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint



def hurst(ts):
    """Return the Hurst Exponent of the time series
    Args:
        ts: 
            the time series
    Returns:
        the Hurst Exponent
    """
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    
    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0



def adf(data):
    """Compute the Augmented Dickey Fuller (ADF) test output thne results
    Args:
        data: 
            the stock for wich we want to compute the test
    """
    result = adfuller(data)
    print('Test statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Number lags used %f' % result[2])
    print('Number of Observations Used %f' % result[3])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def compute_spread(stock1,stock2):
    """Compute the spread between two stocks
    Args:
        stock1: 
            the first stock
        stock2:
            the second stock
    Returns:
        spread: 
            the spread
        spread_normalized:
            the the normalized spread
        spread_mean: 
            the empirical mean
        spread_std:
            the empirical standard deviation
        alpha:
            the alpha of the OLS regression betwen the two stocks
        hedge_ratio:
            the headge ratio bewteen the two stock stocks compute during the last day
    """
    name_stock1 = stock1.name
    
    results = sm.OLS(stock2, sm.add_constant(stock1)).fit()
    alpha = results.params[name_stock1]

    # Z_t = S2 - alpha* S1
    spread = stock2 - alpha * stock1
    spread_mean = spread.mean()
    spread_std = spread.std()
    spread_normalized = (spread - spread_mean) / spread_std
    
    hedge_ratio = alpha * stock1[-1] / stock2[-1]
    
    return spread, spread_normalized, spread_mean, spread_std, alpha, hedge_ratio

def plot_spread(spread_normalized, buy_spread, sell_spread, threshold_activation = 1, threshold_exit = 0.25, threshold_loss = 2):
    
    """Plot the z-score of the spread
    Args:
        spread_normalized:
            the the normalized spread
        threshold_activation:
            the threshold socre to start holding a position
        threshold_exit:
            the threshold socre to exit a position
        threshold_loss:
            the threshold socre to exit a position and cut the losses    
    """
    fig, ax = plt.subplots(figsize=(10,8))
    
    spread_normalized.plot(ax = ax, label = 'spread')
    plt.axhline(0, color='black')  
    
    plt.axhline(threshold_activation, color='green', label = 'activation threshold')
    plt.axhline(-threshold_activation, color='green')
    
    plt.axhline(threshold_exit, color='yellow', label ='exit threshold')
    plt.axhline(-threshold_exit, color='yellow')
    
    plt.axhline(threshold_loss, color='red', label = 'loss threshold')
    plt.axhline(-threshold_loss, color='red')
    
    buy_spread.plot(ax = ax, color='g', linestyle='None', marker='^', label = 'buy')
    sell_spread.plot(ax = ax, color='r', linestyle='None', marker='v', label = 'sell')
    plt.ylim([-4, 4])
    plt.xlim([spread_normalized.index.min(), spread_normalized.index.max()])
    
    plt.legend()
    plt.show()
    
def estimation_phase(data):
    """Estimate which stocks are cointegrrated
    Args:
        data:
            DataFrame with all the stocks
    Returns:
        cointegrated_stocks: 
            a list of stock names tuple that are cointegrated
        spread_means:
            a list of the spread means for each cointegrated pair
        spread_stds: 
            a list of the spread standard deviations for each cointegrated pair
        alphas:
             a list of alphas for each cointegrated pair
        hedge_ratios:
            a list of hedge ratios for each cointegrated pair
    """
    cointegrated_stocks = []
    spread_means = []
    spread_stds = []
    alphas = []
    hedge_ratios = []
    
    for i,name_stock1 in enumerate(data):
        for j,name_stock2 in enumerate(data):
            if name_stock1 != name_stock2 and i > j:
                
                stock1 = data[name_stock1]
                stock2 = data[name_stock2]
                
                #cointgration test
                score, pvalue, _ = coint(stock1, stock2)
                
                if(pvalue < 0.05):
                    
                    cointegrated_stocks.append((name_stock1, name_stock2))
                    
                    spread, spread_normalized, spread_mean, spread_std, alpha, hedge_ratio = compute_spread(stock1, stock2)
                    
                    spread_means.append(spread_mean)
                    spread_stds.append(spread_std)
                    alphas.append(alpha)
                    hedge_ratios.append(hedge_ratio)
     
    return cointegrated_stocks, spread_means, spread_stds, alphas, hedge_ratios




def cointegration_pvalue_marix(data):
    """Compute the p-values for each pair of stocks 
    Args:
        data:
            DataFrame with all the stocks
    Returns:
        pvalue_matrix:
            A matrix containing the p-values of each pair of stock
    """
    pvalue_matrix = np.ones((len(data.columns), len(data.columns)))
    cointegrated_stocks = []

    for i,name_stock1 in enumerate(data):
        for j,name_stock2 in enumerate(data):
            if name_stock1 != name_stock2 and i > j:

                stock1 = data[name_stock1]
                stock2 = data[name_stock2]

                #cointgration test
                score, pvalue, _ = coint(stock1, stock2)

                if(pvalue < 0.05):
                    print(name_stock1+' and '+name_stock2+ ' are cointegrated')
                    cointegrated_stocks.append((stock1,stock2))

                    spread, spread_normalized, spread_mean, spread_std, alpha, hedge_ratio = compute_spread(stock1, stock2)

                    #plot the spread if needed 
                    #plot_spread(spread_normalized,threshold_activation)
                pvalue_matrix[i,j] = pvalue
    return pvalue_matrix

def plot_heatmap(data, pvalue_matrix):
    """Plot a heatmap of the p-values of the cointegration test
    Args:
        data:
            DataFrame with all the stocks
    """
    fig, ax = plt.subplots(figsize = (20,10))
    tickers = data.columns.values
    plt.title('Heatmap of p-values for cointegration test')
    seaborn.heatmap(pvalue_matrix, xticklabels=tickers, yticklabels=tickers, cmap='Greens_r' , mask = (pvalue_matrix >= 0.05), ax= ax)
    plt.savefig('heatmap.png', dpi=300)
    plt.show()
    
    
def value_strategy(stock1, stock2, alpha, spread_mean, spread_std, hedge_ratio, threshold_activation = 1, threshold_exit = 0.25, threshold_loss = 2, plot = False):
    
    #We start with no money and no shares
    money = 0
    stock1_shares = 0
    stock2_shares = 0
    
    # Z_t = S2 - alpha* S1
    spread = stock2 - alpha * stock1
    spread_normalized = (spread - spread_mean) / spread_std
    spread_normalized_trend = spread_normalized.diff().fillna(0)
    spread_normalized_trend.apply(lambda x : 1 if x>=0 else -1)
    spread_normalized_trend = spread_normalized_trend.rolling(window = 5).sum().fillna(0)
    
    positioned = False
    
    buy = [-1000] * len(spread_normalized)
    sell = [-1000] * len(spread_normalized)
    
    for i, (z_score, slope) in enumerate(zip(spread_normalized, spread_normalized_trend)):
        
        # Short the spread if the threshold_activation < z_score < threshold_loss
        # Sell 1 stock2 and buy hedge_ratio * stock1
        if( (threshold_activation < z_score < threshold_loss) and (slope < 0) and not positioned):
            
            sell[i] = z_score
            money += - hedge_ratio * stock1[i] + stock2[i] 
            stock1_shares += hedge_ratio
            stock2_shares -= 1
            positioned = True
        
        # The absolute spread is too high need to cut the losses and clear the position
        # The spread is back to normal need to exit the position
        # Sell everything 
        elif( ((np.abs(z_score) > threshold_loss) or (np.abs(z_score) < threshold_exit)) and positioned):
            
            #In short position need to buy
            if(stock1_shares > 0):
                buy[i] = z_score
            else: 
                sell[i] = z_score
        
            money += stock1_shares * stock1[i] + stock2_shares * stock2[i] 
            stock1_shares = 0
            stock2_shares = 0
            positioned = False
            
        # Remaining case is -threshold_loss < z_score < -threshold_activation
        # Long the spread 
        # Buy 1 stock2 and sell H * stock1
        elif( (-threshold_loss < z_score < -threshold_activation) and (slope > 0) and not positioned ): 
            
            buy[i] = z_score
            money +=  hedge_ratio * stock1[i] - stock2[i] 
            stock1_shares -= hedge_ratio
            stock2_shares += 1
            positioned = True
            
    #need to exit the position at the end 
    if(positioned):
        #if in short poisition need to buy
        if(stock1_shares > 0):
            buy[i] = spread_normalized[-1]
        else: 
            sell[i] = spread_normalized[-1]
            
        money += stock1_shares * stock1[i] + stock2_shares * stock2[i] 
        stock1_shares = 0
        stock2_shares = 0
        positioned = False
    
    #plot the spread if needed 
    if(plot):
        plot_spread(spread_normalized, pd.DataFrame(buy, index = spread_normalized.index, columns = ['buy']), pd.DataFrame(sell, index = spread_normalized.index, columns = ['sell']),\
                    threshold_activation, threshold_exit , threshold_loss)
        
    return money