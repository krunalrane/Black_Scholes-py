import math 
import numpy as np 
import pandas as pd  # type: ignore
import yfinance as yf 
import requests
import json 
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import brentq


# Black-Scholes Formula Creation 
def _d1(S, K, r, q, sigma, tau):
    return (math.log(S / K) + (r -q + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))

def _d2(d1, sigma, tau):
    return d1 - sigma * math.sqrt(tau)

def call_price(S, K, r, q, sigma, tau):
    '''European call price (continous dividend yeild q)'''
    if tau <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, r, q, sigma, tau)
    d2 = _d2(d1, sigma, tau)
    return math.exp(-q * tau) * S * norm.cdf(d1) - math.exp(-r * tau) * K * norm.cdf(d2)

def put_price(S, K, r, q, sigma, tau):
    '''closed form'''
    if tau <= 0:
        return max(K - S, 0.0)
    d1 = _d1(S, K, r, q, sigma, tau)
    d2 = _d2(d1, sigma, tau)
    return math.exp(-r * tau) * K * norm.cdf(-d2) - math.exp(-q * tau) * S * norm.cdf(-d1)

# Historical Volatility 
def historical_v_p(price_series, window_days=252, annualize=True):
    '''
    Here historical_v_p means that historical volatility from prices,
    price_series = it is adjusted close prices indexed by date (most recent last),
    window_days = how many days the trading goes on like working days.
    annualize = returns vol (decimal).
    '''

    returns = np.log(price_series / price_series.shift(1)).dropna()
    vol_daily = returns.rolling(window_days).std(ddof=1).iloc[-1]
    if math.isnan(vol_daily):
        vol_daily = returns.std(ddof=1)
    if annualize:
        return float(vol_daily * math.sqrt(252))
    return float(vol_daily)


#Implied volatility (numerical)
def implied_v(option_market_price, S, K, r, q, tau, is_call=True, tol=1e-8, maxiter=100):
    """
    Solve for sigma such that Black-Scholes price matches market price.
    Uses Brent's method between bounds [1e-8, 5] (0% - 500% vol).
    """
    if option_market_price <= 0:
        return 0.0
    # price function for root-finding
    def price_given_sigma(sigma):
        return(call_price(S, K, r, q, sigma, tau) if is_call else put_price(S< K, r, q, sigma, tau)) - option_market_price
    
    try:
        implied = brentq(price_given_sigma, 1e-8, 5.0, xtol=tol, maxiter=maxiter)
        return float(implied)
    except Exception as e:
        # For Numeric failure will return NaN
        return float('nan')
    
# Helpers: fetch data 
def fetch_spot_and_history(ticker, period='1y', interval='1d'):
    """
    Returns spot_price (last close) and a pandas Series of adjclose prices.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval=interval, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No history for {ticker}")
    # 'Close' or 'Adj Close' after auto_adjust -> 'Close' is adjusted
    prices = hist['Close']
    spot = float(prices.iloc[-1])
    return spot, prices

def fetch_option_chain(ticker, expiry_date):
    """
    expiry_date: string like '2025-10-17' from ticker.options list
    Returns calls_df, puts_df (pandas DataFrames).
    """
    t = yf.Ticker(ticker)
    # yfinance returns (calls, puts) via option_chain() but often provides option_chain(expiry) method
    oc = t.option_chain(expiry_date)
    calls = oc.calls.copy()
    puts = oc.puts.copy()
    return calls, puts

def fetch_treasury_rate():

    url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/debt_subject_to_limit"	

    return 0.01 # 1% as an example - replace with a proper API call / parsing for production.

# Example workflow: single option example 
def example_run(ticker='AAPL', expiry=None, strike=None, is_call=True):
    """
    If expiry/strike omitted, pick nearest expiry and ATM strike.
    """
    # Fetch spot and history 
    spot, prices = fetch_spot_and_history(ticker, period='1y')

    # volatility estimate (historica)
    hist_vol = historical_v_p(prices, window_days=90)  # 90-day historical vol
    
    # choose expiry 
    t = yf.Ticker(ticker)
    options = t.options
    if not options:
        raise ValueError("No option expiries available for ticker " + ticker)
    if expiry is None:
        expiry = options[0]  # closest expiry
    calls, puts = fetch_option_chain(ticker, expiry)
    # pick ATM strike if not given
    if strike is None:
        # approximate ATM by finding strike nearest to spot
        all_strikes = calls['strike'].values if is_call else puts['strike'].values
        strike = float(min(all_strikes, key=lambda k: abs(k - spot)))
    # find market price for selected option
    df_side = calls if is_call else puts
    row = df_side[df_side['strike'] == strike]
    if row.empty:
        raise ValueError("Strike not found in option chain")
    market_mid = (float(row['bid'].values[0]) + float(row['ask'].values[0])) / 2.0
    # if bid/ask are zero or NaN, fallback to lastPrice
    if market_mid <= 0 or math.isnan(market_mid):
        market_mid = float(row['lastPrice'].values[0])

    # risk-free rate (example)
    r = fetch_treasury_rate()

    # time to maturity in years 
    exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
    tau = max((exp_dt - datetime.utcnow()).days / 365.0, 0.0)

    # compute theoretical price using historical vol 
    q = 0.0 # we can adjust it if the stock pays dividend yield 
    model_price_histvol = (call_price if is_call else put_price)(spot, strike, r, q, hist_vol, tau)

    # implied vol from market price 
    implied_vol = implied_v(market_mid, spot, strike, r, q, tau, is_call=is_call)

    out = dict(
        ticker=ticker,
        expiry=expiry,
        strike=strike,
        spot=spot,
        market_price=market_mid,
        hist_vol= hist_vol,
        model_price_histvol=model_price_histvol,
        implied_vol=implied_vol,
        tau=tau,
        risk_free_rate=r
    ) 
    return out 



if __name__ == "__main__":
    res = example_run('AAPL')
    print(json.dumps(res, indent=2))