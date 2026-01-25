"""
Utility module for fetching stock history data using yfinance
"""
import yfinance as yf
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time


def _create_session():
    """Create a requests session with retry strategy."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504)
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    return session


def get_stock_history(
    ticker: str,
    period: str = "3mo",
    interval: str = "1d"
) -> Dict:
    """
    Gets historical stock price data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max')
        interval: Data interval ('1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo')
    
    Returns:
        Dictionary with dates, prices, and OHLCV data
    """
    try:
        session = _create_session()
        stock = yf.Ticker(ticker, session=session)
        
        # Add retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                history = stock.history(period=period, interval=interval)
                if not history.empty:
                    break
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        if history.empty:
            return {"error": f"No history found for {ticker}"}
        
        history.reset_index(inplace=True)
        
        result = {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "data": []
        }
        
        for _, row in history.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date'])
            result["data"].append({
                "date": date_str,
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume'])
            })
        
        return result
    
    except Exception as e:
        error_msg = str(e)
        return {"error": f"Failed to fetch history for {ticker}: {error_msg}"}


def get_stock_history_simple(
    ticker: str,
    period: str = "3mo"
) -> Dict:
    """
    Simple version - returns just dates and close prices.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period
    
    Returns:
        Dictionary with dates and close prices
    """
    try:
        session = _create_session()
        stock = yf.Ticker(ticker, session=session)
        
        # Add retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                history = stock.history(period=period)
                if not history.empty:
                    break
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        if history.empty:
            return {"error": "No history found"}
        
        history.reset_index(inplace=True)
        dates = history['Date'].dt.strftime('%Y-%m-%d').tolist()
        prices = history['Close'].tolist()
        
        return {
            "ticker": ticker,
            "dates": dates,
            "prices": prices,
            "count": len(dates)
        }
    
    except Exception as e:
        return {"error": str(e)}


def get_stock_history_range(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> Dict:
    """
    Gets historical data for a specific date range.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval
    
    Returns:
        Dictionary with OHLCV data for the date range
    """
    try:
        session = _create_session()
        stock = yf.Ticker(ticker, session=session)
        
        # Add retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                history = stock.history(start=start_date, end=end_date, interval=interval)
                if not history.empty:
                    break
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        if history.empty:
            return {"error": f"No data found between {start_date} and {end_date}"}
        
        history.reset_index(inplace=True)
        
        result = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "data": []
        }
        
        for _, row in history.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date'])
            result["data"].append({
                "date": date_str,
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume'])
            })
        
        return result
    
    except Exception as e:
        return {"error": str(e)}


def get_multiple_stocks_history(
    tickers: List[str],
    period: str = "3mo"
) -> Dict:
    """
    Gets historical data for multiple stocks.
    
    Args:
        tickers: List of stock ticker symbols
        period: Time period
    
    Returns:
        Dictionary with data for all tickers
    """
    results = {}
    
    for ticker in tickers:
        results[ticker] = get_stock_history_simple(ticker, period)
    
    return results


# Example usage
if __name__ == "__main__":
    # Example 1: Get last 3 months of data
    print("=== Last 3 Months (OHLCV) ===")
    result = get_stock_history("AAPL", period="3mo")
    if "error" not in result:
        print(f"Ticker: {result['ticker']}")
        print(f"Data points: {len(result['data'])}")
        if result['data']:
            print(f"Latest: {result['data'][-1]}")
    else:
        print(f"Error: {result['error']}")
    
    # Example 2: Get simple close prices
    print("\n=== Simple Close Prices (Last 3 Months) ===")
    result = get_stock_history_simple("MSFT", period="3mo")
    if "error" not in result:
        print(f"Ticker: {result['ticker']}")
        print(f"Data points: {result['count']}")
        if result['dates']:
            print(f"Date range: {result['dates'][0]} to {result['dates'][-1]}")
            print(f"Price range: ${min(result['prices']):.2f} to ${max(result['prices']):.2f}")
    else:
        print(f"Error: {result['error']}")
    
    # Example 3: Get specific date range
    print("\n=== Specific Date Range (OHLCV) ===")
    result = get_stock_history_range("TSLA", "2025-10-01", "2026-01-20")
    if "error" not in result:
        print(f"Ticker: {result['ticker']}")
        print(f"Data points: {len(result['data'])}")
        if result['data']:
            print(f"Latest: {result['data'][-1]}")
    else:
        print(f"Error: {result['error']}")
    
    # Example 4: Get multiple stocks
    print("\n=== Multiple Stocks ===")
    result = get_multiple_stocks_history(["AAPL", "MSFT", "GOOGL"], period="3mo")
    for ticker, data in result.items():
        if "error" not in data:
            print(f"{ticker}: {data['count']} data points")
        else:
            print(f"{ticker}: Error - {data['error']}")
