import requests
import os
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()


def get_alpha_stock_data(symbol: str) -> dict:
    """Fetch daily stock data from Alpha Vantage API."""
    API_KEY = os.getenv("ALPHA_API_KEY")
        #symbol = "AAPL"

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    return data

# Extract date-wise trade history from the API response
def extract_trade_history(api_data):
    """
    Extract date-wise trade history from Alpha Vantage API response
    Returns a JSON object with structured trade data
    """
    trade_history = {
        "meta_data": {},
        "time_series": []
    }
    
    # Extract metadata
    if "Meta Data" in api_data:
        meta = api_data["Meta Data"]
        trade_history["meta_data"] = {
            "symbol": meta.get("2. Symbol", ""),
            "last_refreshed": meta.get("3. Last Refreshed", ""),
            "interval": meta.get("4. Interval", ""),
            "output_size": meta.get("5. Output Size", ""),
            "time_zone": meta.get("6. Time Zone", "")
        }
    
    # Extract time series data
    if "Time Series (Daily)" in api_data:
        time_series = api_data["Time Series (Daily)"]
        
        for date, values in time_series.items():
            trade_record = {
                "date": date,
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "volume": int(values.get("5. volume", 0))
            }
            trade_history["time_series"].append(trade_record)
    
    # Sort by date (most recent first)
    trade_history["time_series"].sort(key=lambda x: x["date"], reverse=True)
    
    # Add summary statistics
    if trade_history["time_series"]:
        closes = [t["close"] for t in trade_history["time_series"]]
        volumes = [t["volume"] for t in trade_history["time_series"]]
        
        trade_history["summary"] = {
            "total_records": len(trade_history["time_series"]),
            "highest_close": max(closes),
            "lowest_close": min(closes),
            "average_close": sum(closes) / len(closes),
            "total_volume": sum(volumes)
        }
    
    return trade_history

def extract_dates_and_prices(trade_data):
    """
    Extract dates and close prices in simple format
    Returns: {"dates": dates, "prices": prices}
    """
    dates = [record["date"] for record in trade_data["time_series"]]
    prices = [record["close"] for record in trade_data["time_series"]]
    
    return {
        "dates": dates,
        "prices": prices
    }

def stock_data_to_json(symbol: str) -> dict:
    """
    Fetch stock data and convert to structured JSON format
    """
    data = get_alpha_stock_data(symbol)
    trade_data = extract_trade_history(data)
    price_data = extract_dates_and_prices(trade_data)
    return price_data

if __name__ == "__main__":
    symbol = "AAPL"
    stock_data = stock_data_to_json(symbol)
    print(json.dumps(stock_data, indent=2))
