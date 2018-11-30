import requests
import numpy as np
from glearn.utils.file_cache import FileCache
from glearn.utils.path import script_relpath

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # noqa
import matplotlib.dates as mdates  # noqa


HOST = "https://www.alphavantage.co"
API_KEY = "15NNF4EV8LJBRF4L"


class StockData(object):
    def __init__(self, symbol, function="TIME_SERIES_DAILY", interval=None, compact=True,
                 datatype="json", expires=None):
        """
        Interface for AlphaVantage API:
        https://www.alphavantage.co/documentation

        symbol: stock symbol
        function: data to fetch
            "TIME_SERIES_INTRADAY"
            "TIME_SERIES_DAILY"  (default)
            "TIME_SERIES_DAILY_ADJUSTED"
            "TIME_SERIES_WEEKLY"
            "TIME_SERIES_WEEKLY_ADJUSTED"
            "TIME_SERIES_MONTHLY"
            "TIME_SERIES_MONTHLY_ADJUSTED"
            "GLOBAL_QUOTE"  (single latest datapoint)
        interval: datapoint interval  (TIME_SERIES_INTRADAY only)
            "1min", "5min", "15min", "30min" or "60min" (default)
        compact: only latest 100 datapoints, else up to 20 years of history
        datatype: "json" (default) or "csv"

        Example Result:
        {
            "Meta Data": {
                "1. Information": "Daily Prices (open, high, low, close) and Volumes",
                "2. Symbol": "CMNWX",
                "3. Last Refreshed": "2018-11-28",
                "4. Output Size": "Compact",
                "5. Time Zone": "US/Eastern"
            },
            "Time Series (Daily)": {
                "2018-11-28": {
                    "1. open": "64.6800",
                    "2. high": "64.6800",
                    "3. low": "64.6800",
                    "4. close": "64.6800",
                    "5. volume": "0"
                },
                "2018-11-27": ...
        """
        data_path = script_relpath("../../data/finance")
        cache = FileCache(data_path)

        datatype = datatype or "60min"
        outputsize = "compact" if compact else "full"

        # API fetch request
        def fetch():
            params = {
                "apikey": API_KEY,
                "symbol": symbol,
                "function": function,
                "outputsize": outputsize,
                "datatype": datatype,
            }
            query = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"{HOST}/query?{query}"

            r = requests.get(url)
            r.raise_for_status()

            if (r.status_code == 200):
                return r.json()
            raise Exception(f"Failed to fetch stock data ({symbol}): [{r.status_code}] {r.text}")

        # get raw data from cache or API
        if function == "TIME_SERIES_INTRADAY":
            function = f"{function}_{datatype}"
        cache_path = f"{symbol}/{function}/{outputsize}"
        self.raw_data = cache.block(cache_path, fetch, expires=expires)

        # extract processed data
        self.metadata = self.raw_data["Meta Data"]

    def _extract(self, series):
        points = self.raw_data["Time Series (Daily)"]
        dates = list(points.keys())
        dates.sort()
        values = np.float32([points[date][series] for date in dates])
        dates = [np.datetime64(date) for date in dates]
        return dates, values

    def _smooth(self, data, smoothing=0.6):
        size = len(data)
        last = size > 0 if data[0] else None
        smoothed = [0] * size
        for i, d in enumerate(data):
            if last is None:
                smoothed[i] = d
            else:
                smoothed[i] = last * smoothing + (1 - smoothing) * d
            last = smoothed[i]
        return smoothed


def _plot(ax, x, y, ylabel=None):
    ax.set_title("Price")
    kwargs = {"color": "blue", "alpha": 1, "label": "Close"}
    ax.plot(x, y, **kwargs)
    ax.legend(loc='lower left', fontsize="small")

    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    # datemin = np.datetime64(r.date[0], 'Y')
    # datemax = np.datetime64(r.date[-1], 'Y') + nax.timedelta64(1, 'Y')
    # ax.set_xlim(datemin, datemax)

    def price(x):
        return '$%1.2f' % x
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.format_ydata = price
    ax.grid(True)
    fig.autofmt_xdate()

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # ax.yaxis.label.set_color(SUCCESSES_COLOR)
    # ax.tick_params(axis='y', colors=SUCCESSES_COLOR)


data = StockData("CMNWX", compact=False, expires=3600)
# print(json.dumps(r, indent=4))

print("Plotting historical comparison...")
fig, plots = plt.subplots(1, 2, figsize=(20, 10))

dates, values = data._extract("4. close")
_plot(plots[0], dates, values, "Close")

plt.tight_layout()
plt.show()
