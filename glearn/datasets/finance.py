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

        Example API Response:
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
        self.y = {}
        for k, v in self.raw_data.items():
            if k == "Meta Data":
                self.metadata = {vk.split(" ")[-1]: vv for vk, vv in v.items()}
            else:
                dates = list(v.keys())
                dates.sort()
                size = len(dates)
                for i, date in enumerate(dates):
                    for vk, vv in v[date].items():
                        label = vk.split(" ")[-1]
                        if label not in self.y:
                            self.y[label] = np.zeros(size)
                        self.y[label][i] = np.float32(vv)
                self.x = [np.datetime64(date) for date in dates]


def _smooth(data, smoothing=0.6):
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


def _plot(ax, x, ys):
    for ylabel, y in ys.items():
        ax.plot(x, y, label=ylabel)

    ax.set_title("Price")
    ax.legend(loc='upper left', fontsize="small")

    ax.set_xlabel("Date")
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
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

    # if ylabel is not None:
    #     ax.set_ylabel(ylabel)
    # ax.yaxis.label.set_color(SUCCESSES_COLOR)
    # ax.tick_params(axis='y', colors=SUCCESSES_COLOR)


if __name__ == "__main__":
    data = StockData("CMNWX", compact=False, expires=None)  # 3600)

    print("Plotting historical comparison...")
    fig, plots = plt.subplots(1, 2, figsize=(20, 10))

    _plot(plots[0], data.x, data.y)

    plt.tight_layout()
    plt.show()
