import requests
import json
import pandas as pd
from finance.alphavantage import config


def read_json(symbol):
  payload = {
    'function': 'TIME_SERIES_DAILY_ADJUSTED',
    'symbol': symbol,
    'outputsize': 'full',
    'apikey': config.ALPHA_API_KEY,
    # AA api default data type is json, another option is csv
    'datatype': 'json'
  }
  r = requests.get('https://www.alphavantage.co/query?', params=payload)
  data = json.loads(r.text)['Time Series (Daily)']
  #with open(f'./data/data_{symbol}.json', 'w') as outfile:
  #json.dump(data, outfile)

  #df = pd.read_json(f'./data/data_{symbol}.json')
  #df.to_csv(f'./data/data_{symbol}.csv', index=None)

  return data

if __name__ == '__main__':
  store_json('ROKU')
  store_json('AAPL')