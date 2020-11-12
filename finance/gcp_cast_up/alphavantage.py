import requests
import json
import config

def data_json(symbol):
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

  return data

def data_json_dumps(symbol):
  return json.dumps(data_json(symbol))

if __name__ == '__main__':
  data_json('ROKU')
