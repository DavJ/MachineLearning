import requests
import json
from config import ALPHA_API_KEY, SYMBOLS, DB_FILE
from datetime import datetime
import aiohttp
import asyncio
from aiodbclient import insert_stock
from logging import getLogger

logger = getLogger('alphavantage')

def data_json(symbol):
  payload = {
    'function': 'TIME_SERIES_DAILY_ADJUSTED',
    'symbol': symbol,
    'outputsize': 'full',
    'apikey': ALPHA_API_KEY,
    # AA api default data type is json, another option is csv
    'datatype': 'json'
  }

  r = requests.get('https://www.alphavantage.co/query?', params=PAYLOAD)
  data = json.loads(r.text)['Time Series (Daily)']

  return data

def data_json_dumps(symbol):
  return json.dumps(data_json(symbol))


def today_data(symbol):
  data = data_json(symbol)
  if today() in data:
    return data

def today():
    return datetime.now().date().isoformat()

def store_data():

    async def get_data(symbol):
        async with aiohttp.ClientSession() as session:
            payload ={'function': 'TIME_SERIES_DAILY_ADJUSTED',
                      'symbol': symbol,
                      #'interval': '15min',
                      'outputsize': 'full',
                      'apikey': ALPHA_API_KEY,
                      'datatype': 'json'
            }
            async with session.get('https://www.alphavantage.co/query?',
                                   params=payload) as r:
                response = await r.text()

                try:
                    data = json.loads(response)['Time Series (Daily)']
                    #if today() in data:
                    #close_price = data[today()]['4. close']
                    close_price = data['2020-11-13']['4. close']
                    stock = dict(date=today(), future_date=today(), symbol=symbol, price=close_price,
                                 future_price_predict=None, future_price=close_price, json_data=response)
                    await insert_stock(**stock)
                except:
                    logger.warning(f'cannot parse or store response for symbol {symbol}')

    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(get_data(symbol)) for symbol in SYMBOLS]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

if __name__ == '__main__':
    # data_json('ROKU')
    store_data()
