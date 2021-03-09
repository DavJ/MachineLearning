import requests
import json
from time import sleep
from config import ALPHA_API_KEY, SYMBOLS, DB_FILE, CHUNK_SIZE, CHUNK_SLEEP
from datetime import datetime
import aiohttp
import asyncio
from aiodbclient import (aio_insert_stock, get_count_of_stock_records, get_data_for_date_and_symbol,
                         update_stock_price_predicted, aio_update_stock)
from logging import getLogger
from kalman import predict_stock_price
from businessdates import next_business_day

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

  r = requests.get('https://www.alphavantage.co/query?', params=payload)
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


def store_data(date=today()):

    def get_symbol_chunk():
        index = 0
        while index < len(SYMBOLS):
            yield SYMBOLS[index: index + CHUNK_SIZE]
            index += CHUNK_SIZE
            sleep(CHUNK_SLEEP)

    def download_data_for_symbols(symbols):
        loop = asyncio.new_event_loop()
        tasks = [loop.create_task(get_data(s)) for s in symbols]
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

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
                    if date in data:
                       close_price = data[date]['4. close']
                       stock = dict(date=date, future_date=next_business_day(date), symbol=symbol, price=close_price,
                                future_price_predict=None, future_price=None, json_data=response)
                       await aio_insert_stock(**stock)
                       await aio_update_stock(symbol=symbol, future_date=date, future_price=close_price)
                    else:
                       raise Exception('old data')
                except:
                    logger.warning(f'cannot parse or store response for symbol {symbol}')

    original_stock_records = get_count_of_stock_records()

    for symbol_chunk in get_symbol_chunk():
        download_data_for_symbols(symbol_chunk)

    final_stock_records = get_count_of_stock_records()

    return f'added {final_stock_records - original_stock_records} stock price records'

def predict_all_data(date=today()):
    for symbol in SYMBOLS:
      for symbol in SYMBOLS:
        data_json = get_data_for_date_and_symbol(date, symbol)
        if data_json:
            logger.debug(f'predict data for symbol {symbol}')
            predicted_price = predict_stock_price(symbol, 1, data_json)
            logger.debug(f'predicted price for symbol {symbol}: {predicted_price}')
            update_stock_price_predicted(date=date, future_date=next_business_day(date),
                                         symbol=symbol, future_price_predict=predicted_price)

if __name__ == '__main__':
    # data_json('ROKU')
    print(store_data())
    #predict_all_data('2020-11-13')
    predict_all_data()
    pass