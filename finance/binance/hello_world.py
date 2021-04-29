from decimal import Decimal
from config import API_KEY, SECRET_KEY
from binance.client import Client

client = Client(API_KEY, SECRET_KEY)


def max_converts(from_symbol, amount):
    rates = client.get_all_tickers()
    conversions = [{'to_symbol': rate['symbol'].split(from_symbol)[1],
                    'value': Decimal(rate['price']) * Decimal(amount)}
                   for rate in rates if rate['symbol'].split(from_symbol)[0] == '']
    return sorted(conversions, key=lambda x: x['value'], reverse=True)


def convert(from_symbol, to_symbol, amount):
    rates = client.get_all_tickers()
    try:
       return {'to_symbol': to_symbol, 'value': [Decimal(rate['price']) * Decimal(amount)
                                              for rate in rates if rate['symbol'] == from_symbol + to_symbol][0]
              }
    except:
        return None

def print_list_dict(list_dict):
    _ = [print(item) for item in list_dict]


#print(max_converts('BNB', '2.25'))


#print_list_dict(client.get_all_tickers())


print_list_dict(sorted(max_converts('BNB', '2.25'), key=lambda x: x['to_symbol']))

print('>>>>''')
print(convert('BNB', 'BUSD', '2.25'))
print(convert('BUSD', 'BNB', '1'))

print('>>>>''')

print_list_dict(sorted(max_converts('AUD', 1618), key=lambda x: x['to_symbol']))

print('>>>>''')

print_list_dict(sorted(max_converts('BUSD', 1253), key=lambda x: x['to_symbol']))

print('>>>>''')

print('>>>>''')
print(convert('BUSD', 'ETH', '1253'))

print('>>>>')
print_list_dict(sorted(max_converts('BEUR', 30), key=lambda x: x['to_symbol']))








