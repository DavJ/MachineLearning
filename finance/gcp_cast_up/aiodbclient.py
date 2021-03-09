import aiosqlite
import sqlite3
import json
from config import DB_FILE

def create_table():
       try:
           with sqlite3.connect(DB_FILE) as conn:
                conn.execute('''CREATE TABLE stocks
                             (date TEXT NOT NULL, 
                              future_date TEXT NOT NULL,
                              symbol TEXT NOT NULL,
                              price REAL,
                              future_price_predict REAL,
                              future_price REAL,
                              json_data TEXT,
                              PRIMARY KEY (date, future_date, symbol)
                              );''')
       except:
           print('cannot create table')

async def aio_insert_stock(date=None, future_date=None, symbol=None, price=None,
                           future_price_predict=None, future_price=None, json_data=None):
        insert_sql = '''
        INSERT INTO stocks (date, future_date, symbol, price, future_price_predict, future_price, json_data)
        VALUES (?, ?, ?, ?, ?, ?, ?) ;
        '''
        async with aiosqlite.connect(DB_FILE) as conn:
            stock = (date, future_date, symbol, price, future_price_predict, future_price, json_data)
            await conn.execute(insert_sql, stock)
            await conn.commit()

async def aio_update_stock(date=None, symbol=None, price=None):
    update_sql = '''
        UPDATE stocks SET price=? WHERE date=? AND symbol=?;
        '''
    async with aiosqlite.connect(DB_FILE) as conn:
        stock = (price, date, symbol)
        await conn.execute(update_sql, stock)
        await conn.commit()


def update_stock_price_predicted(date=None, future_date=None, symbol=None, price=None, future_price_predict=None):
    update_sql = '''
        UPDATE stocks
        SET future_price_predict=?
        WHERE date=?
        AND future_date=?
        AND symbol=?;
        '''
    with sqlite3.connect(DB_FILE) as conn:
        stock = (future_price_predict, date, future_date, symbol)
        conn.execute(update_sql, stock)
        conn.commit()

def get_count_of_stock_records():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT count(*) FROM stocks;')
        return cursor.fetchone()[0]

def get_data_for_date_and_symbol(date, symbol):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''SELECT json_data
                          FROM stocks
                          WHERE date=?
                          AND symbol=?
                          ORDER BY future_date;''',
                       (date, symbol))
        try:
            return json.loads(cursor.fetchone()[0])
        except Exception:
            return None

if __name__ == '__main__':
    #create_table()
    data = get_data_for_date_and_symbol('2020-11-13', 'AAPL')
    pass



