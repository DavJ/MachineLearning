import aiosqlite
from config import DB_FILE

def create_table(self):
       try:
           with aiosqlite.connect(DB_FILE) as conn:
                conn.execute('''CREATE TABLE stocks
                             (date TEXT NOT NULL, 
                              future_date TEXT NOT NULL,
                              symbol TEXT NOT NULL,
                              price REAL,
                              future_price_predict REAL,
                              future_price REAL,
                              PRIMARY KEY (date, future_date, symbol)
                              );''')
       except:
           print('cannot create table')

async def insert_stock(self, date=None, future_date=None, symbol=None, price=None, future_price_predict=None, future_price=None, json_data=None):
        insert_sql = '''
        INSERT INTO stocks (date, future_date, symbol , price, future_price_predict, future_price, json_data)
        VALUES (?, ?, ?, ?, ?, ?) ;
        '''
        async with aiosqlite.connect(DB_FILE) as conn:
            stock = (date, future_date, symbol , price, future_price_predict, future_price, json_data)
            await conn.execute(insert_sql, stock)
            await conn.commit()

if __name__ == '__main__':
    create_table()



