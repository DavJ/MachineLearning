import sqlite3

class db_client():

    conn = sqlite3.connect('stocks.db')

    def __init__(self):
        self.create_table()

    def create_table(self):
       try:
           self.conn.execute('''CREATE TABLE stocks
                             (date TEXT NOT NULL, 
                              future_date TEXT NOT NULL,
                              symbol TEXT NOT NULL,
                              PRIMARY KEY (date, future_date, symbol),
                              price REAL,
                              future_price_predict REAL,
                              future_price REAL)''')
       except:
           print('cannot create table')

    def insert_stock(self, stock):
        insert_sql = '''
        INSERT INTO stocks (date, future_date, symbol , price, future_price_predict, future_price, json_data)
        VALUES (?, ?, ?, ?, ?, ?) ;
        '''
        cur = self.conn.cursor()
        cur.execute(insert_sql, stock)
        self.conn.commit()




