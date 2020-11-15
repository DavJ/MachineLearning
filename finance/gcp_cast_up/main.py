# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python38_app]
from flask import Flask, request
from markupsafe import escape
from kalman import predict_stock_price
from alphavantage import data_json_dumps, store_data

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)


@app.route('/')
def usage():
    """Return a friendly HTTP greeting."""
    return ("/stock-cast-up?symbol=AAPL&days=1\n"
            "/data?symbol=AAPL\n"
            )

@app.route("/stock-cast-up/")
def predict():
  symbol = request.args.get('symbol', default = 'AAPL', type = str)
  days = request.args.get('days', default = 1, type = int)
  return str(predict_stock_price(symbol, days))

@app.route("/data/")
def data():
    symbol = request.args.get('symbol', default='AAPL', type=str)
    return data_json_dumps(symbol)

@app.route("/store-data/")
def store():
    return store_data()


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python38_app]
