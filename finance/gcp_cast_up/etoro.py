import http.client, urllib.request, urllib.parse, urllib.error, base64

headers = {
    # Request headers
    'Ocp-Apim-Subscription-Key': '{subscription key}',
}

params = urllib.parse.urlencode({
    # Request parameters
    'Prefix': '{string}',
    'InstrumentCount': '{number}',
    'UserCount': '{number}',
})

try:
    conn = http.client.HTTPSConnection('api.etoro.com')
    conn.request("GET", "/System/V1/AutoComplete?%s" % params, "{body}", headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))