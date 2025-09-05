#!/usr/bin/env python3
import requests
import json
import time
import hmac
import hashlib

def test_bybit_auth():
    # Load config
    with open('bybit_config.json') as f:
        config = json.load(f)
    
    api_key = config['demo']['api_key']
    api_secret = config['demo']['api_secret']
    
    # Test wallet balance
    timestamp = str(int(time.time() * 1000))
    params = {
        'api_key': api_key,
        'timestamp': timestamp,
        'recv_window': '5000',
        'accountType': 'UNIFIED'
    }
    
    # Generate signature exactly like Bybit expects
    sorted_params = sorted(params.items())
    query_string = '&'.join([f'{k}={v}' for k, v in sorted_params])
    signature = hmac.new(
        api_secret.encode('utf-8'), 
        query_string.encode('utf-8'), 
        hashlib.sha256
    ).hexdigest()
    params['sign'] = signature
    
    print(f"Query string: {query_string}")
    print(f"Signature: {signature}")
    
    # Make request
    response = requests.get(
        'https://api-demo.bybit.com/v5/account/wallet-balance', 
        params=params
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    return response.status_code == 200

if __name__ == "__main__":
    success = test_bybit_auth()
    print(f"Auth test: {'PASSED' if success else 'FAILED'}")
