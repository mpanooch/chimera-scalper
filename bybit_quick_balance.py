#!/usr/bin/env python3
import argparse
import json
import os
import time
import hmac
import hashlib
import requests

CONFIG_FILE = "bybit_config.json"

BASE_URLS = {
    "live": "https://api.bybit.com",
    "testnet": "https://api-testnet.bybit.com",
    "demo": "https://api.bybit.com",  # Demo runs on main domain, different account
}

def load_config(env: str):
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"{CONFIG_FILE} not found")
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    if env not in config:
        raise KeyError(f"No config for env '{env}' in {CONFIG_FILE}")
    return config[env]

def sign_request(api_secret, params):
    # Sign parameters (Bybit V5)
    sorted_params = sorted(params.items())
    qs = "&".join([f"{k}={v}" for k, v in sorted_params])
    return hmac.new(api_secret.encode("utf-8"), qs.encode("utf-8"), hashlib.sha256).hexdigest()

def get_balance(env, config):
    base_url = BASE_URLS[env]
    endpoint = "/v5/account/wallet-balance"

    ts = str(int(time.time() * 1000))
    params = {
        "accountType": "UNIFIED",  # demo + live unified accounts
        "api_key": config["api_key"],
        "timestamp": ts,
        "recv_window": "5000",
    }

    sig = sign_request(config["api_secret"], params)
    params["sign"] = sig

    resp = requests.get(base_url + endpoint, params=params)
    return resp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["live", "testnet", "demo"], default="testnet")
    args = parser.parse_args()

    config = load_config(args.env)
    resp = get_balance(args.env, config)

    print(f"HTTP {resp.status_code}")
    try:
        print(resp.json())
    except Exception:
        print(resp.text)

if __name__ == "__main__":
    main()
