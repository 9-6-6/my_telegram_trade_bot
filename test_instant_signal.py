#!/usr/bin/env python3
"""Quick test to verify signal delivery is instant"""
import requests
import time

NGBOT_TOKEN = "8548359658:AAE420kDIrgpyExD8gJwi9b4kZfNsJ1nJYA"
CHAT_ID = "603932135"

def send_test():
    timestamp = time.strftime('%H:%M:%S')
    msg = f"🧪 TEST SIGNAL\n📍 Sent at: {timestamp}\n\nIf you see this, signals are working!"
    
    url = f"https://api.telegram.org/bot{NGBOT_TOKEN}/sendMessage"
    params = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "HTML"
    }
    
    print(f"📤 Sending at {timestamp}...")
    start = time.time()
    response = requests.get(url, params=params)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        data = response.json()
        msg_id = data.get('result', {}).get('message_id', 'unknown')
        print(f"✅ Delivered in {elapsed:.2f}s (msg_id: {msg_id})")
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    send_test()
