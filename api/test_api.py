import subprocess
import sys
import time
import requests


server = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000"
    ]
)

time.sleep(5)

customer = {
    "call_failure": 10,
    "complains": 1,
    "subscription_length": 12,
    "charge_amount": 20,
    "seconds_of_use": 3000,
    "frequency_of_use": 30,
    "frequency_of_sms": 15,
    "distinct_called_numbers": 20,
    "age_group": 3,
    "tariff_plan": 1,
    "status": 1,
    "age": 30,
    "customer_value": 80
}

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=customer
)

print("Status code:", response.status_code)
print("Response:", response.json())

if response.status_code != 200:
    server.terminate()
    raise SystemExit(1)

server.terminate()

print("API test successful")
