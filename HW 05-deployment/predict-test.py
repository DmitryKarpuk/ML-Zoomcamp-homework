import requests
def predict_test():
    url = 'http://localhost:9696/predict'
    client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
    response = requests.post(url, json=client)
    result = response.json()
    print(result)

if __name__ == "__main__":
    predict_test()