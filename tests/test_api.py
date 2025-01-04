import requests

BASE_URL = "http://127.0.0.1:5000"

def test_predict():
    params = {
        'bedrooms': 3,
        'bathrooms': 2,
        'sqft_living': 2000,
        'sqft_lot': 5000,
        'floors': 1,
        'sqft_above': 1500,
        'sqft_basement': 500
    }
    response = requests.get(f"{BASE_URL}/predict", params=params)
    print(response.json())

if __name__ == "__main__":
    test_predict()