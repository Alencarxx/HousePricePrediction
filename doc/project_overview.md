# House Price Prediction API

This project predicts house prices based on features such as:
- Number of bedrooms
- Number of bathrooms
- Living area (sqft)
- Lot size (sqft)
- Number of floors
- Above-ground area (sqft)
- Basement area (sqft)

## How to Use

1. Train the model:
   ```bash
   python src/train_model.py

## Start the API:

python src/api.py

## Make predictions:

curl "http://127.0.0.1:5000/predict?bedrooms=3&bathrooms=2&sqft_living=2000&sqft_lot=5000&floors=1&sqft_above=1500&sqft_basement=500"


## Dependencies

pip install -r requirements.txt

---

### 6. `doc/project_overview.md`


# Project Overview: House Price Prediction API

## 1. Description
This project provides a RESTful API for predicting house prices using a TensorFlow-trained model.

## 2. Workflow
1. Train the model with selected features.
2. Deploy the model via Flask API.
3. Predict house prices in real-time.

## 3. Technologies Used
- Python
- TensorFlow
- scikit-learn
- Flask
- Pandas

## 4. Next Steps
- Add additional features to improve accuracy.
- Deploy the API to a cloud platform.
