import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import pickle

def train_model(csv_path, model_path):
    # Carregar os dados
    data = pd.read_csv(csv_path)
    
    # Selecionar características
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']
    X = data[features]
    y = data['price']
    
    # Escalar os dados
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    # Dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25, random_state=42)
    
    # Construir o modelo
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=100, activation='relu', input_shape=(len(features),)),
        tf.keras.layers.Dense(units=100, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Treinar o modelo
    model.fit(X_train, y_train, epochs=100, batch_size=50, validation_split=0.2)
    
    # Avaliação
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    
    # Salvar o modelo e os escaladores
    with open(model_path, 'wb') as file:
        pickle.dump((model, scaler_X, scaler_y), file)
    print(f"Modelo salvo em {model_path}")

if __name__ == "__main__":
    train_model("../data/kc-house-data.csv", "../src/model/house_price_model.pkl")
