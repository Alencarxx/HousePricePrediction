from flask import Flask, request, jsonify
import pickle
import numpy as np

# Inicializar o Flask
app = Flask(__name__)

# Carregar o modelo e os escaladores
MODEL_PATH = "./model/house_price_model.pkl"
with open(MODEL_PATH, 'rb') as file:
    model, scaler_X, scaler_y = pickle.load(file)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Obter parâmetros da requisição
        bedrooms = float(request.args.get('bedrooms'))
        bathrooms = float(request.args.get('bathrooms'))
        sqft_living = float(request.args.get('sqft_living'))
        sqft_lot = float(request.args.get('sqft_lot'))
        floors = float(request.args.get('floors'))
        sqft_above = float(request.args.get('sqft_above'))
        sqft_basement = float(request.args.get('sqft_basement'))
        
        # Criar o vetor de entrada
        features = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement]])
        features_scaled = scaler_X.transform(features)
        
        # Fazer previsão
        prediction_scaled = model.predict(features_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled)[0][0]
        
        # Retornar resultado
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True)
