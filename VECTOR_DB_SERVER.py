from flask import Flask, request, jsonify
from VECTOR_DATABASE import VectorDatabase
from CARD_DATA import Card
import numpy as np


app = Flask(__name__)
vd = VectorDatabase()
vd.parse_json("AllPrintings.json",runtime=True,max_lines=2500)

@app.route('/add_card', methods=['POST'])
def add_card():
    card_data = request.json # Assume card data comes in JSON format
    card = Card(card_data)
    vd.add_card(card)
    return jsonify({"message": "Card added"}), 201

@app.route('/get_vector/<string:v_id>', methods=['GET'])
def get_vector(v_id):
    vector = vd.get_vector(v_id)
    return jsonify({"vector": vector.tolist() if vector is not None else None})

@app.route('/get_random_vector', methods=['GET'])
def get_random_vector():
    random_vector = vd.get_random_vector()
    return jsonify({"id":random_vector[0], "vector": random_vector[1]})

@app.route('/get_random_vector_description', methods=['GET'])
def get_random_vector_description():
    return jsonify(vd.get_vector_description_dict(vd.get_random_vector()[0]))

@app.route('/get_similar_vectors/<string:v_id>', methods=['GET'])
def get_similar_vectors():
    q_vector = request.json.get('vector')
    n_results = request.json.get('n_results', 5)
    results = vd.get_similar_vectors(np.array(q_vector), n_results)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)