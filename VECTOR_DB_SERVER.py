from flask import Flask, request, jsonify
from VECTOR_DATABASE import VectorDatabase
from CARD_DATA import Card
import numpy as np


app = Flask(__name__)
vd = VectorDatabase()
vd.parse_json("AllPrintings.json",runtime=True,max_lines=2500)

#Example Request:
#(Invoke-RestMethod -Uri "http://127.0.0.1:5000/get_vector/Shunt")
@app.route('/get_vector/<string:v_id>', methods=['GET'])
def get_vector(v_id):
    vector = vd.get_vector(v_id)
    return jsonify({"vector": vector.tolist() if vector is not None else None})

#Example Request:
#(Invoke-RestMethod -Uri "http://127.0.0.1:5000/get_vector_description/Shunt")
@app.route('/get_vector_description/<string:v_id>', methods=['GET'])
def get_vector_description(v_id):
    return jsonify(vd.get_vector_description_dict(v_id=v_id))

#Example Request:
#curl http://127.0.0.1:5000/get_random_vector
@app.route('/get_random_vector', methods=['GET'])
def get_random_vector():
    random_vector = vd.get_random_vector()
    return jsonify({"id":random_vector[0], "vector": random_vector[1]})

#Example Request:
#curl http://127.0.0.1:5000/get_random_vector_description
@app.route('/get_random_vector_description', methods=['GET'])
def get_random_vector_description():
    return jsonify(vd.get_vector_description_dict(vd.get_random_vector()[0]))

#Example Request:
#(Invoke-RestMethod -Uri "http://127.0.0.1:5000/get_similar_vectors/Shunt?num_vectors=10")
@app.route('/get_similar_vectors/<string:v_id>', methods=['GET'])
def get_similar_vectors(v_id):
    try:
        num_vectors = request.args.get('num_vectors', default=5, type=int)
    except ValueError:
        return jsonify({"error": "num_vectors must be an integer"}), 400
    
    results_list = vd.get_similar_vectors(vd.get_vector(v_id=v_id), num_vectors)
    
    results = {}
    for i in range(len(results_list)):
        results[str(i)] = vd.get_vector_description_dict(results_list[i][0])

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)