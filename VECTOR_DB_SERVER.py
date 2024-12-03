from flask import Flask, request, jsonify
from VECTOR_DATABASE import VectorDatabase

app = Flask(__name__)
vd = VectorDatabase()
vd.parse_json("AllPrintings.json",8000)

#Example Request:
#curl -X GET "http://127.0.0.1:5000/get_vector/Shunt"
@app.route('/get_vector/<string:v_id>', methods=['GET'])
def get_vector(v_id):
    try:
        vector = vd.find_vector(format_id(v_id))
    except KeyError:
        return jsonify({"error": "id not found"}), 400
    
    return jsonify({"vector": vector.tolist() if vector is not None else None})

#Example Request:
#curl -X GET "http://127.0.0.1:5000/get_vector_description/Shunt"
@app.route('/get_vector_description/<string:v_id>', methods=['GET'])
def get_vector_description(v_id):
    try:
        id = vd.find_id(format_id(v_id))
    except KeyError:
        return jsonify({"error": "id not found"}), 400
    
    return jsonify(vd.get_vector_description_dict(id))

#Example Request:
#curl -X GET "http://127.0.0.1:5000/get_random_vector"
@app.route('/get_random_vector', methods=['GET'])
def get_random_vector():
    random_vector = vd.get_random_vector()
    return jsonify({"id":(random_vector[0]), "vector": random_vector[1].tolist() if random_vector[1] is not None else None})

#Example Request:
#curl -X GET "http://127.0.0.1:5000/get_random_vector_description"
@app.route('/get_random_vector_description', methods=['GET'])
def get_random_vector_description():
    return jsonify(vd.get_vector_description_dict(vd.get_random_vector()[0]))

#Example Request:
#curl -X GET "http://127.0.0.1:5000/get_similar_vectors/Shunt?num_vectors=10"
@app.route('/get_similar_vectors/<string:v_id>', methods=['GET'])
def get_similar_vectors(v_id):
    try:
        vector_id = vd.find_id(format_id(v_id))
    except KeyError:
        return jsonify({"error": "id not found"}), 400

    try:
        num_vectors = request.args.get('num_vectors', default=5, type=int)
    except ValueError:
        return jsonify({"error": "num_vectors must be an integer"}), 400
    
    
    results_list = vd.get_similar_vectors(vector_id, num_vectors)
    
    results = {}
    for i in range(len(results_list)):
        results[str(i)] = (vd.get_vector_description_dict(results_list[i][0]))

    return jsonify(results)

def format_id(v_id: str):
    transition_words: set[str] = {'of', 'the', 'in', 'on', 'at', 'to', 'for', 'and', 'but', 'or', 'nor'}
    
    words: list[str] = v_id.split(' ')
    
    capitalized_words: list[str] = [
        word.capitalize() if word.lower() not in transition_words or i == 0 else word.lower()
        for i, word in enumerate(words)
    ]
    
    return ' '.join(capitalized_words)

if __name__ == '__main__':
    app.run(debug=False)