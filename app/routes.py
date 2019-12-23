from app import app as api
from app import db 
from flask import jsonify, request, send_from_directory
from app.models import Algo
import app.util


@api.route('/')
def hello():
    return jsonify({'message': 'Welcome to ML-Algos-API'})

@api.route('/algos')
def algos():
    algos_list = Algo.query.all()
    return jsonify({'algos': [algo.to_json() for algo in algos_list]})

@api.route('/algos/<id>')
def algo_by_id(id):
    algo = Algo.query.filter_by(id=id).first()
    return jsonify({'algo': algo.to_json()})

@api.route('/algos/<id>/predict', methods=['POST'])
def algo_predict(id):
    algo = Algo.query.filter_by(id=id).first()
    if algo is None:
        return jsonify({'error': '{} is not a valid ID.'.format(id)})
    params = request.json
    prediction = app.util.make_prediction(algo, params)
    return jsonify({'prediction': int(prediction[0])})