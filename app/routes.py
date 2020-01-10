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
    return jsonify({'prediction': prediction})

@api.route('/predict', methods=['POST'])
def predict():
    params = request.json 
    algo = Algo.query.filter_by(
        optimizer=params['optimizer'], 
        layers=params['layers']).first()
    if algo is None:
        return jsonify({'error': '{}, {} is not a valid combo.'.format(
            params['optimizer'],
            params['layers']
        )})
    prediction = app.util.make_prediction(algo, params)
    result = algo.to_json()
    result['prediction'] = prediction
    return jsonify(result)

# might be able to remove this route and use one above
# new react app will only request a single prediction at a time
@api.route('/predictions', methods=['POST'])
def predictions():
    params = request.json 
    print(params)
    predictions = []
    for algo in list(params['algos']):
        curr_algo = Algo.query.filter_by(id=algo).first()
        if curr_algo is None:
            continue
        prediction = app.util.make_prediction(curr_algo, params)
        curr_result = curr_algo.to_json()
        curr_result['prediction'] = prediction
        predictions.append(curr_result)
    return jsonify({'predictions': predictions})