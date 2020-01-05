from app import db
from app.models import Algo

''' 
This file will be used on the prod server to fill the db with correct 
data without having to run the build_algos script on the small 
ec2 instance (where memory issues with training could cause issues).
'''

algo_data = {
    'SGD': {
        'title': 'SGDClassifier',
        'description': 'Typically a binary classifier but can use OvA to apply to multiclass classificaiton problems.',
        'type': 'Gradient Descent'
    },
    'RFC': {
        'title': 'RandomForestClassifier',
        'description': 'Ensemble ML algorithm where a set of descision trees are used to "vote" on which class an image is',
        'type': 'Ensemble'
    }

}

print("Adding SGD")
SGD = Algo.query.filter_by(title=algo_data['SGD']['title']).first()
if SGD is None:
    SGD = Algo(title=algo_data['SGD']['title'], 
            description=algo_data['SGD']['description'],
            type=algo_data['SGD']['type'])
    db.session.add(SGD)
else:
    SGD.description = algo_data['SGD']['description']
    SGD.type = algo_data['SGD']['type']
db.session.commit()

print("Adding RFC")
RFC = Algo.query.filter_by(title=algo_data['RFC']['title']).first()
if RFC is None:
    RFC = Algo(title=algo_data['RFC']['title'], 
            description=algo_data['RFC']['description'],
            type=algo_data['RFC']['type'])
    db.session.add(RFC)
else:
    RFC.description = algo_data['RFC']['description']
    RFC.type = algo_data['RFC']['type']
db.session.commit()

print("All Algos:")
algos = Algo.query.all()
for algo in algos:
    print(algo.to_json())