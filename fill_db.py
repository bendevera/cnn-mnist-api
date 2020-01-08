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
    },
    'KN': {
        'title': 'KNeighborsClassifier',
        'description': 'Classifier implementing the k-nearest neighbors vote.',
        'type': 'Neighbors'
    },
    'CNN': {
        'title': 'ConvolutionalNeuralNetwork',
        'description': 'CNNs use a "frame"/"slide" to deduce paterns in an image or piece of data.',
        'type': 'Neural Network'
    }
}

def add_to_db(model):
    print("Adding {} to db".format(model))
    curr = Algo.query.filter_by(title=algo_data[model]['title']).first()
    if curr is None:
        curr= Algo(title=algo_data[model]['title'], 
                description=algo_data[model]['description'],
                type=algo_data[model]['type'])
        db.session.add(curr)
    else:
        curr.description = algo_data[model]['description']
        curr.type = algo_data[model]['type']
    db.session.commit()

add_to_db("SGD")
add_to_db("RFC")
add_to_db("KN")
add_to_db("CNN")

print("\n All Algos:")
algos = Algo.query.all()
for algo in algos:
    print(algo.to_json())