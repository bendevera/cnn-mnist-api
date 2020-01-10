''' 
This script is basically to run on prod servers
where you are not able to run the build_algos.py
script to build models, save models and then fill 
db with model data
'''
from app import db
from app.models import Algo 


algo_data = {
    'optimizers': ['rmsprop', 'adam'],
    'layers': [2, 3],
    'description': 'CNNs use a "frame"/"slide" to deduce paterns in an image or piece of data.',
    'type': 'Neural Network'
}

def add_to_db(optimizer, layers_num):
    model = optimizer + str(layers_num)
    print("Adding {} to db".format(model))
    curr = Algo.query.filter_by(title=model).first()
    if curr is None:
        curr= Algo(title=model, 
                description=algo_data['description'],
                type=algo_data['type'],
                optimizer=optimizer,
                layers=layers_num)
        db.session.add(curr)
    else:
        curr.description = algo_data['description']
        curr.type = algo_data['type']
        curr.optimizer = optimizer
        curr.layers = layers_num
    db.session.commit()

for optimizer in algo_data['optimizers']:
        for layer in algo_data['layers']:
            add_to_db(optimizer, layer)