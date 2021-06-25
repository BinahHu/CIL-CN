import importlib

'''
file_class_pair = {'ccll':'CCLL',
                   'covnorm':'CovNorm',
                   'derpp':'DERpp',
                   'joint':'Joint',
                   'sgd':'SGD'}
'''
file_class_pair = {'sgd':'SGD',
                   'joint':'Joint',
                   'covnorm':'CovNorm'}

models = {}
for k,v in file_class_pair.items():
    mod = importlib.import_module('models.' + k)
    models[k] = getattr(mod, v)