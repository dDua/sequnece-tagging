from gurobipy import *
from utils import *

root_dir = ""
label_count = 6
labels = ["O", "I-PER", "I-MISC", "B-MISC", "I-LOC", "B-LOC"]

def populate_weights_map(path):
    weight_map = {}
    with open(path) as fr:
        for line in fr:
            tokens = line.strip().split()
            features = tokens[0].split(":")
            word = features[0].replace("Wi=","")
            label = features[-1].replace("Ti=","")
            weight_map[word+"_"+label] = float(tokens[1])
    return weight_map

def create_objective_fn(model, sentence, weights):
    obj_fn = []
    for token in sentence:
        for label in labels:
            if (token+"_"+label) in weights:
                wgt = weights[token+"_"+label]
            else:
                wgt = 0.0
            x = model.addVar(vtype=GRB.BINARY, name=token+"_"+label)
            obj_fn.append(x * wgt)
    return obj_fn

def add_constraint_1(model, sentence):
    variables = model.getVars()
    count = 0
    for i in range(0, len(variables), label_count):
        label_constraint = []
        for j in range(0, label_count):
            label_constraint.append(variables[i*label_count + j])
        model.addConstr(quicksum(label_constraint) == 1, name=sentence[count])
        count+=1


if __name__ == '__main__':
    weights = populate_weights_map("./data/files/weights")
    sentences = create_dataset("./data/files/dev")
    model = Model("viterbi")
    objective_terms = []
    for sentence in sentences[1:]:
        objective_terms = create_objective_fn(model, sentence, weights)
        cost = quicksum(objective_terms)
        model.setObjective(cost, GRB.MAXIMIZE)
        add_constraint_1(model, sentence)
        model.optimize()
        for var in model.getVars():
            print(var)
        break
    