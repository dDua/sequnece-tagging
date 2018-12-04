from utils import *
from gurobipy import *
import pyOpt
import numpy as np

label_count = 6
labels = ["O", "I-PER", "I-MISC", "B-MISC", "I-LOC", "B-LOC"]
pi = [0.167]*label_count

def create_objective_fn(model, sentence):
    transition = [[None]*label_count for i in range(label_count)]
    emission = {}
    alpha = [[None]*label_count for i in range(len(sentence))]
    beta = [[None]*label_count for i in range(len(sentence))]
    gamma = [[None]*label_count for i in range(len(sentence))]

    for i in range(label_count):
        for j in range(label_count):
            transition[i][j] = model.addVar(name=labels[i]+"_"+labels[j])

    for i in range(label_count):
        for j in range(len(sentence)):
            word = sentence[j]
            emission[word+"_"+labels[i]] = model.addVar(name=word+"_"+labels[i])
            
    for i in range(label_count):
        alpha[0][i] = pi[i] * emission[sentence[0]+"_"+labels[i]]

    for t in range(len(sentence)):
        for j in range(label_count):
            expr_tplus1_j = []
            for i in range(label_count):
                expr_tplus1_j.append(alpha[t][i] * transition[i][j])
            alpha[t+1][j] = quicksum(expr_tplus1_j) * emission[sentence[t+1]+"_"+labels[j]]
            

    for t in range(len(sentence)):
        for j in range(label_count):
            expr_tplus1_j = []
            for i in range(label_count):
                expr_tplus1_j.append(beta[t+1][j] * transition[i][j])
            beta[t][i] = quicksum(expr_tplus1_j)
            beta[t+1][j] = beta[t+1][j] * emission[sentence[t+1]+"_"+labels[j]]

    for i in range(len(sentence)):
        for j in range(label_count):
            gamma[i][j] = alpha[i][j] * beta[i][j]
    
    for t in range(len(sentence)):
        gamma[:,t] /= np.sum(gamma[:,t])
    
    
    

if __name__ == '__main__':
    sentences = create_dataset("./data/files/train")
    model = Model("tagger")
    objective_terms = []
    for sentence in sentences:
        objective_terms = create_objective_fn(model, sentence)
        cost = quicksum(objective_terms)
        model.setObjective(cost, GRB.MAXIMIZE)
        add_constraint_1(model, sentence)
        model.optimize()
        for var in model.getVars():
            print(var)
        break