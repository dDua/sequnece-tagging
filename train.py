from utils import *
import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

label_count = 6
labels = ["O", "I-PER", "I-MISC", "B-MISC", "I-LOC", "B-LOC"]
pi = [0.167]*label_count

sentence=None
label=None

def create_objective_fn(parameters):
    transition = parameters[:label_count*label_count].reshape(label_count, label_count)
    emission = parameters[label_count*label_count:].reshape(-1,label_count)
    alpha = [[0]*label_count for i in range(len(sentence))]
    beta = [[0]*label_count for i in range(len(sentence))]
    gamma = [[0]*label_count for i in range(len(sentence))]

    for i in range(label_count):
        alpha[0][i] = pi[i] * emission[sentence[0]][i]

    for i in range(label_count):
        beta[-1][i] = 1

    for t in range(1,len(sentence)):
        for i in range(label_count):
            for j in range(label_count):
                alpha[t][i] += alpha[t-1][j] * transition[i][j]
            alpha[t][i] *= emission[sentence[t]][i]
            

    for t in range(len(sentence)-2,-1,-1):
        for i in range(label_count):
            for k in range(label_count):
                beta[t][i] += beta[t+1][k] * transition[i][k] * emission[sentence[t+1]][k]

    for i in range(len(sentence)):
        for j in range(label_count):
            gamma[i][j] = alpha[i][j] * beta[i][j]
    
    for t in range(sentence.shape[0]):
        temp = 0
        for i in range(label_count):
            temp += gamma[t][i]
        for i in range(label_count):
            gamma[t][i] = gamma[t][i] / temp
    
    #y_pred = np.max(np.array(gamma),1)
    #y_pred = np.array(gamma)[:,0]

    error = np.sum((y_pred - label)**2)

    return error

        
if __name__ == '__main__':
    sentences,labels, vocab, label_vocab = create_vocabulary("./data/files/train")
    params = np.random.uniform(0,1,len(vocab)*label_count+36)
    objective_terms = []
    sentence = np.array(sentences[0])
    label= np.array(labels[0])
    #objective = create_objective_fn(params)
    objective_grad = grad(create_objective_fn)
    #pars = adam(objective_grad, params, step_size=0.1, 
    #            num_iters=1)
    #print(pars)
    init = np.ones(params.shape[0])
    print(objective_grad(init))
        