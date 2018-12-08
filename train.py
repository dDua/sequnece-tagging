from utils import *
import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

#label_count = None
#labels = ["O", "I-PER", "I-MISC", "B-MISC", "I-LOC", "B-LOC"]
#pi = [0.167]*label_count

#sentences=None
#label=None
vocab_size = None

def constraint_check(transition, emission):
    
    num_violations = 0
    
    # sum_rows and sum_cols both have to be 1
    # constraints = 2 x hidden_size
    num_violations += (np.sum(transition, axis=0) != 1).sum()
    num_violations += (np.sum(transition, axis=1) != 1).sum()
    
    # for every hidden state sum of emitting prob of a word = 1
    # sum of all columns = 1
    # constraints = hidden_size
    num_violations += (np.sum(emission, axis=0) != 1).sum()
    
#    print("num_violations = ", num_violations)
    return num_violations
    

def create_objective_fn(parameters, iteration_number=1):
    sentence = np.array(sentences[0])
    
    #prior probabilities
    pi = parameters[:label_count]
    
    # a_ij in the paper
    # probabilities hidden_size x hidden_size = 6 x 6
    transition = parameters[label_count:label_count*(label_count+1)].reshape(label_count, label_count)
    # b_i in the paper
    # probabilities vocab_size x hidden_size = 9 x 6
    emission = parameters[label_count*(label_count+1):label_count*(label_count+1+vocab_size)].reshape(vocab_size, label_count)
#    vocab_size = emission.shape[0]
    lagrangian_params = parameters[label_count*(label_count+1+vocab_size):]

    """
    dimensions: sent_len x hidden_size
    alpha[i][j] = P(sentence till i, state_at_time_i = j | model)
    beta[i][j] = P(sentence after i | state_at_time_i = j, model)
    gamma[i][j] = P(state_at_time_i = j | sentence, model)
    """
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
    
    global gamma
    error = 0

    # Log loss
    for i in range(len(sentence)):
        for j in range(label_count):
            p = gamma[i][label[i]]
            if j == label[i]:
                error -= np.log(1-p)
            else:
                error -= np.log(p)

#    print("NLL Loss:", error)

    # One-hot encoding not working
#    true_labels = [[0]*label_count for i in range(len(sentence))]
#    for i, val in enumerate(label):
#        true_labels[i][val] = 1
#
#    predictions = [[0]*label_count for i in range(len(sentence))]
#    for i, val in enumerate(y_pred):
#        predictions[i][val] = 1

#    error = np.sum((y_pred - label)**2)

    _lambda = 0
    inequalities = []
    
    """
    Total #constraints = #probabilities + hidden_size + hidden_size
    #probabilities = (hidden_size+vocab_size) x hidden_size
    """
    
    # hidden_size x hidden_size
    for i in range(label_count):
        temp = 0
        for j in range(label_count):
            temp = temp + transition[i][j]
            # probabilities >= 0
            inequalities.append(-transition[i][j])
        # sum_rows and sum_cols both have to be <= 1
        inequalities.append(temp-1)

    # vocab_size x hidden_size
    for i in range(label_count):
        temp = 0
        for j in range(vocab_size):
            temp = temp + emission[j][i]
            # probabilities >= 0
            inequalities.append(-emission[j][i])
        # sum of all columns <= 1
        inequalities.append(temp-1)
    inequalities = np.array(inequalities)
    ineq_error = np.dot(lagrangian_params, inequalities)

    error = error + ineq_error
    return error


def callback(x, i, g):
    if i%10 == 0:
        print("iter#", i, create_objective_fn(x))


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def print_params(parameters, label_count):
    pi = parameters[:label_count]
    transition = parameters[label_count:label_count*(label_count+1)].reshape(label_count, label_count)
    emission = parameters[label_count*(label_count+1):label_count*(label_count+1+vocab_size)].reshape(vocab_size, label_count)
    lagrangian_params = parameters[label_count*(label_count+1+vocab_size):]
    print("pi:", pi)
    print("transition:")
#    matprint(pi)
    matprint(transition)
    print("emission:")
    matprint(emission)
    print("lagrangian:", lagrangian_params)
#    matprint(lagrangian_params)

if __name__ == '__main__':
    sentences,labels, vocab, label_vocab = create_vocabulary("./data/files/train")
    global label_count
    label_count = len(label_vocab)
    global vocab_size
    vocab_size = len(vocab)
    init_pi = [1/label_count]*label_count
    init_pi = np.array(init_pi).reshape(label_count)

    init_trans = [[1/label_count]*label_count for i in range(label_count)]
    init_trans = np.array(init_trans).reshape(label_count*label_count)
    
    init_em = [[1/vocab_size]*label_count for i in range(vocab_size)]
    init_em = np.array(init_em).reshape(vocab_size*label_count)

    init_params = np.concatenate((init_pi, init_trans, init_em))
    
    # random initialization
    init_params = np.random.uniform(0,1,(1 + len(vocab) + label_count)*label_count)
    
    lagrangian_params = np.random.uniform(0,1,(2 + len(vocab) + label_count)*label_count)
    
    init_full_params = np.concatenate((init_params, lagrangian_params))
    
    objective_terms = []
    sentence = np.array(sentences[0])
    label= np.array(labels[0])
    objective_grad = grad(create_objective_fn)
    final_params = adam(objective_grad, init_full_params, step_size=0.01, callback=callback,
                num_iters=100)
    
    print("********End of training *******")
    print("optimal params")
    print_params(final_params, label_count)
    print("---------------------")
    print("func value: ", create_objective_fn(final_params))
    print("gamma: ", gamma)
#    matprint(gamma)
    y_pred = np.argmax(np.array(gamma),1)
    print("y_pred: ", y_pred)
    print("truth: ", label)
    correct = 0
    for x,y in zip(y_pred, label):
        if x == y:
            correct += 1
    print("accuracy: ", correct/len(label))

#Stochastic or Batch gradient descent?
#what about constraints
# is this even relevant to opti?
# why can't we use likelihood as obj function?
# no. of hidden states == label_count?