from utils import *
import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
from sklearn.metrics import log_loss, accuracy_score
import pickle
import string
import random

#labels = ["O", "I-PER", "I-MISC", "B-MISC", "I-LOC", "B-LOC"]

def create_objective_fn(parameters, iteration_number=1, inference_mode = False):
    
    total_error = 0
    y_preds = []
    for sentence_num, sentence in enumerate(sentences):
        label = labels[sentence_num]
        #prior probabilities
        pi = parameters[:label_count]
        
        # a_ij in the paper
        # probabilities hidden_size x hidden_size = 6 x 6
        transition = parameters[label_count:label_count*(label_count+1)].reshape(label_count, label_count)
        # b_i in the paper
        # probabilities vocab_size x hidden_size = 9 x 6
        emission = parameters[label_count*(label_count+1):label_count*(label_count+1+vocab_size)].reshape(vocab_size, label_count)
        lagrangian_params = parameters[label_count*(label_count+1+vocab_size):]
    
        """
        dimensions: sent_len x hidden_size
        alpha[i][j] = P(sentence till i, state_at_time_i = j | model)
        beta[i][j] = P(sentence after i | state_at_time_i = j, model)
        gamma[i][j] = P(state_at_time_i = j | sentence, model)
        """
        alpha = [[0]*label_count for i in range(len(sentence))]
        beta = [[0]*label_count for i in range(len(sentence))]
        global gamma
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
        
        eps = 1e-15
        gamma = np.clip(gamma, eps, 1-eps)
        gamma = gamma/np.sum(gamma, axis = 1)[:,np.newaxis]
        
        error = 0
        # Log loss
        for i in range(len(sentence)):
            for j in range(label_count):
                p = gamma[i][label[i]]
                if j == label[i]:
                    error -= np.log(p)
    
    #    print("NLL Loss:", error)
    #    print("Expected: ", log_loss(label, gamma))
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
        total_error = total_error + error
        ########### INFERENCE MODE ##########
        if inference_mode:
            y_pred = np.argmax(np.array(gamma),1)
            y_preds.append(y_pred)
    print("iter#", iteration_number, sentence_num, total_error)
    return y_preds if inference_mode else total_error


def callback(x, i, g):
    if i%100 == 0 and i>0:
        print("iter#", i, create_objective_fn(x))


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def print_params(parameters, label_count, vocab_size):
    pi = parameters[:label_count]
    transition = parameters[label_count:label_count*(label_count+1)].reshape(label_count, label_count)
    emission = parameters[label_count*(label_count+1):label_count*(label_count+1+vocab_size)].reshape(vocab_size, label_count)
    lagrangian_params = parameters[label_count*(label_count+1+vocab_size):]
    print("pi:", pi)
    print("transition:")
#    matprint(pi)
    matprint(transition)
#    print("emission:")
#    matprint(emission)
#    print("lagrangian:", lagrangian_params)
#    matprint(lagrangian_params)

def accuracy_stats(accs):
    hist = {}
    for l, a in accs:
        temp = hist.get(l, [])
        temp.append(a)
        hist[l] = temp
    print(hist)
    for l, a_s in hist.items():
        print(l, np.median(a_s))

if __name__ == '__main__':
    global sentences, labels
    sentences,labels, vocab, label_vocab = create_vocabulary("./data/files/train")
    sentences = np.array(sentences)
    labels = np.array(labels)
    global label_count
    label_count = len(label_vocab)
    global vocab_size
    vocab_size = len(vocab)
    print("size: ", len(vocab))
    init_pi = [1/label_count]*label_count
    init_pi = np.array(init_pi).reshape(label_count)

    init_trans = [[1/label_count]*label_count for i in range(label_count)]
    init_trans = np.array(init_trans).reshape(label_count*label_count)
    
    init_em = [[1/vocab_size]*label_count for i in range(vocab_size)]
    init_em = np.array(init_em).reshape(vocab_size*label_count)

    init_params = np.concatenate((init_pi, init_trans, init_em))
    
    # random initialization
#    init_params = np.random.uniform(0,1,(1 + len(vocab) + label_count)*label_count)
    
    lagrangian_params = np.random.uniform(0,1,(2 + len(vocab) + label_count)*label_count)
    init_full_params = np.concatenate((init_params, lagrangian_params))
        
    objective_grad = grad(create_objective_fn)
    
    final_params = init_full_params
    final_params = pickle.load(open('results/final_params_100_100iters.pkl', 'rb'))

#    final_params = adam(objective_grad, final_params, step_size=0.01, callback=callback,
#                    num_iters=100)
    pkl_name = 'results/final_params_' + ''.join(random.choice(string.ascii_letters) for _ in range(6)) + '.pkl'
    pickle.dump(final_params, open(pkl_name, 'wb'))

    print("********End of training *******")
#    print("optimal params")
#    print_params(final_params, label_count, vocab_size)
#    print("---------------------")
#    print("Total error: ", create_objective_fn(final_params))

    print("****** INFERENCE on train data *******")
    train_predictions = create_objective_fn(final_params, inference_mode=True)
    train_accs = []
    for i, (pred, true) in enumerate(zip(train_predictions, labels)):
        acc = accuracy_score(true, pred)
        train_accs.append((len(pred), acc))
#        print(i, acc)
#        print("predi:", pred)
#        print("truth:", np.array(true))
    print("*** Training accuracy ***")
    print(train_accs)
    
    print("****** INFERENCE on test data *******")
    sentences, labels, vocab, label_vocab = create_vocabulary("./data/files/test", 
                                                             train_vocab = vocab,
                                                             train_label_vocab = label_vocab)
    sentences = np.array(sentences)
    labels = np.array(labels)
    print("size: ", len(vocab))
    test_predictions = create_objective_fn(final_params, inference_mode=True)
    test_accs = []
    for i, (pred, true) in enumerate(zip(test_predictions, labels)):
        print(pred, true)
        acc = accuracy_score(true, pred)
        test_accs.append((len(pred), acc))
#        print(i, acc)
#        print("predi:", pred)
#        print("truth:", np.array(true))
    print("*** Test accuracy ***")
    print(test_accs)
    



#Stochastic or Batch gradient descent?
#what about constraints
# is this even relevant to opti?
# why can't we use likelihood as obj function?
# no. of hidden states == label_count?