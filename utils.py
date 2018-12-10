def create_dataset(path):
    sentences = []
    for tokens in open(path).read().split("\n\n"):
        if tokens.strip():
            token_list = tokens.split("\n")        
            sentences.append([tok.split()[0] for tok in token_list])
    return sentences

def create_vocabulary(path, train_vocab = {}, train_label_vocab = {}):
    """
    Create test data by providing train_vocab
    """
    vocab = dict(train_vocab)
    label_vocab = train_label_vocab
    sentences = []
    labels = []
    dataset_limit = None
    train_vocab_size = len(train_vocab)
    i = 0
    for tokens in open(path).read().split("\n\n"):
#        print(tokens)
#        print("*****************")
        sentence = []
        label = []
        if tokens.strip():
            token_list = tokens.split("\n")
            for tok in token_list:
                words = tok.split(" ")
                if words[0] not in vocab:
                    # Give a new vocab number if the word is not present in 
                    # training data
                    if train_vocab:
                        continue
                    vocab[words[0]] = train_vocab_size if train_vocab else len(vocab)
                sentence.append(vocab[words[0]])
                if words[-1] not in label_vocab:
                    if words[-1] == '':
                        print(tokens)
                    if train_vocab:
                        print("####################")
                        print("Test data has new labels")
                        print("####################")
                    label_vocab[words[-1]] = len(label_vocab)
                label.append(label_vocab[words[-1]])
            if sentence:
                sentences.append(sentence)
                labels.append(label)
        i += 1
        if dataset_limit and i == dataset_limit:
            break
    return sentences, labels, vocab, label_vocab
