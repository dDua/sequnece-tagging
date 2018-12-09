def create_dataset(path):
    sentences = []
    for tokens in open(path).read().split("\n\n"):
        if tokens.strip():
            token_list = tokens.split("\n")        
            sentences.append([tok.split()[0] for tok in token_list])
    return sentences

def create_vocabulary(path):
    vocab = {}
    label_vocab = {}
    sentences = []
    labels = []
    dataset_limit = 100
    i = 0
    for tokens in open(path).read().split("\n\n"):
        print(tokens)
        print("*****************")
        sentence = []
        label = []
        if tokens.strip():
            token_list = tokens.split("\n")
            for tok in token_list:
                words = tok.split(" ")
                if words[0] not in vocab:
                    vocab[words[0]] = len(vocab)
                sentence.append(vocab[words[0]])
                if words[-1] not in label_vocab:
                    label_vocab[words[-1]] = len(label_vocab)
                label.append(label_vocab[words[-1]])
            sentences.append(sentence)
            labels.append(label)
        i += 1
        if i == dataset_limit:
            break
    return sentences, labels, vocab, label_vocab
