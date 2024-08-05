import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# ex1: i tried the hardest i could but cannopt go under 2.4
# ex2: implemented a uniform initialization with ones on the weights and zeros on the bias
# also implemented xaiver initialization on the weights (works better than uniform)


def encoders(f):
    words = f.read().splitlines()
    names = ''.join(words)
    vocab = sorted(list(set(names)))
    vocab = ['.'] + vocab
    vocab_size = len(vocab)
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {c: i for i, c in stoi.items()}

    return vocab, vocab_size, stoi, itos, words

def process(words,stoi):
    X = []
    Y = []
    block_size = 3
    for w in words:
        context = [0]*block_size
        for c in w+ '.':
            X.append(context)
            Y.append(stoi[c])
            context = context[1:] + [stoi[c]]
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    
    return X,Y

def data():
    with open('names.txt', 'r') as f:
        vocab, vocab_size, stoi, itos, words = encoders(f)
        n1 = int(len(words)*0.8)
        n2 = int(len(words)*0.9)
        xtrain, ytrain = process(words[:n1],stoi)
        xtest, ytest = process(words[n1:n2],stoi)
        xval, yval = process(words[n2:],stoi)

        print(f"xtrain and ytrain shape: {xtrain.shape}, {ytrain.shape}")
        print(f"xtest and ytest shape: {xtest.shape}, {ytest.shape}")
        print(f"xval and yval shape: {xval.shape}, {yval.shape}")
        return xtrain, ytrain, xtest, ytest, xval, yval, vocab_size, stoi, itos

def init(vocab_size, d_model, block_size):
    C = torch.zeros((vocab_size, d_model),requires_grad=True)
    W1 = torch.empty((d_model*block_size, 800)).uniform_(-math.sqrt(6/d_model*block_size + 300), math.sqrt(6/d_model*block_size + 300))
    # W1 = torch.ones((d_model*block_size, 300))
    W1.requires_grad = True
    b1 = torch.zeros((800,), requires_grad=True)
    W2 = torch.empty((800, vocab_size)).uniform_(-math.sqrt(6/(300 + vocab_size)), math.sqrt(6/(300 + vocab_size)))
    # W2 = torch.ones((300, vocab_size))
    W2.requires_grad = True
    b2 = torch.zeros((vocab_size,),requires_grad=True)
    return [C, W1, b1, W2, b2]

def forward(x, y, params, d_model, block_size):
    C, W1, b1, W2, b2 = params
    #x.shape = (N, block_size)
    ix = torch.randint(0, x.shape[0], (64,))
    emb = C[x[ix]] # (N, block_size, d_model)
    emb = emb.view(64, block_size*d_model) # (N, block_size*d_model)
    h = torch.tanh(torch.matmul(emb, W1) + b1) # (N, 300)
    logits = torch.matmul(h, W2) + b2 # (N, vocab_size)
    loss = F.cross_entropy(logits, y[ix])
    return loss

def update(loss, params,lr=0.1):
    for p in params:
        p.grad = None
    loss.backward()
    for p in params:
        p.data -= lr * p.grad

def train(x, y, vocab_size, d_model, block_size, train_steps=25000):
    params = init(vocab_size, d_model, block_size)

    for i in range(train_steps):
        loss = forward(x, y, params, d_model, block_size)
        update(loss, params, lr=0.1)

        if i % 4000 == 0:
            print(f'step {i}, loss {loss.item()}')
    for i in range(train_steps):
        loss = forward(x, y, params, d_model, block_size)
        update(loss, params, lr=0.01)

        if i % 4000 == 0:
            print(f'step {i}, loss {loss.item()}')
    
    
    return params

def test(params,x,y,d_model,block_size):
    C, W1, b1, W2, b2 = params
    
    h = torch.tanh(torch.matmul(C[x].view(-1,d_model*block_size), W1) + b1)
    logits = torch.matmul(h, W2) + b2
    loss = F.cross_entropy(logits, y).item()
    return loss


# just works for d_model = 2
def visualize2d(C,itos):
    for i, c in itos.items():
        x, y = C[i]
        plt.text(x, y, c)
    
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    
    plt.show()




def main():
    xtrain, ytrain, xtest, ytest, xval, yval, vocab_size, stoi, itos = data()
    d_model = 100
    block_size = 3
    params = train(xtrain, ytrain, vocab_size, d_model, block_size)
    print(f"test loss: {test(params, xtest, ytest, d_model, block_size)}")
    print(f"val loss: {test(params, xval, yval, d_model, block_size)}")

    # visualize2d(params[0].detach().numpy(),itos)


if __name__ == '__main__':
    main()