import torch
import torch.nn.functional as F


def generate_data():
    with open('names.txt', 'r') as f:
        names = f.read().splitlines()

    names = '.' + '.'.join(names)
    vocab = sorted(set(names))
    vocab_size = len(vocab)
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    xdata = torch.tensor([char_to_idx[char] for char in names[:-1]])
    ydata = torch.tensor([char_to_idx[char] for char in names[1:]])
    onehotx = F.one_hot(xdata, vocab_size).float()
    onehoty = F.one_hot(ydata, vocab_size).float()

    return onehotx, onehoty, xdata, ydata, vocab_size, char_to_idx, idx_to_char

def train(onehotx, onehoty, ydata, vocab_size, train_steps=200):
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((vocab_size, vocab_size), requires_grad=True, generator=g)
    num = onehotx.size(0)
    for step in range(train_steps):
        logits = onehotx @ W # (num, vocab_size)
        counts = logits.exp() 
        probs = counts / counts.sum(dim=1, keepdim=True)
        loss = -probs[torch.arange(num), ydata].log().mean() + 0.1 * W.pow(2).mean()
        if step % 100 == 0:
            print(f'step {step}, loss {loss.item()}')

        W.grad = None
        loss.backward()
        
        W.data += -50 * W.grad
    
    return W

def generate(W, char_to_idx, idx_to_char, vocab_size, length=50):
    with torch.no_grad():
        string = '.'
        
        idx = 2
        for i in range(length):
            onehotx = F.one_hot(torch.tensor([idx]), vocab_size).float()
            logits = onehotx @ W
            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdim=True)
            
            idx = torch.multinomial(probs,1)
            tok = idx_to_char[idx.item()]
            
            if tok == '.':
                string += '\n.'
            else:
                string += tok
            
    
    return string




def main():
    onehotx, onehoty, xdata, ydata, vocab_size, char_to_idx, idx_to_char = generate_data()
    W = train(onehotx, onehoty, ydata, vocab_size)
    string = generate(W, char_to_idx, idx_to_char, vocab_size)
    print(string)
    
if __name__ == '__main__':
    main()