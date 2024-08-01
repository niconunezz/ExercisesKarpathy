import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# the main difference between neg log likelihood (NLL) and cross entropy (CE) is that
# NLL expects the input to be log probabilities, while CE expects just raw logits

def generate_data(type='train'):
    with open('names.txt', 'r') as f:
        names = f.read().splitlines()
        names = ('.' + '.'.join(names))
        chars=  [c for c in names]
        xs=  [(chars[i],chars[i+1]) for i in range(len(chars)-2)]
        xvocab = sorted(set(xs))
        xvocab_size = len(xvocab)
        xchar_to_idx = {ch: i for i, ch in enumerate(xvocab)}
        xidx_to_char = {i: ch for i, ch in enumerate(xvocab)}
        if type == 'train':
            xs = xs[:int(len(xs)*0.8)]
        elif type == 'test':
            xs = xs[int(len(xs)*0.8):int(len(xs)*0.9)]
        elif type == 'val':
            xs = xs[int(len(xs)*0.9):]

        
        
        yvocab = sorted(set(names))
        yvocab_size = len(yvocab)
        ychar_to_idx = {char: idx for idx, char in enumerate(yvocab)}
        yidx_to_char = {idx: char for char, idx in ychar_to_idx.items()}
        ys = [i for i in names[2:]]
        if type == 'train':
            ys = ys[:int(len(ys)*0.8)]
        elif type == 'test':
            ys = ys[int(len(ys)*0.8):int(len(ys)*0.9)]
        elif type == 'val':
            ys = ys[int(len(ys)*0.9):]


        xdata= torch.tensor([xchar_to_idx[i] for i in xs])
        onehotx = F.one_hot(xdata, xvocab_size).float()
        ydata = torch.tensor([ychar_to_idx[i] for i in ys])
        onehoty = F.one_hot(ydata, yvocab_size).float()
    
    
    return onehotx, onehoty, xdata, ydata, yvocab, yvocab_size, xvocab, xvocab_size, xchar_to_idx, yidx_to_char

def train(onehotx, onehoty, ydata, xvocab_size, yvocab_size, train_steps=15):
    W = torch.randn(xvocab_size, yvocab_size, requires_grad=True)
    num = onehotx.size(0)
    for step in range(train_steps):
        logits = onehotx @ W # (num, xvocab_size) @ (xvocab_size, yvocab_size) = (num, yvocab_size)
        
        counts = logits.exp()
        probs = counts / counts.sum(dim=1, keepdim=True) # (num, yvocab_size)
        logprobs = probs.log()
        loss = -logprobs[torch.arange(num), ydata].mean()
        CE = nn.CrossEntropyLoss()
        loss2 = CE(logits, ydata)

        
        



        if step % 100 == 0:
            print(f'step {step}, loss {loss.item()}')
        
        W.grad = None
        loss.backward()
        
        W.data -= 50 * W.grad

    return W
import random
def generate(W, xchar_to_idx, yidx_to_char, xvocab_size,length=13):

    with torch.no_grad():
        seeds = [('a','n'),('a','l'),('a','r'),('a','b'),('a','c'),('a','d'),('a','e'),('a','f'),('a','g'),('a','h'),('a','i'),('a','j'),('a','k'),('a','m'),('a','o'),('a','p'),('a','q'),('a','s'),('a','t'),('a','u'),('a','v'),('a','w'),('a','x'),('a','z'),('e','n'),('e','l'),('e','r'),('e','s'),('e','m'),('e','t'),('e','d'),('e','c'),('i','n'),('i','l'),('i','r'),('i','s'),('i','m'),('i','d'),('i','c'),('i','t'),('o','n'),('o','l'),('o','r'),('o','s'),('o','m'),('o','t'),('o','d'),('o','c'),('u','n'),('u','l'),('u','r'),('u','s'),('u','m'),('u','t'),('u','d'),('u','c'),('b','a'),('b','e'),('b','i'),('b','o'),('b','u'),('b','r'),('b','l'),('c','a'),('c','e'),('c','i'),('c','o'),('c','u'),('c','r'),('c','l'),('d','a'),('d','e'),('d','i'),('d','o'),('d','u'),('d','r'),('f','a'),('f','e'),('f','i'),('f','o'),('f','u'),('f','r'),('f','l'),('g','a'),('g','e'),('g','i'),('g','o'),('g','u'),('g','r'),('g','l'),('h','a'),('h','e'),('h','i'),('h','o'),('h','u'),('j','a'),('j','e'),('j','i'),('j','o'),('j','u'),('k','a'),('k','e'),('k','i'),('k','o'),('k','u'),('l','a'),('l','e'),('l','i'),('l','o'),('l','u'),('m','a'),('m','e'),('m','i'),('m','o'),('m','u'),('n','a'),('n','e'),('n','i'),('n','o'),('n','u'),('p','a'),('p','e'),('p','i'),('p','o'),('p','u'),('p','r'),('p','l'),('r','a'),('r','e'),('r','i'),('r','o'),('r','u'),('s','a'),('s','e'),('s','i'),('s','o'),('s','u'),('t','a'),('t','e'),('t','i'),('t','o'),('t','u'),('t','r'),('v','a'),('v','e'),('v','i'),('v','o'),('v','u'),('w','a'),('w','e'),('w','i'),('w','o'),('w','u'),('x','a'),('x','e'),('x','i'),('x','o'),('x','u'),('y','a'),('y','e'),('y','i'),('y','o'),('y','u'),('z','a'),('z','e'),('z','i'),('z','o'),('z','u')]
        seed = random.choice(seeds)
        string = seed[0]+seed[1]
        
        idx = xchar_to_idx[seed]
        for i in range(length):
            onehotx = F.one_hot(torch.tensor([idx]), xvocab_size).float()
            logits = onehotx @ W
            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdim=True) # (1, yvocab_size)
        
            
            sidx = torch.multinomial(probs,1)
            # sidx = torch.argmax(probs)
            
            tok = yidx_to_char[sidx.item()]
            
            
            nstring = ((string[-1],tok))
            if tok == '.':
                break
            else:
                string += tok
                print(f"string: {string}")
            
            idx = xchar_to_idx.get(nstring,0)
            if idx == 0:
                break
    return string

def test(W, type = 'test'):
    onehotx, onehoty, xdata, ydata, yvocab, yvocab_size, xvocab, xvocab_size, xchar_to_idx, yidx_to_char = generate_data(type=type)

    logits = onehotx @ W
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)
    loss = -probs[torch.arange(onehotx.size(0)), ydata].log().mean()
    return loss

def main():
    #
    onehotx, onehoty, xdata, ydata, yvocab, yvocab_size, xvocab, xvocab_size, xchar_to_idx, yidx_to_char= generate_data(type='train')
    
    print(f'x vocab_size {xvocab_size}') 
    print(f'y vocab_size {yvocab_size}')
    # now the prob table(W) is 574 x 55
    W = train(onehotx, onehoty, ydata, xvocab_size, yvocab_size)

    for i in range(5):
        string = generate(W, xchar_to_idx, yidx_to_char, xvocab_size)
        print(string)
        print("=================================================")
    
    print("=================================================")
    print("Test Loss")
    print(test(W, type='test'))
    print("=================================================")
    print("Val Loss")
    print(test(W, type='val'))


if __name__ == '__main__':
    main()