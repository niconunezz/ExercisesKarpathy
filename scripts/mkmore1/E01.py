import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
def generate_data():
    with open('names.txt', 'r') as f:
        names = f.read().splitlines()

        names = ('..' + '..'.join(names))
        chars=  [c for c in names]
        xs=  [(chars[i],chars[i+1]) for i in range(len(chars)-2)]
        xvocab = sorted(set(xs))
        xvocab_size = len(xvocab)
        xchar_to_idx = {ch: i for i, ch in enumerate(xvocab)}
        xidx_to_char = {i: ch for i, ch in enumerate(xvocab)}
        
        
        yvocab = sorted(set(names))
        yvocab_size = len(yvocab)
        ychar_to_idx = {char: idx for idx, char in enumerate(yvocab)}
        yidx_to_char = {idx: char for char, idx in ychar_to_idx.items()}
        ys = [i for i in names[2:]]

        xdata= torch.tensor([xchar_to_idx[i] for i in xs])
        onehotx = F.one_hot(xdata, xvocab_size).float()
        ydata = torch.tensor([ychar_to_idx[i] for i in ys])
        onehoty = F.one_hot(ydata, yvocab_size).float()
    
    
    return onehotx, onehoty, xdata, ydata, yvocab, yvocab_size, xvocab, xvocab_size, xchar_to_idx, yidx_to_char

def train(onehotx, onehoty, ydata, xvocab_size, yvocab_size, train_steps=500):
    W = torch.randn(xvocab_size, yvocab_size, requires_grad=True)
    num = onehotx.size(0)
    for step in range(train_steps):
        logits = onehotx @ W # (num, xvocab_size) @ (xvocab_size, yvocab_size) = (num, yvocab_size)
        counts = logits.exp()
        probs = counts / counts.sum(dim=1, keepdim=True) # (num, yvocab_size)
        loss = -probs[torch.arange(num), ydata].log().mean() + 0.1 * (W**2).mean()

        if step % 100 == 0:
            print(f'step {step}, loss {loss.item()}')
        
        W.grad = None
        loss.backward()
        
        W.data -= 50 * W.grad

    return W
import random
def generate(W, xchar_to_idx, yidx_to_char, xvocab_size,length=13):

    with torch.no_grad():
        string = '..'
        idx = xchar_to_idx[('.', '.')]
        for i in range(length):
            onehotx = F.one_hot(torch.tensor([idx]), xvocab_size).float()
            logits = onehotx @ W
            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdim=True) # (1, yvocab_size)
        
            
            sidx = torch.multinomial(probs,1)
            
            
            tok = yidx_to_char[sidx.item()]
            
            
            nstring = ((string[-1],tok))
            if tok == '.':
                break
            else:
                string += tok
            
            idx = xchar_to_idx.get(nstring,0)
            if idx == 0:
                break
    return string


def main():
    #
    onehotx, onehoty, xs, ys, yvocab, yvocab_size, xvocab, xvocab_size, xchar_to_idx, yidx_to_char= generate_data()
    
    print(f'x vocab_size {xvocab_size}') 
    print(f'y vocab_size {yvocab_size}')
    # now the prob table(W) is 574 x 55
    W = train(onehotx, onehoty, ys, xvocab_size, yvocab_size)

    for i in range(15):
        string = generate(W, xchar_to_idx, yidx_to_char, xvocab_size)
        print(string)
        print("=================================================")

if __name__ == '__main__':
    main()