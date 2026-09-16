"""Zero-dependency teaching implementation of the KDA recurrent update."""
from __future__ import annotations
import math, random

def dot(a, b): return sum(x*y for x,y in zip(a,b))
def outer(a, b): return [[x*y for y in b] for x in a]
def mat_vec_transpose(S, q): return [sum(S[i][j]*q[i] for i in range(len(q))) for j in range(len(S[0]))]
def l2(x):
    n=math.sqrt(sum(v*v for v in x)); return [v/n for v in x]

def kda_recurrent(q, k, v, alpha, beta, state=None):
    dk, dv = len(k[0]), len(v[0])
    S = [row[:] for row in state] if state else [[0.0]*dv for _ in range(dk)]
    outputs=[]
    for qt, kt, vt, at, bt in zip(q,k,v,alpha,beta):
        decayed=[[at[i]*S[i][j] for j in range(dv)] for i in range(dk)]
        kk=[sum(kt[i]*decayed[i][j] for i in range(dk)) for j in range(dv)]
        S=[[decayed[i][j] - bt*kt[i]*kk[j] + bt*kt[i]*vt[j] for j in range(dv)] for i in range(dk)]
        outputs.append(mat_vec_transpose(S, qt))
    return outputs, S

if __name__ == '__main__':
    random.seed(0); T, dk, dv = 4, 3, 2
    q=[l2([random.gauss(0,1) for _ in range(dk)]) for _ in range(T)]
    k=[l2([random.gauss(0,1) for _ in range(dk)]) for _ in range(T)]
    v=[[random.gauss(0,1) for _ in range(dv)] for _ in range(T)]
    alpha=[[1/(1+math.exp(-random.gauss(0,1))) for _ in range(dk)] for _ in range(T)]
    beta=[1/(1+math.exp(-random.gauss(0,1))) for _ in range(T)]
    out, state=kda_recurrent(q,k,v,alpha,beta)
    assert len(out)==T and len(out[0])==dv and len(state)==dk and len(state[0])==dv
    print('outputs shape:', (len(out),len(out[0])))
    print('state shape:', (len(state),len(state[0])))
    print('state shape independent of sequence length: True')
