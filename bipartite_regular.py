'''
label propagation on bipartite graphs
'''

# Author: Xing Yu <staryu.udi@gmail.com>

def BGLabelPropagation(X, Y, offset, labeledANum, labeledBNum, epsilon = 1e-2, maxIterNum = 50000):
    
    '''Label propagation classifier on a bipartited graph.
    
    This version of label propagation algorithm can classify nodes on a bipartite graph.  
    The bipartited graph has two types of nodes: A and B. An edge never connects two nodes with the same type.
    The two types of node shares a same classes.
    
    Parameters:
    -----------
    X : numpy array of floats | doubles
        X is a N by N matrix that represent the adjacency matrix of the bipartite graph.
        Each row is a representation of a node. 
        Each row is normalized (sums to 1).
        Type A nodes are on top of all type B nodes

    Y : numpy array of floats | doubles
        Y is a N by L matrix.
        Each row is the distribution of classes' probabilities of a node. L is the number of all possible classes. 
        Also normalized by row

    offset : int
        The number of type A nodes, which are assumed on top of type B nodes in X and Y

    labeledANum : int
        Number of labeled instances for type A nodes

    labeledBNum : int
        Number of labeled instances for type B nodes

    epsilon: float, default 1e-5
        The error value for checking convergence

    maxIterNum: int, default 50000
        The maximum number of iteration.
        If the algorithm did not converge after this number of iteration
        -1 will be returned
    '''
    
    import numpy as np
    from sklearn import preprocessing
    import sys

    # sys.setrecursionlimit(maxIterNum + 1)
    
    # separate the adjacency matrix
    T_aubl = X[labeledANum : offset, offset : offset + labeledBNum]
    T_aubu = X[labeledANum : offset, offset + labeledBNum : ]
    T_bual = X[offset + labeledBNum :, 0 : labeledANum]
    T_buau = X[offset + labeledBNum :, labeledANum : offset]
    
    # separate the class distribution
    Y_al = Y[0 : labeledANum, :]
    Y_au = Y[labeledANum : offset, :]
    Y_bl = Y[offset : offset + labeledBNum, :]
    Y_bu = Y[offset + labeledBNum : , :]
    
    # propagation functions
    # Y_au = T_aubl.Y_bl + T_aubu.Y_bu
    # Y_bu = T_bual.Y_al + T_buau.Y_au
    
    def propagate(T_aubl, T_aubu, T_bual, T_buau, Y_al, Y_au, Y_bl, Y_bu):
        Y_au_new = preprocessing.normalize(T_aubl.dot(Y_bl) + T_aubu.dot(Y_bu), norm = 'l1')
        Y_bu_new = preprocessing.normalize(T_bual.dot(Y_al) + T_buau.dot(Y_au), norm = 'l1')
        return (Y_au_new, Y_bu_new)
    
    # convergence function to check if Y_au and Y_bu converged
    convergeCondition = lambda current, pre: True if (np.abs(current[0] - pre[0]).sum() + np.abs(current[1] - pre[1]).sum()) < epsilon else False
       
    count = 0
    converged = False

    while count < maxIterNum:

        nextResult = propagate(T_aubl, T_aubu, T_bual, T_buau, Y_al, Y_au, Y_bl, Y_bu)

        Y_au_new = nextResult[0]
        Y_bu_new = nextResult[1]

        if convergeCondition((nextResult), (Y_au, Y_bu)):
            converged = True
            return np.concatenate((Y_al, Y_au , Y_bl, Y_bu))
        else:
            Y_au = Y_au_new
            Y_bu = Y_bu_new

        count += 1

    # not converged after maximum iteration number
    if converged == False:
        return -1