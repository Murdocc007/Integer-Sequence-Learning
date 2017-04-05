from __future__ import division
import numpy as np
# functions and classes go here


#extract all the sequences from the file and returns the sequence and the expected final value
def getSequences(filePath):
    testfile=filePath
    data = open(testfile).readlines()
    finalVal={} # (key,value)
    sequences={}   #(key, value) = (id , sequence)
    for i in range(1,len(data)):
        line=data[i]
        line =line.replace('"','')
        line = line[:-1].split(',')
        id = int(line[0])
        n=len(line)
        sequence=[int(x) for x in line[1:n-1]]
        finalVal[id]=int(line[n-1])
        sequences[id]=sequence
    return sequences,finalVal

def fb_alg(A_mat, O_mat, observ,ranks):
    # set up
    k = observ.size
    (n,m) = O_mat.shape
    prob_mat = np.zeros( (n,k) )
    fw = np.zeros( (n,k+1) )
    bw = np.zeros( (n,k+1) )
    # forward part
    fw[:, 0] = 1.0/n
    for obs_ind in xrange(k):
        f_row_vec = np.matrix(fw[:,obs_ind])
        fw[:, obs_ind+1] = f_row_vec * np.matrix(A_mat) *np.matrix(np.diag(O_mat[:,ranks[obs_ind]]))
    fw[:,obs_ind+1] = fw[:,obs_ind+1]/np.sum(fw[:,obs_ind+1])
    # backward part
    bw[:,-1] = 1.0
    for obs_ind in xrange(k, 0, -1):
        b_col_vec = np.matrix(bw[:,obs_ind]).transpose()
        bw[:, obs_ind-1] = (np.matrix(A_mat) * np.matrix(np.diag(O_mat[:,ranks[obs_ind-1]])) * b_col_vec).transpose()
    bw[:,obs_ind-1] = bw[:,obs_ind-1]/np.sum(bw[:,obs_ind-1])
    # combine it
    prob_mat = np.array(fw)*np.array(bw)

    prob_mat = prob_mat/np.sum(prob_mat, 0)
    # get out
    return prob_mat, fw, bw

def baum_welch( num_states, num_obs, observ,ranks ):
    # allocate
    A_mat = np.ones( (num_states, num_states) )
    A_mat = A_mat / A_mat.sum(axis=1)[:,None]
    O_mat = np.ones( (num_states, num_obs) )
    O_mat = O_mat / O_mat.sum(axis=1)[:,None]
    theta = np.zeros( (num_states, num_states, observ.size) )
    while True:
        old_A = A_mat
        old_O = O_mat
        A_mat = np.ones( (num_states, num_states) )
        O_mat = np.ones( (num_states, num_obs) )
        # expectation step, forward and backward probs
        P,F,B = fb_alg( old_A, old_O, observ,ranks)
        # need to get transitional probabilities at each time step too

        for a_ind in xrange(num_states):
            for b_ind in xrange(num_states):
                for t_ind in xrange(observ.size):
                    theta[a_ind,b_ind,t_ind] = F[a_ind,t_ind] * B[b_ind,t_ind+1] * old_A[a_ind,b_ind] * old_O[b_ind, ranks[t_ind]]

        # form A_mat and O_mat
        for a_ind in xrange(num_states):
            for b_ind in xrange(num_states):
                A_mat[a_ind, b_ind] = np.sum( theta[a_ind, b_ind, :] )/np.sum(P[a_ind,:])

        A_mat = A_mat / A_mat.sum(axis=1)[:,None]

        for a_ind in xrange(num_states):
            for o_ind in xrange(num_obs):
                right_obs_ind = np.array(np.where(observ == observ[o_ind]))+1
                O_mat[a_ind, o_ind] = np.sum(P[a_ind,right_obs_ind])/ np.sum( P[a_ind,1:])

        O_mat = O_mat / O_mat.sum(axis=1)[:,None]
        # compare
        if np.linalg.norm(old_A-A_mat) < .00001 and np.linalg.norm(old_O-O_mat) < .00001:
            break
    # get out
    return A_mat, O_mat

def solve(sequence,numObs,numStates):
    sequence=sequence[-numObs:]
    sequence=np.array(sequence)
    _,id = np.unique(sequence,return_inverse=True)
    ranks = (id.max() - id ).reshape(sequence.shape)
    transition_matrix,emission_matrix=baum_welch(numStates,np.unique(sequence).size,sequence,ranks)
    return transition_matrix,emission_matrix


if __name__=='__main__':
    sequences,finalVals=getSequences('/home/aditya/Desktop/Aditya/Statistical Methods ML/Project/test.csv')
    print solve(sequences[1],10,3)



