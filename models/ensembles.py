from utils import *
from binTree import *

def bagging(x, y, max_depth, attribute_value_pairs, num_trees):

    h_ens = []
    xTranspose = np.transpose(x)
    seed=int.from_bytes(os.urandom(64), sys.byteorder)
    random.seed(seed) #re-seed generator

    for i in range(0,num_trees):

        bootstrapX = []
        bootstrapY = []
        indices = []

        for j in range(0,len(xTranspose)):
            k = random.randint(0,len(xTranspose)-1) # generate a random index
            #indices.append(k)
            # add the kth sample to the bootstrap sample set
            bootstrapX.append(xTranspose[k])
            bootstrapY.append(y[k])


        #setIndices = set(indices) # To track the proportion of unique examples in each bag
        #uniques.append(len(setIndices)/len(indices))

        # create a tree and add it to the ensemble hypothesis
        decision_tree = id3(np.transpose(bootstrapX), bootstrapY, attribute_value_pairs=attribute_value_pairs.copy(), max_depth=max_depth)
        h_ens.append([1,decision_tree]) # Note: weight of each classifier is the same

    return h_ens

def boosting(x, y, max_depth, num_stumps, attribute_value_pairs):

    h_ens = []

    #create a list to hold the weights of the training example
    #also called the distribution d
    #initialize weight for each training example to be 1/N
    n = len(y)
    d = [1/n]*n

    while len(h_ens) < num_stumps:
        h = id3(x,y,attribute_value_pairs,max_depth=max_depth, w=d)
        epsilon, preds = weighted_errorAndPreds(h, x, y, d)
        if epsilon == 0:
            epsilon == min(d)
            if epsilon == 0:
                epsilon = 0.0001
        alpha = calc_alpha(epsilon)
        wSum = 0
        for i in range(len(d)):
            if preds[i]:
                factor = exp(-alpha)
            else:
                factor = exp(alpha)
            d[i] = d[i]*factor
            wSum += d[i]

        d[:] = [w/wSum for w in d]
        h_ens.append([alpha, h])

    return h_ens
