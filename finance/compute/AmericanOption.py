#s = 4 #CurrentStockPrice
#depth = 2 #TreeDepth Number of Days
#k = 5  #Strike Price
#u = 2.0
#d = 0.5
#r = 0.25

def getAmericanValuesBinomial(S,depth,k,u,d,r):
    l  = 2**(depth+1) -1
    s = [-1,S]
    p = (1.0+r-d)/(u-d)
    q = (u-(1.0+r))/(u-d)
    #print p,q
    #print range(2,l+1)
    for i in range(2,l+1):
        inx = i//2
        if i%2 == 0:
            s.append(u*s[inx])
        else:
            s.append(d*s[inx])
    #print s
    v = [0]*(l+1)
    #print v
    for i in range(2**depth,l+1):
        v[i] = max(k-s[i],0)
    #print v
    for i in range(2**depth-1,0,-1):
        price = (1.0/(1.0+r))*(p*v[2*i]+q*v[2*i+1])
        v[i] = max(k-s[i],price)
    return v

def getAmericanValuesLeastSquareNormal(n_paths, p_length, S_zero, K, r, volatility):
    return 0.5
def getAmericanValuesLeastSquareNeural(n_paths, p_length, S_zero, K, r, volatility):
    return 1.5

#V = getAmericanValuesBinomial(4,depth,k,u,d,r)
#print (V)