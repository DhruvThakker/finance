#s = 4 #CurrentStockPrice
#depth = 2 #TreeDepth Number of Days
#k = 5  #Strike Price
#u = 2.0
#d = 0.5
#r = 0.25
import numpy as np
import math
from sklearn.neural_network import MLPRegressor

# Function to generate Paths
def generate_paths(paths, n_paths, p_length, S_zero, r, volatility):
	for j in range(n_paths):
		paths[j][0] = S_zero

	mu = r
	sigma = volatility

	for i in range(n_paths):
		for j in range(1, p_length):
			W = np.random.normal(0, 1, 1)
			paths[i][j] = paths[0][0] * math.exp((mu - ((sigma * sigma) / 2)) * j + sigma * W)

# Mathematical Function Y = A0 + A1 X + A2 X X
def estimateFunctionNN(x, clf):
	X = np.array([x])
	X = X.reshape(1,-1)
	return clf.predict(X)

# Main Method
def getAmericanValuesLeastSquareNeural(n_paths, p_length, S_zero, K, r, volatility):
    t = 1
    paths = np.zeros((n_paths, p_length), dtype=float)
    generate_paths(paths, n_paths, p_length, S_zero, r, volatility)

    payoff = np.zeros((n_paths, p_length), dtype=float)
    cash_flow = np.zeros((n_paths, p_length), dtype=float)
    stopping_rule = np.zeros((n_paths, p_length), dtype=int)
    clf = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(5, 2))
	
    for j in range(p_length):
    	i = p_length - j - 1
    	for k in range(n_paths):
            payoff[k][i] = max(0, K - paths[k][i])
    for j in range(n_paths):
        cash_flow[j][p_length - 1] = max(0, K - paths[j][p_length - 1])
    for j in range(p_length - 2):
	    i = p_length - 2 - j
	    indicies = []
	    X = []
	    Y = []
	    for k in range(n_paths):
	    	if (payoff[k][i] > 0):
	    		X.append(paths[k][i])
	    		Y.append(cash_flow[k][i + 1] * math.exp(- r * t))
	    		indicies.append(k)
	    X = np.asarray(X, dtype=np.float64)
	    Y = np.asarray(Y, dtype=np.float64)
	    X = X.reshape(-1, 1)
	    clf.fit(X, Y)
	    for k in range(len(indicies)):
	    	if estimateFunctionNN(paths[indicies[k]][i], clf) < payoff[indicies[k]][i]:
	    		cash_flow[indicies[k]][i] = payoff[indicies[k]][i]
	    		for y in range(i + 1, p_length):
	    			cash_flow[indicies[k]][y] = 0
	    	else:
	    		cash_flow[indicies[k]][i] = 0
    tempSum = 0
    for j in range(n_paths):
	    for i in range(p_length):
	    	if (cash_flow[j][i] > 0):
	    		stopping_rule[j][i] = 1
	    		tempSum += cash_flow[j][i] * math.exp(- r * i)
    option_price = tempSum/n_paths
    return option_price

# Mathematical Function Y = A0 + A1 X + A2 X X
def estimateFunction(x, A):
	tempSum = 0
	for i in range(len(A)):
		tempSum += A[i] * (x ** i)
	return tempSum

# Function to calculcate A0, A1, A2, ... using Matrix Multiplication
def regression(Y, X, reg_length):
	A = np.zeros((1, reg_length), dtype=float)
	B = np.zeros((reg_length, reg_length), dtype=float)
	C = np.zeros((reg_length, 1), dtype=float)
	l_X = len(X)

	for i in range(reg_length):
		for j in range(reg_length):
			tempSum = 0
			for k in range(l_X):
				tempSum += X[k]**(i + j)
			B[i][j] = tempSum
	
	for j in range(reg_length):
		tempSum = 0
		for k in range(l_X):
			tempSum += Y[k] * (X[k]**j)
		C[j] = tempSum

	A = np.matmul(np.linalg.inv(B), C)
	return A

def getAmericanValuesLeastSquareNormal(n_paths, p_length, S_zero, K, r, volatility):
    t = 1
    paths = np.zeros((n_paths, p_length), dtype=float)
    generate_paths(paths, n_paths, p_length, S_zero, r, volatility)

    payoff = np.zeros((n_paths, p_length), dtype=float)
    cash_flow = np.zeros((n_paths, p_length), dtype=float)
    stopping_rule = np.zeros((n_paths, p_length), dtype=int)

    for j in range(p_length):
    	i = p_length - j - 1
    	for k in range(n_paths):
    		payoff[k][i] = max(0, K - paths[k][i])

    for j in range(n_paths):
    	cash_flow[j][p_length - 1] = max(0, K - paths[j][p_length - 1])

    for j in range(p_length - 2):
    	i = p_length - 2 - j
    	indicies = []
    	X = []
    	Y = []
    	for k in range(n_paths):
    		if (payoff[k][i] > 0):
    			X.append(paths[k][i])
    			Y.append(cash_flow[k][i + 1] * math.exp(- r * t))
    			indicies.append(k)

    	A = regression(Y, X, 3)

    	for k in range(len(indicies)):
    		if estimateFunction(paths[indicies[k]][i], A) < payoff[indicies[k]][i]:
    			cash_flow[indicies[k]][i] = payoff[indicies[k]][i]
    			for y in range(i + 1, p_length):
    				cash_flow[indicies[k]][y] = 0
    		else:
    			cash_flow[indicies[k]][i] = 0

    tempSum = 0
    for j in range(n_paths):
    	for i in range(p_length):
    		if (cash_flow[j][i] > 0):
    			stopping_rule[j][i] = 1
    			tempSum += cash_flow[j][i] * math.exp(- r * i)

    option_price = tempSum/n_paths
    return option_price

def getAmericanValuesBinomial(S,depth,k,u,d,r):
    l  = 2**(depth+1) -1
    s = [-1,S]
    p = (1.0+r-d)/(u-d)
    q = (u-(1.0+r))/(u-d)
    for i in range(2,l+1):
        inx = i//2
        if i%2 == 0:
            s.append(u*s[inx])
        else:
            s.append(d*s[inx])
    v = [0]*(l+1)
    for i in range(2**depth,l+1):
        v[i] = max(k-s[i],0)
    for i in range(2**depth-1,0,-1):
        price = (1.0/(1.0+r))*(p*v[2*i]+q*v[2*i+1])
        v[i] = max(k-s[i],price)
    return v

def getAmericanValuesFiniteDifference(S0,K,r,sig,t,M,N,dS):
	dt = t/N
	f = [ [-1]*(M+1) for _ in range(N+1) ]
	for i in range(M+1):
	    f[N][i] = max(K-i*dS,0)
	for i in range(N+1):
	    f[i][0] = K
	    f[i][M] = 0

	for idx in range(N):
	    a = []
	    b = []
	    i = N-1-idx
	    for jdx in range(M-1):
	        j = jdx+1
	        at = [0]*(M-1)
	        if jdx>0:
	            at[jdx-1] = 1/2*(r*j*dt) - 1/2*(sig*sig*j*j*dt)
	        at[jdx] = 1 + r*dt + sig*sig*j*j*dt
	        if jdx<M-2:
	            at[jdx+1] = -1/2*(r*j*dt) - 1/2*(sig*sig*j*j*dt)
	        bt = f[i+1][j]
	        if jdx==0:
	            bt = bt - (1/2*(r*j*dt) - 1/2*(sig*sig*j*j*dt))*f[i][0]
	        if jdx==M-2:
	            bt = bt - (-1/2*(r*j*dt) - 1/2*(sig*sig*j*j*dt))*f[i][M]
	        a.append(at)
	        b.append(bt)
	    a = np.array(a)
	    b = np.array(b)
	    x = np.linalg.solve(a, b)
	    for jdx in range(M-1):
	        j = jdx+1
	        f[i][j] = max(x[jdx],K-j*dS)
	return f[0][int(S0/dS)] 

#V = getAmericanValuesBinomial(4,depth,k,u,d,r)
#print (V)