"""
Created on Fri Apr  3 01:23:19 2020

@author: Narges Alavi
"""
import numpy as np
import statsmodels.api as sm

def backElimination(x, y, sl):
    dataSize = len(x[:,1])
    x = np.append(arr = np.ones((dataSize,1)).astype(int), values = x, axis = 1)
    numVars = len(x[1,:])
    
    for i in range(0,numVars):
        regressor_OLS = sm.OLS(endog = y, exog = x).fit() 
        maxP = max(regressor_OLS.pvalues).astype(float)
        
        if maxP > sl:
            for j in range(0, numVars - i):
                if regressor_OLS.pvalues[j] == maxP:
                    x = np.delete(x, j, 1)
    return x

