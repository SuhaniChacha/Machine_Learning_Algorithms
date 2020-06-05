
#Suhani Chacha
#1001776270

import sys
import numpy as np
def bayes_estimation(textfile):
    def load_data(textfile):
        #print("reading data ---------")
        data = [char for char in open(textfile).read()]
        #print("completion of reading phase -------")
        return data
    
    data = load_data(textfile)
    #print("printing data ---------------")
    #print(data)
    
    #prior defination 
    m = [0.1,0.3,0.5,0.7,0.9]
    p = [0.9,0.04,0.03,0.02,0.01]
    
    #temporary variable declaration
    m_n = m
    p_n = p

    #function that calculates posterior probability for given m and p list
    def posterior_calculator(m,p):
        store = np.sum(np.multiply(m,p))
        z = np.multiply(m,p)/store
        return z
    
    #calculating values when each and every element is encontered in a
    for i in data:   
        # when a is encontered
        if i =='a':
            p_n = posterior_calculator(m,p_n)
        else:
            #when b is encontered
            p_n =posterior_calculator(m[::-1],p_n[::-1])
            
    #prinitg the calculated posterior probabilities
    for i in range(len(m)):
        print("p(m=",m[i],"|data) = %.4f" %p_n[i])
    
    #printing p(c='a'|data) using sum rule
    print("p(c='a'|data) = %.4f "%(np.sum(np.multiply(m,p_n))))
    
bayes_estimation(sys.argv[0])  