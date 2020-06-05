import math
import pandas as pd
import numpy as np
r=20
c = 2
array = np.random.random((r,c))
array = array.tolist()
k = 3
def function_here(array,k):
    new_list = []
    size = math.ceil(len(array)/k)
    new_list.append([array[i:i+size] for i  in range(0, len(array), size)])
    sum_error = math.inf
    prev_error = 0
    while (sum_error != prev_error):
        error = []
        for i in range(0,len(array)):
            mean = []
            new_l=[]
            for l in range(0,k):
                df =  pd.DataFrame(new_list[0][l])
                mean.append(df.mean().values)
            for j in range(0,k):
                compare_list = abs(array[i]-mean[j])
                #print("c",compare_list)
                val = compare_list.sum()
                #print(val)
                new_l.append(val)
            #print("new_l",new_l)
            min_val = min(new_l)
            error.append(min_val)
            ind = new_l.index(min_val)
            #print(ind)
            for pi in range(0,len(new_list[0])):
                p = new_list[0][pi]
                for qi in range(0,len(p)):
                    q = p[qi]
                    if q == array[i]:
                        del new_list[0][pi][qi]
                        break
            new_list[0][ind].append(array[i])
        sum_error = sum(error)
        prev_error = sum_error
        return new_list
cluster_list = function_here(array,k)
for i in range(0,k):
    print("cluster " , i+1 , " : " )
    print(cluster_list[0][i])
