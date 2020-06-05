import pandas as pd
import numpy as np
import statistics as st
import math
import sys
def linear_regression(training_file, degree, lambda_val, test_file):
    def load_data(filename):
        df = pd.read_csv(filename, delim_whitespace = True, header = None)
        return df

    def generate_phie_matrix(df,degree):
        output = df[len(df.columns)-1]
        df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
        n = df.values
        a= []
        for x in n:
            for j in x:
                for i in range(1,degree+1):
                    a.append(math.pow(j,i))           
        phie = np.reshape(a,(len(df),len(df.columns)*degree))
        df = pd.DataFrame(phie,columns=None)
        df.insert(0,'0',1,allow_duplicates = True)  
        df_transpose =df.T
        df_list = df.values
        df_transpose_list = df_transpose.values
        return(df_list,df_transpose_list,output.values)

    def weight_randomize(df_list):
        x = np.shape(df_list)
        w = np.random.random(x)
        return w

    def calculate_weight(phie_data,phie_data_transpose,w,lemda_val,output):
        c_ln = np.multiply(lemda_val,np.eye(len(phie_data[0])))
        c_rn = np.dot(phie_data_transpose,phie_data)
        c_n = c_ln + c_rn
        c_n_inverse = np.linalg.pinv(c_n)
        l_m = np.dot(c_n_inverse, phie_data_transpose)
        f = np.dot(l_m,output)
        return f
    def predict(weight_matrix,testing_data,degree):
        phie_data,phie_data_transpose,output = generate_phie_matrix(testing_data,degree)
        predicted = np.dot(phie_data,weight_matrix)
        return predicted.astype(int),output
    
    training_data = load_data(training_file)
    testing_data = load_data(test_file)
    phie_data, phie_data_transpose,output = generate_phie_matrix(training_data,degree)
    w = weight_randomize(phie_data)
    weight_matrix = calculate_weight(phie_data,phie_data_transpose,w,lambda_val,output)
    print("---------------------Training Phase --------------------------") 
    for i in range(len(weight_matrix)):
        print("w",i,"= %.4f" %(weight_matrix[i]))
    p_o,a_o = predict(weight_matrix,testing_data,degree)
    print("---------------------Testing Phase --------------------------")
    for i in range(1,len(p_o)+1):
        error = math.pow((p_o[i-1]-a_o[i-1]),2)
        print("ID = %5d, OUTPUT = %14.4f, TAGRET = %10.4f, SQAURED ERROR = %4f" %(i,p_o[i-1],a_o[i-1],error))
linear_regression(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4])