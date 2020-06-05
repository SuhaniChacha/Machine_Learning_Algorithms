import pandas as pd
import numpy as np
import math
def naive_bayes(traindata,testdata):
    def load_data(filename):
        df = pd.read_csv(filename, delim_whitespace = True,header = None)
        return df

    def gd(x,mean,std):
        if std < 0.01:
            std = 0.01
        exponent = math.exp(-(math.pow((x-mean),2)/(2*math.pow(std,2))))
        pro = (1 / (math.sqrt(2*math.pi) * std ) * exponent)
        return pro

    df = load_data(traindata)
    j = len(df.columns)
    df.columns = np.array(range(1,j+1))
    attr_vals = dict()
    class_list = df.iloc[:,-1].unique()
    class_list = class_list.tolist()
    class_list = sorted(class_list)
    for cl in class_list:
        for attr in range(1,df.shape[1]):
            mean_val = np.mean(df[df[df.shape[1]]==cl][attr])
            std_val = np.std(df[df[df.shape[1]]==cl][attr])
            attr_vals[(cl,attr)] = {"Mean":mean_val,"Std":std_val}
    probclass=dict()
    for cl in class_list:
        probclass[cl] = df[df.iloc[:,-1]==cl].shape[0]/df.shape[0]
    for cl in class_list:
        for attr in range(1,df.shape[1]):
            print("Class: % d, attribute : % d, Mean: % .2f, Std: % .2f" %( cl, attr, attr_vals[(cl,attr)]['Mean'],attr_vals[(cl,attr)]['Std']))
    testing_data = load_data(testdata)
    h = len(testing_data.columns)
    testing_data.columns = np.array(range(1,j+1))
    prob_list=[]
    pred_list=[]
    accu=[]
    row_num=0
    total_prob=[]
    for index, row in dft.iterrows():
        prob = 0
        total = 0
        for cl in class_list:
            temp2 = 1
            for attr in range(1,dft.shape[1]):
                temp2*=gd(row[attr],attr_vals[(cl,attr)]['Mean'],attr_vals[(cl,attr)]['Std'])
            temp2*=probclass[cl]
            total+=temp2
        total_prob.append(total)
    for index, row in dft.iterrows():
        prob = 0
        pred = -1
        pred2 = []
        for cl in class_list:
            temp2 = 1
            acc = 1
            for attr in range(1,dft.shape[1]):
                temp2*=gd(row[attr],attr_vals[(cl,attr)]['Mean'],attr_vals[(cl,attr)]['Std'])
            temp2*=probclass[cl]
            temp2/=total_prob[row_num]
            if(temp2>prob):
                prob = temp2
                pred = cl
        if(pred==dft.loc[row_num,Test_Col_len]):
            acc=1
            accu.append(acc)
        else:
            acc=0
            accu.append(acc)

        pred_list.append(pred)
        prob_list.append(prob)
        print("ID: % 3d, Predicted : % 2d, Probability: % 3f, True: % 3d, Accuracy: % 4.2f" %(row_num+1,pred,prob,dft.loc[row_num,Test_Col_len],acc))
        row_num+=1
    print("Classification accuracy = %6.4f" %((sum(accu)/len(accu))*100))
    
naive_bayes(input(),input())

