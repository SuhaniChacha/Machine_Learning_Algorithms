import pandas as pd
import numpy as np
import sys
def knn_classifier(trainingfile, testfile, k):
	def load(file):
		df = pd.read_csv(file,delim_whitespace = True,header = None)
		colu = df.columns
		df = df.rename(columns= {colu[-1]:"label"})
		return df

	def normalize_data(df):
		df_mean = df.sub(df.mean())
		df_std = df_mean.div(df.std())
		del df_std["label"]
		return df_std
	 
	def euclidian_distance(row,train,train_actual,k):
		l = []
		a = []
		demo = train
		demo = demo.sub(row)
		demo = demo**2
		demo = demo.sum(axis=1)
		demo = demo**(1/2)
		demo = demo.sort_values()
		demo["index"] = demo.index
		demo = demo.reset_index()
		rows = demo.iloc[:k]
		l = rows["index"].tolist()
		for i in l:
			b = train_actual.iloc[[i]]
			b = b["label"].tolist()
			a.append(b)
		cl_vl = []
		for i in a:
			for j in i:
				cl_vl.append(j)
		return (max(set(cl_vl), key=cl_vl.count))



	def classify(train,train_actual,test,test_actual,k):
		class_predicted = []
		error = []
		actual_class_label = test_actual["label"].tolist()
		
		for index, row in test.iterrows():
			class_predicted.append(euclidian_distance(row,train,train_actual,k))
		for i in range(0,len(test)):
			if(class_predicted[i]==actual_class_label[i]):
				error.append(1)
			else:
				error.append(0)
		classification_error = sum(error)/len(error)
		print(class_predicted)
		print(actual_class_label)
		for i in range(0,len(test)):
			print('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2lf\n'%(i+1, class_predicted[i], actual_class_label[i], error[i]));
		print('classification accuracy=%6.4f\n'%(classification_error))

	train = load(trainingfile)
	test = load(testfile)
	train_normalized = normalize_data(train)
	test_normalized = normalize_data(test)
	classify(train_normalized,train,test_normalized,test,k)
	

knn_classifier(sys.argv[1],sys.argv[2],int(sys.argv[3]))