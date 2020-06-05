#SUHANI CHACHA, 1001776270

import pandas as pd
import numpy as np
import random
import math
import sys
class neural_network(object):
	def __init__(self, training_file,test_file, layers, units_per_layer, rounds):
		self.training_file = training_file
		self.weights = []
		self.act_out = []
		self.test_file = test_file
		self.layers = layers
		self.units_per_layer = units_per_layer
		self.rounds = rounds
	def load(self, filename):
		df = pd.read_csv(filename, delim_whitespace = True, header = None)
		return df
	def split(self, data):
		data_class = data.iloc[:,-1]
		data = data.iloc[:,:-1]
		data = (data/(np.max(data.values))).values
		data = data.T
		data = np.vstack((np.ones(data.shape[1]),data)).T
		return data, data_class
	def weight_mat(self, non,i_d):
		weight = []
		for i in range(non):
			for j in range(i_d):
				x = random.uniform(-0.5,0.05)
				weight.append(x)
		weight = np.reshape(weight,(non,i_d))
		return weight
	def oneHot(self, training_class):
		df = []
		for i in training_class:
			bina = list()
			for j in range(0,len(np.unique(training_class))):
				bina.append(0)
			bina[int(i)-1] = 1
			df.append(bina)
		dataf = pd.DataFrame(df)
		return dataf

	def predict(self, weight_matrix, training_data):
		net = np.dot(weight_matrix, training_data)
		activation = (1 / (1 + np.exp(-net))).T
		b = pd.DataFrame(activation)
		return b.T
	def forward(self, training_data1):
		x = self.predict(self.weights[0], training_data1)
		self.act_out.append(x)
		if self.layers > 2:
			for k in range(3,layers):
				x = self.predict(network[k-2], self.act_out[-1])
				self.act_out.append(x)
			x = self.predict(network[-1], self.act_out[-1])
			self.act_out.append(x)
		return self.act_out[-1]
	def initialize_weight(self, n_inputs, n_outputs):
		
		if(layers == 2):
			w = self.weight_mat(n_outputs,n_inputs)
			self.weights.append(w)
		else:
			w = self.weight_mat(units_per_layer,n_inputs)
			self.weights.append(w)

			for i in range(3,layers):	
				w = self.weight_mat(self.weights[-1].shape[0],units_per_layer)
				self.weights.append(w)
			w = self.weight_mat(n_outputs,self.weights[-1].shape[0])
			self.weights.append(w)
		return self.weights
	def training(self, training_data,training_class,desired):
		training_data1 = training_data.T
		desired = desired.values
		for i in range(self.rounds):
			d = self.forward(training_data.T)
			if self.layers == 2:
				abc = (self.act_out[-1] - desired.T)*self.act_out[-1]*(1-self.act_out[-1])
				self.weights[-1] = self.weights[-1] - (math.pow(0.98,rounds-1))*np.dot(abc, training_data)
			else:
				abc = (self.act_out[-1] - desired.T)*self.act_out[-1]*(1-self.act_out[-1])
				self.weights[-1] = self.weights[-1] - (math.pow(0.98,rounds-1))*np.dot(abc, self.act_out[self.layers-3].T)
				for l in range(self.layers-3,1,-1):
					su = []
					nm = 1
					for c in range(self.layers-2,1,-1):
						xyz = self.act_out[c-1]*(1-self.act_out[c-1])
						ab = np.dot(self.weights[c-1],xyz)
						su.append(ab)
					xcd = sum(su)
					self.weights[nm] = self.weights[nm] - (math.pow(0.98,rounds-1))*np.dot(xcd, self.act_out[nm].T)
					nm = nm+1
	def testing(self, test_data,test_class):
		test_predict = self.forward(test_data.T)
		test_predict = test_predict.T
		output = np.argmax(test_predict.values, axis = 1)
		test_class1 = test_class.values
		acc = []
		for i in range(len(output)):
			if(output[i]==test_class1[i]):
				acc.append(1)
			else:
				acc.append(0)
		for i in range(len(test_class)):
			print("ID=%5d, predicted=%3d, true=%3d, Accuracy=%4.2f" %(i+1, output[i], test_class1[i], acc[i] ))
		print("Classification Accuracy = %6.4f\n" %((sum(acc)/len(test_class))*100))
if __name__ == "__main__":
	
	training_file = sys.argv[1]
	test_file = sys.argv[2]
	layers = int(sys.argv[3])
	units_per_layer = int(sys.argv[4])
	rounds = int(sys.argv[5])
	model = neural_network(training_file,test_file, layers, units_per_layer, rounds)
	training_data = model.load(training_file)
	test_data = model.load(test_file)
	training_data, training_class = model.split(training_data)
	test_data, test_class = model.split(test_data)
	network = model.initialize_weight(len(training_data[0]),len(np.unique(training_class)))
	desired = model.oneHot(training_class)
	model.training(training_data, training_class, desired)
	model.testing(test_data,test_class)