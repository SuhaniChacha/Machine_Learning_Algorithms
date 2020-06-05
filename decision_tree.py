import numpy as np
import random
import math
import sys
import pandas as pd

class Tree(object):
	def __init__(self,value=None,threshold = None,gainval=None):
		self.data = value
		self.threshold = threshold
		self.left_child = None
		self.right_child= None
		self.gainval = gainval

class Leaf(object):
	def __init__(self,class_label,dist):
		self.class_label = class_label
		self.dist = dist
def decision_tree(trainingfile,testingfile,option,pruning_thr):
	def load(filehere):
		df = pd.read_csv(filehere, delim_whitespace = True,header = None)
		colu = df.columns
		df = df.rename(columns = {colu[-1]:"label"})
		return df

	def is_unique(train):
		class_val = np.unique(np.array(train["label"].tolist()))
		if(len(class_val) == 1):
			return True
		else:
			return False

	def dtl_toplevel(train,pruning_thr):
		attribute = train.columns
		attribute = list(attribute)
		attribute.remove("label")
		default_v = distribution(train)
		return dtl(train,attribute,default_v,pruning_thr)


	def distribution(train):
		unique_classlist = np.unique(np.array(train["label"].tolist()))
		a = []
		for i in unique_classlist:
			a.append(len(train[train["label"] == i])/len(train))	
		return a

	def information_gain(examples,a,threshold):
		if threshold != 0:
			unique_classlist = np.unique(np.array(examples["label"].tolist()))
			class_node = examples
			class_data1 =  examples[examples[a] < threshold]
			class_data2 = examples[examples[a] >= threshold]
			k_class1= []
			k_class2 = []
			k_node = []
			for class_val in unique_classlist:
				c = class_data1[class_data1["label"] == class_val]
				d = class_data2[class_data2["label"] == class_val]
				g = class_node[class_node["label"] == class_val]
				if len(class_data1) != 0:
					entropy1 = len(c)/len(class_data1)
					if entropy1 != 0:
						entropy1 = entropy1 * math.log2(entropy1)
				else:
					entropy1 = 0
				if len(class_data2) != 0:
					entropy2 = len(d)/len(class_data2) 
					if entropy2 != 0:
						entropy2 = entropy2 * math.log2(entropy2)
				else:
					entropy2 = 0
				if len(class_node) != 0 :
					entropy3 = len(g)/len(class_node) 
					if entropy3 != 0:
						entropy3 = entropy3 * math.log2(entropy3)
				else:
					entropy3 = 0
				k_class1.append(-1 * entropy1)
				k_class2.append(-1 * entropy2)
				k_node.append(-1 * entropy3)
			entropy1 = sum(k_class1)
			entropy2 = sum(k_class2)
			entropy3 = sum(k_node)
			if len(class_node) != 0:
				infogain = entropy3 - ((len(class_data1)/len(class_node))*entropy1) - ((len(class_data2)/len(class_node))*entropy2)
			else:
				infogain = 0
			return infogain
		else:
			return 0

	def choose_attribute_randomly(examples,att):
		max_gain = best_threshold = -1
		a = random.choice(att)
		attribute_value = examples[a]
		l = attribute_value.min()
		m = attribute_value.max()
		for k in range(1,51):
			threshold = l + k * (m-l)/51
			gain = information_gain(examples,a,threshold)
			if gain > max_gain:
				max_gain = gain
				best_threshold = threshold
		return (a,best_threshold,max_gain)

	def choose_attribute_optimized(examples,attributes):
		max_gain = best_attribute = best_threshold = -1
		for a in attributes:
			attribute_value = examples[a]
			l = attribute_value.min()
			m = attribute_value.max()
			for k in range(1,51):
				threshold = l + k *(m-l)/51
				gain = information_gain(examples, a, threshold)
				if gain > max_gain:
					max_gain = gain
					best_threshold = threshold
					best_attribute = a
		return (best_attribute,best_threshold,max_gain)

	def dtl(examples,att,default_v,pruning_thr):
		if len(examples)<pruning_thr:
			return Leaf(default_v.index(np.max(default_v)), default_v)
		elif is_unique(examples):
			c = np.unique(np.array(examples["label"].tolist()))
			return Leaf(c[0],distribution(examples))
		elif len(att)==0:
			max_class_dist = distribution(examples)
			max_class = max_class_dist.index(np.max(max_class_dist))
			return Leaf(max_class,max_class_dist)
		else:
			if (option == "randomized"):
				best , threshold , gain = choose_attribute_optimized(examples,att)
			elif (option == "optimized"):
				best , threshold , gain = choose_attribute_randomly(examples,att)
			root = Tree(best,threshold,gain)
			examples_left = examples[examples[best] < threshold]
			examples_right = examples[examples[best] >= threshold]
			att.remove(best)
			root.left_child = dtl(examples_left,att,distribution(examples),pruning_thr)
			root.right_child = dtl(examples_right,att,distribution(examples),pruning_thr)
			return root

	def print_tree(root, tree_id):
	    queue = [root]
	    node_id = 1
	    while(queue):
	        node = queue[0]
	        if isinstance(node,Leaf):
	            print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n' %(tree_id, node_id, -1, -1, 0))
	        else:
	            print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n' %(tree_id, node_id, (node.data+1), node.threshold, node.gainval))
	            if node.left_child is not None:
	                queue.append(node.left_child)
	            if node.right_child is not None:
	                queue.append(node.right_child)
	        queue.pop(0)
	        node_id += 1

	def tree_traversal(element,examples):
		if isinstance(element, Leaf):
			return element.dist
		else:
			column_no = element.data
			colthreshold = element.threshold
			if examples[column_no] < colthreshold:
				dist = tree_traversal(element.left_child, examples)
				return dist
			else:
				dist = tree_traversal(element.right_child, examples)
				return dist
	train= load(trainingfile)
	test = load(testingfile)

	forest = []
	if ((option == "randomized") or (option == "optimized")):
		tree_here = dtl_toplevel(train,pruning_thr)
		print_tree(tree_here,1)
		forest.append(tree_here)
	elif ((option == "forest3") ):
		option = "randomized"
		for i in range(1,4):
			tree = dtl_toplevel(train, pruning_thr)
			print_tree(tree, i)
			forest.append(tree)

	elif ((option == "forest7")):
		option = "randomized" 
		for i in range(1,8):
			tree = dtl_toplevel(train, pruning_thr)
			print_tree(tree, i)
			forest.append(tree)
	elif ((option == "forest15")):
		option = "randomized" 
		for i in range(1,16):
			tree = dtl_toplevel(train, pruning_thr)
			print_tree(tree, i)
			forest.append(tree)
	
	def test_funxction(test,forest):
		accura = []
		class_accura = 0
		estimatedclass = []
		for index, row in test.iterrows():
			class_label = row["label"]
			del row["label"]
			test_dist = []
			for tree in forest:
				test_dist.append(tree_traversal(tree,row))
			if len(test_dist)==0:
				test_dist = test_dist[0]
			else:
				len_test_dist = len(test_dist)
				test_dist = [sum(x) for x in zip(*test_dist)]
				test_dist = [i/len_test_dist for i in test_dist]

			estcls = test_dist.index(max(test_dist))
			estimatedclass.append(test_dist.index(max(test_dist)))
			
			if estcls == class_label:
				accura.append(1)
				class_accura += 1
			else:
				accura.append(0)
		return (accura,estimatedclass)

	accura, estimatedclass = test_funxction(test,forest)

	def testing_pr(test, accura, estimatedclass):
		true = test["label"].tolist()
		for i in range(len(test)):
			print("ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n" %(i+1, estimatedclass[i], true[i], accura[i]))
		print('classification accuracy=%6.4f\n'% (sum(accura)/len(test)*100))
	
	testing_pr(test,accura,estimatedclass)
	def testing_print(test,forest):
		tree_id = 1
		for tree_here in forest:
			root =  tree_here
			queue = [root]
			node_id = 1
			while(queue):
			    node = queue[0]
			    if isinstance(node,Leaf):
			        print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n' %(tree_id, node_id, -1, -1, 0))
			    else:
			        print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n' %(tree_id, node_id, (node.data+1), node.threshold, node.gain))
			        if node.left is not None:
			            queue.append(node.left)
			        if node.right is not None:
			            queue.append(node.right)
			    queue.pop(0)
			    node_id += 1
			tree_id = tree_id  + 1
		

decision_tree(sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]))