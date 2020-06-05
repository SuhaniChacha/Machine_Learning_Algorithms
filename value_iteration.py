import pandas as pd
import numpy as np
import math
import sys

def value_iteration(environment_file,non_terminal_reward,gamma,k):
	def load_data(data_file):
		df = pd.read_csv(data_file, sep = ',',header = None)
		return df
	def make_states(data_list):
		states = np.empty(data_list.shape).tolist()

		for i in range(0,len(data_list)):
		    ij = data_list[i]
		    for j in range(0,len(ij)):
		        states[i][j] = [i,j]
		return states
	def make_action(data_list):
		actions = np.empty(data_list.shape).tolist()
		a1 = ["up","right","left","down"]
		a2 = ["None","None","None","None"]
		for i in range(0,len(data_list)):
		    ij = data_list[i]
		    for j in range(0,len(ij)):
		        if(data_list[i][j] == "X") or (data_list[i][j] == "1.0") or (data_list[i][j] == "-1.0"):
		            actions[i][j] = a2
		        else:
		            actions[i][j] = a1
		return actions
	def make_rewards(rewards,non_terminal_reward):
		rewards = np.empty(data_list.shape).tolist()
		b1 = ""+non_terminal_reward
		b2 = "0"
		for i in range(0,len(data_list)):
		    ij = data_list[i]
		    for j in range(0,len(ij)):
		        if(data_list[i][j] == "X"):
		            rewards[i][j] = b2
		        elif (data_list[i][j] == "1.0"):
		            rewards[i][j] = "1.0"
		        elif (data_list[i][j] == "-1.0"):
		            rewards[i][j] = "-1.0"
		        else:
		            rewards[i][j] = b1
		return rewards
	def cal(s,list_here,act,u):
	#     print(u)
	    sumc = []
	    
	    for i in range(0,len(list_here)):
	        j = list_here[i][0]
	        actio = list_here[i][1]
	       
	        if (j == s) and (actio == act):
	            #print(list_here[i])
	            s_dash_r = list_here[i][2][0][0]
	            s_dash_c = list_here[i][2][0][1]
	            p_dash = list_here[i][2][1]
	            u_h = u[s_dash_r][s_dash_c]
	#             print('u_h',u_h,'p_dash',p_dash)
	               
	            v = u_h*p_dash
	#             print("vvvvv", v)
	            sumc.append(v)
	            
	    #print(sumc,sum(sumc))
	    return sum(sumc)

	def val_itr(states,act,list_here,rewards,discount_factor,k):
	    n1 = len(states)
	    n2 = len(states[0])
	    N = (n1,n2)
	   # print(N)

	    u_dash = np.zeros(N)
	    #print(u_dash)
	    for i in range(0,k):
	        u = u_dash.copy()
	        #print('u',u)
	        for isd in range(0,len(states)):
	            for ij in range(0,len(states[0])):
	                s = states[isd][ij]
	#                 print(s)
	                if (data_list[isd][ij]!="1.0") and (data_list[isd][ij]!="-1.0") and (data_list[isd][ij]!="X"):
	                    #print(isd,ij,u_dash[isd][ij])
	                    action_max = []
	                    for a in act[isd][ij]:
	#                         print('a',a)
	                        j = cal(s,list_here,a,u)
	                        action_max.append(j)
	                    m = max(action_max)
	#                     print("mmm ---- " ,m)
	                    u_dash[isd][ij] = float(rewards[isd][ij]) + (float(discount_factor)*float((max(action_max))))
	                elif (data_list[isd][ij]=="1.0"):
	                    u_dash[isd][ij] = 1.0
	                elif (data_list[isd][ij]=="-1.0"):
	                    u_dash[isd][ij] = -1.0
	                elif (data_list[isd][ij]=="X"):
	                    u_dash[isd][ij] = 0
	    return u_dash

	def trans_pro(s,tr, act, po, states):
	#     print(s, act, p)
	    probalem = []
	    p = []
	    l = []
	    r = len(states)
	    c = len(states[0])
	#     print(s[0],s[1])
	    for ij in range(0,len(act)):
	        i =act[ij]
	        if i == "up":
	            row = s[0]-1
	            col = s[1]
	        elif i == "right":
	            row = s[0]
	            col = s[1]+1
	        elif i == "down":
	            row = s[0]+1
	            col = s[1]
	        elif i == "left":
	            row = s[0]
	            col = s[1]-1
	        else:
	            row = s[0]
	            col = s[1]
	        if (row!= -1) and (col!= -1) and (row!= r) and (col!= c)  and ([row,col] != [1,1]):
	            st = states[row][col]
	        else:
	            st = states[s[0]][s[1]]
	        #print(s,tr,act,i,st)
	        probalem.append([s,tr,act,i,st])
	        l.append([st,po[ij]])
	    #print("ppp",probalem)
	    df = pd.DataFrame(l)
	    ro = []
	    i = []
	    if(probalem[0][1]!="None"):
	        for index,row in df.iterrows():
	            if(row[0] not in ro):
	                ro.append(row[0])
	                i.append(index)

	            else:
	                pre_index = ro.index(row[0])
	                a = l[pre_index][1]
	                b = l[index][1]
	                l[pre_index][1] = a+b
	                del l[index]
	                #print(a,b)
	    else:
	        #l = l[0]
	        l[0][1] = 1
	        l = l[0]
	        
	    #print(l)
	    return l

	data = load_data(environment_file)
	data_list = data.values
	states = make_states(data_list)
	actions = make_action(data_list)
	rewards = make_rewards(data_list,non_terminal_reward)
	list_here = []
	trans_m = []
	a =[]
	for i in range(0,len(states)):
	    ij = states[i]
	    for j in range(0,len(ij)):
	        s = ij[j]
	        a = actions[i][j]
	        
	        for tr in a:
	            act = []
	            if(tr == "up"):
	                act = ["left","up","right"]
	            elif(tr == "right"):
	                act = ["up","right","down"]
	            elif(tr == "down"):
	                act = ["right","down","left"]
	            elif(tr == "left"):
	                act = ["down","left","up"]                                    
	            else:
	                act = ["None","None","None"]
	            p = [0.1,0.8,0.1]
	            t = trans_pro(s,tr,act,p,states)
	            if(tr!="None"):
	                for it in t:
	                    list_here.append([s,tr,it])
	            else:
	                 list_here.append([s,tr,t])
	df = pd.DataFrame(list_here)
	df_l = df.values
	u_dash_list = val_itr(states,actions,list_here,rewards,gamma,int(k))
	
	print("Utility Matrix: ")
	for j in range(0,len(u_dash_list)):
		for k in range(0,len(u_dash_list[0])):
			print("%6.3f," %u_dash_list[j][k],end ="")
		print()
	print()
value_iteration(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])