import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pprint import pprint 

#read and prepare the data 
df = pd.read_csv('ex_3.csv')

Y= df['Y']
Y=np.array(Y).reshape(-1,1)

X_ = df['X']
X = np.c_[X_,X_**2,X_**3,X_**4]  #Polynomial features
X=pd.DataFrame(X)
X_original=np.array(X)
#normalize features
'''
col = list(X.columns)
for i in range(len(col)):
    X[i] = (X[i]-X[i].mean())/(X[i]).std()
'''
X = (X - X.mean(axis=0))/X.std(axis=0)
X=np.array(X)

#hyper_parameters
alpha = .03
epochs = 1000
m = X.shape[0]

#parameters
b=0
w = np.zeros(X.shape[1]).reshape(-1,1)
#cost function 
def cost_value(m,y,y_p):
    cost = ((1/(2*m))*((y-y_p)**2)).sum()
    return cost




#loop to get the solution
cost_history=[]
for i in range(epochs):
    y_pred = np.dot(X,w) + b

    cost = cost_value(m,Y,y_pred)
    cost_history.append(cost)

    db = (1/(m)) * (y_pred-Y).sum()
    dw = (1/(m)) * (np.dot(X.T,(y_pred-Y)))
    b=b -alpha*db
    w=w- alpha*dw


#check for convergence 
plt.figure()
plt.plot(range(len(cost_history)),cost_history,color='k',label='convergence')
plt.scatter(np.argmin(cost_history),np.min(cost_history),label=f'{np.min(cost_history)} @{np.argmin(cost_history)}')
plt.ylabel('cost_history')
plt.xlabel('epochs')
plt.legend()
plt.grid(True)
plt.show()

#pprint(f'y={b}+{w[0]}x+{w[1]}x^2+{w[2]}x^3+{w[3]}x^4')

# Reverse scaling to get weights and bias for original features
mean = X_original.mean(axis=0)
std = X_original.std(axis=0)

# Transform weights back to original scale (real scale)
w_real = w.flatten() / std  # Divide each weight by its corresponding feature's std deviation

# Adjust the bias term
b_real = b - np.sum((w.flatten() * mean) / std)

# Print real-scale weights and bias
pprint(f"Real scale polynomial equation: y_pred_ = {b_real} + {w_real[0]}x + {w_real[1]}x^2 + {w_real[2]}x^3 + {w_real[3]}x^4")
#plot the original data 
y_pred_ =np.dot(X_original,w_real) + b_real 
plt.figure()
plt.plot(X_,Y,color='r',label='original data')
plt.plot(X_,y_pred_,label='predicted data',color='k',linewidth=3)
plt.ylabel('Y')
plt.xlabel('X')
plt.legend()
plt.grid(True)
plt.show()