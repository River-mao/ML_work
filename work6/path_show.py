import numpy as np
import matplotlib.pyplot as plt

class path_show(object):

    def __init__(self,T,N,delta,i_star):
        self.T = T
        self.N = N
        self.state_list = np.zeros([self.T*self.N,2],dtype=int)
        self.delta = np.around(np.array(delta,dtype=float),5)
        self.i_star = np.array(i_star,dtype=int)

    def state_scatter(self):
        for t in range(self.T):
            for state in range(self.N):
                self.state_list[t*self.N+state] = np.array([t,state])+1

    def show(self):

        plt.title("The path")
        plt.xlabel("T")
        plt.ylabel("State")
        plt.yticks([i+1 for i in range(self.N)],[i+1 for i in range(self.N)])
        plt.xticks([i+1 for i in range(self.T)],[i+1 for i in range(self.T)])
        plt.scatter(self.state_list[:,0],self.state_list[:,1],s=200)
        plt.scatter([i+1 for i in range(self.T)],self.i_star,s=400,facecolors='none', edgecolors='r')
        for i in range(self.T):
            for state in range(self.N):
                plt.annotate(self.delta[i][state], xy=(self.state_list[:,0][i*self.N+state],self.state_list[:,1][i*self.N+state]))

        plt.gca().plot([i+1 for i in range(self.T)],self.i_star)
        plt.grid()
        plt.show()
