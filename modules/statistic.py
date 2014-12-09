from math import *
from scipy import *
import numpy as np

def get_cl(hist_SB, hist_B, Q_grid_SB, Q_grid_B, N_bins, steps, accuracy):

    dqSB = Q_grid_SB[4] - Q_grid_SB[3]
    dqB = Q_grid_B[4] - Q_grid_B[3]

    normSB = np.sum( dqSB * hist_SB )
    normB  = np.sum( dqB  * hist_B )

    hist_SB = hist_SB/normSB
    hist_B = hist_B/normB

    Q_min=np.min(Q_grid_SB)
    Q_max=np.max(Q_grid_B)

    Q_int=np.linspace(Q_min,Q_max,steps)
    dQ_int=1.*(Q_max-Q_min)/steps
   
    ###The distributions with correct normalisation
    ###
    def pdf_B(Qv):
        if Qv<min(Q_grid_B) or Qv>max(Q_grid_B) \
                            or Qv==max(Q_grid_B):
            return 0.
        Qv=np.array([Qv,Qv])
        id=np.digitize(Qv,bins=Q_grid_B)
        id=id[0]-1
        return hist_B[id]

    def pdf_SB(Qv):
        if Qv<min(Q_grid_SB) or Qv>max(Q_grid_SB) \
                             or Qv==max(Q_grid_SB) :
            return 0.
        Qv=np.array([Qv,Qv])
        id=np.digitize(Qv,bins=Q_grid_SB)
        id=id[0]-1
        return hist_SB[id]

    ###calculate overlap of both distributions
    ###
    cl=[]
    int_B=0.
    int_SB=0.
    f_B0=pdf_B(Q_int[0])
    f_SB0=pdf_SB(Q_int[0])
    for ii in range(steps-1):
        f_B1=pdf_B(Q_int[ii+1])
        f_SB1=pdf_SB(Q_int[ii+1])
        int_B+=dQ_int*0.5*(f_B0+f_B1)
        int_SB+=dQ_int*0.5*(f_SB0+f_SB1)
        f_B0=f_B1
        f_SB0=f_SB1
        
        if np.min(Q_grid_SB)<np.min(Q_grid_B):
            if(abs((1.-int_SB)-int_B)<accuracy):
                if(int_B>0.):
                    cl.append(abs(1.-int_SB))
                else:
                    cl.append(0.)
        else:
            if(abs((1.-int_B)-int_SB)<accuracy):
                if(int_SB>0.):
                        cl.append(abs(1.-int_B))
                else:
                        cl.append(0.)
        if(int_SB>0.995):
            break

    ###Test distributions
    ###
    #if print_info==0:
    #   qq=np.linspace(min(Q_grid_SB),max(Q_grid_B),250)
    #   plt.figure(2)   
    #   plt.hist(Q_B,bins=Q_grid_B,\
    #           facecolor='b',alpha=0.5,normed=1.)
    #   plt.hist(Q_SB,bins=Q_grid_SB,\
    #           facecolor='r',alpha=0.5,normed=1.)
    #   #for ii in range (len(qq)):
    #   #   plt.plot(qq[ii],pdf_B(qq[ii]),'bo')
    #   #   plt.plot(qq[ii],pdf_SB(qq[ii]),'ro')
    #   plt.show(all)

    if(len(cl)==0):
        cl_avg=0.
    else:
        cl_avg=1.*np.sum(cl)/len(cl)
    return cl_avg
