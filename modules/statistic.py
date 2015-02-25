from math import *
from scipy import *
import numpy as np

def evaluate_Q( t_edges, t_nu, E_rec_nu, cos_nu, f_array, Pb_ipl,\
              f_array_noangle, Pb_noangle_ipl, pdf_DM_t, pdf_nu_t,\
              NN_nu, n_nu_arr, N_Q, N_tt):
    dif=np.zeros((len(t_edges),len(t_nu)))
    for ii in range(len(t_edges)):
        dif[ii]=abs(t_edges[ii]-t_nu)
###Bug removed!
###Wrong calculation of time seperation
#    dif=np.reshape(dif,(len(t_nu),len(t_edges)))

    id1_nu=np.argmin(dif,axis=0)

    t0_nu=t_edges[id1_nu]
    id2_nu=np.where(id1_nu==N_tt-1,id1_nu-1,0)
    id2_nu=np.where(id1_nu==0,1,id2_nu)
    
    id2_nu_tmp1=np.where(id2_nu==0,id1_nu+1,id2_nu)
    t1_nu=t_edges[id2_nu_tmp1]
    
    id2_nu_tmp2=np.where(id2_nu==0,id1_nu-1,id2_nu)
    t2_nu=t_edges[id2_nu_tmp2]
    
    id2_nu=np.where(abs(t1_nu-t_nu)>abs(t2_nu-t_nu),\
                        id2_nu_tmp2,id2_nu)
    id2_nu=np.where(abs(t1_nu-t_nu)<abs(t2_nu-t_nu),\
                        id2_nu_tmp1,id2_nu)
    d1=abs(t_nu-t_edges[id1_nu])
    d2=abs(t_nu-t_edges[id2_nu])
    pdf1,pdf2=np.zeros(NN_nu),np.zeros(NN_nu)
    for ii in range (N_tt):
        pdf1=np.where(id1_nu==ii,\
                f_array[ii].ev(E_rec_nu,cos_nu),pdf1)
        pdf2=np.where(id2_nu==ii,\
                f_array[ii].ev(E_rec_nu,cos_nu),pdf2)
    prob_nu_B_angle=Pb_ipl.ev(E_rec_nu,cos_nu)
    prob_nu_S_angle=pdf1+d1/(d1+d2)*(pdf2-pdf1)
    for ii in range (N_tt):
        pdf1=np.where(id1_nu==ii,\
                f_array_noangle[ii](E_rec_nu),pdf1)
        pdf2=np.where(id2_nu==ii,\
                f_array_noangle[ii](E_rec_nu),pdf2)
    prob_nu_B_erec=Pb_noangle_ipl(E_rec_nu)
    prob_nu_S_erec=pdf1+d1/(d1+d2)*(pdf2-pdf1)

    prob_nu_S_time=pdf_DM_t(t_nu)
    prob_nu_B_time=pdf_nu_t(t_nu)

    prob_nu_S_time=np.tile(prob_nu_S_time,N_Q)
    prob_nu_B_time=np.tile(prob_nu_B_time,N_Q)
    prob_nu_S_angle=np.tile(prob_nu_S_angle,N_Q)
    prob_nu_B_angle=np.tile(prob_nu_B_angle,N_Q)
    prob_nu_S_erec=np.tile(prob_nu_S_erec,N_Q)
    prob_nu_B_erec=np.tile(prob_nu_B_erec,N_Q)

    n_split=np.cumsum(n_nu_arr)
    n_split=np.delete(n_split,-1)
    prob_nu_B_time = np.split(prob_nu_B_time,n_split)
    prob_nu_S_time = np.split(prob_nu_S_time,n_split)
    prob_nu_B_angle = np.split(prob_nu_B_angle,n_split)
    prob_nu_S_angle = np.split(prob_nu_S_angle,n_split)
    prob_nu_B_erec = np.split(prob_nu_B_erec,n_split)
    prob_nu_S_erec = np.split(prob_nu_S_erec,n_split)

    return prob_nu_B_time, prob_nu_S_time, prob_nu_B_erec,  prob_nu_S_erec,\
           prob_nu_B_angle, prob_nu_S_angle

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
