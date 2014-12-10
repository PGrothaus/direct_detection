import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,LogFormatter
import scipy.interpolate


###Load data here
###
data_1=np.loadtxt('June4/SENS_Xe_6GeV_angle_uncertain.txt')
data_2=np.loadtxt('June4/SENS_Xe_6GeV_erec_uncertain.txt')
#data_3=np.loadtxt('June4/SENS_Xe_30GeV_angle_uncertain.txt')
#data_4=np.loadtxt('June4/SENS_Xe_30GeV_erec_uncertain.txt')
#data_5=np.loadtxt('June4/SENS_Xe_1000GeV_both_uncertain.txt')
#data_6=np.loadtxt('June4/SENS_Xe_1000GeV_both_uncertain.txt')

###The discovery limits for the dark matter masses that we plot
###(here 6 GeV, 30 GeV and 1000 GeV
###
sigma_lim=np.array([3.8676e-45,3.8676e-45,8.2235e-50,
                    8.2235e-50,2.0071e-48,2.0071e-48])


fig=plt.figure(1)
ax1=plt.subplot(111)
ax2=ax1.twiny()
ax1.axvline(x=500,color='0.8')
ax1.set_yscale('log')

###loop over the dark matter masses
###and directional/non-directional cases
###
for jj in range (2):
    if jj==0:
        data=data_1
    if jj==1:
        data=data_2
    if jj==2:
        data=data_3
    if jj==3:
        data=data_4
    if jj==4:
        data=data_5
    if jj==5:
        data=data_6

    M_det_arr=data[:,0]
    nu_evts=data[:,1]
    mass=data[:,2]
    mass=mass[0]
    sigma=data[:,3]
    cl1=data[:,5]
    cl2=data[:,6]

    N_nu_out=np.array([])
    sigma1_out,sigma2_out=np.array([]),np.array([])
    M_det_out=np.array([])
    
    ii=0
    while ii<len(nu_evts):
        N_nu_0,N_nu_new=nu_evts[ii],nu_evts[ii]
        M_det=M_det_arr[ii]
        cl1_arr,cl2_arr=np.array([]),np.array([])
        sig1,sig2=np.array([]),np.array([])
        while N_nu_new==N_nu_0:  
            cl1_new=cl1[ii]
            cl2_new=cl2[ii]
            cl1_arr=np.concatenate((cl1_arr,np.array([cl1[ii]])))
            cl2_arr=np.concatenate((cl2_arr,np.array([cl2[ii]])))   
            sig1=np.concatenate((sig1,np.array([sigma[ii]])))
            sig2=np.concatenate((sig2,np.array([sigma[ii]])))
            tmp=zip(cl1_arr,sig1)
            tmp.sort(key= lambda tup:tup[1])
            cl1_arr,sig1=zip(*tmp)
            tmp=zip(cl2_arr,sig2)
            tmp.sort(key=lambda tup:tup[1])
            cl2_arr,sig2=zip(*tmp)
            
            cl1_arr=cl1_arr[::-1]
            cl2_arr=cl2_arr[::-1]
            sig1=sig1[::-1]
            sig2=sig2[::-1]

            ii+=1
            if ii==len(nu_evts):
                break
            N_nu_new=nu_evts[ii]
        print jj
        if jj==0 or jj==2 or jj==4:
            print cl1_arr
            print sig1
            sig1_ipl=scipy.interpolate.interp1d(cl1_arr,sig1,kind='linear')
            sigma1_out=np.concatenate(
                        (sigma1_out,np.array([sig1_ipl(0.00135)])))
        if jj==1 or jj==3 or jj==5:
            print cl2_arr
            print sig2
            sig2_ipl=scipy.interpolate.interp1d(cl2_arr,sig2,kind='linear')
            sigma2_out=np.concatenate(
                            (sigma2_out,np.array([sig2_ipl(0.00135)])))
        N_nu_out=np.concatenate((N_nu_out,np.array([N_nu_0])))
        M_det_out=np.concatenate((M_det_out,np.array([M_det]))) 
                                                                            
    if jj==0:
        ax1.plot(N_nu_out,sigma1_out,'-bo',markersize=4,linewidth=2)
        ax2.plot(M_det_out/1.E6,sigma1_out,'-bo',markersize=0.1,linewidth=0.2)
    if jj==1:
        ax1.plot(N_nu_out,sigma2_out,'-.bo',markersize=4,linewidth=1.6)
        ax2.plot(M_det_out/1.E6,sigma2_out,'-.bo',markersize=0.1,linewidth=0.2)
    if jj==2:
        ax1.plot(N_nu_out,sigma1_out,'-ro',markersize=4,linewidth=2)
        ax2.plot(M_det_out/1.E6,sigma1_out,'-ro',markersize=0.1,linewidth=0.2)
    if jj==3:
        ax1.plot(N_nu_out,sigma2_out,'-.ro',markersize=4,linewidth=1.6)
        ax2.plot(M_det_out/1.E6,sigma2_out,'-.ro',markersize=0.1,linewidth=0.1)
    if jj==4:
        ax1.plot(N_nu_out,sigma1_out,'-ko',markersize=4,linewidth=2)
        ax2.plot(M_det_out/1.E6,sigma1_out,'-ko',markersize=0.1,linewidth=0.1)
    if jj==5:
        ax1.plot(N_nu_out,sigma2_out,'-.ko',markersize=4,linewidth=1.6)
        ax2.plot(M_det_out/1.E6,sigma2_out,'-.ko',markersize=0.1,linewidth=0.1)
    c_ar=['blue','blue','red','red','black','black']
    ax2.hlines(sigma_lim[jj],0.1,1000000,color=c_ar[jj],
               linestyle='dashed',linewidth=0.8)


###Some things to make the plots look nice
###e.g. adding labels for target material
###
ax1.text(45,5.E-49,'Xe',fontsize=25)
ax1.text(25,2.E-49,r'$E_{\rm thr}=2\,{\rm keV}$',fontsize=20)

#ax1.text(45,5.5E-49,r'$CF_4$',fontsize=25)
#ax1.text(25,1.8E-49,r'$E_{\rm thr}=5\,{\rm keV}$',fontsize=20)

ax1.set_xlabel(r'$N_\nu$',fontsize=20)
ax1.set_xscale('log')
ax1.set_ylim(1.E-50,1.E-43)
ax1.set_xlim(10,100000)
ax1.set_ylabel(r'$\sigma_p \, [{\rm cm}^2]$',fontsize=20)
ax2.set_yscale('log')
ax2.set_ylim(1.E-50,1.E-43)
ax2.set_xlim(7345660.83583/1.E6,73456608358.3/1.E6)
#ax2.set_xlim(732128.471623/1.E6,7321284716.23/1.E6)
ax2.set_xlabel(r'$M_{\rm det} \, [{\rm ton - year} ]$',fontsize=20)
ax2.tick_params(axis='x',labelsize=16)
ax1.tick_params(axis='both',labelsize=16)
ax2.set_xscale('log')
plt.show()
