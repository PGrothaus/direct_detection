import numpy as np
import matplotlib.pyplot as plt
from dm_functions import *
from neutrino_functions import *
from constants import *
###################
#drawing functions#
###################
color=('blue','green','red','purple','orange','black')

def draw_exclusion(E_thr,T,Npoints=500):
    plt.figure(10)
    for var in range (len(E_thr)):
        nu_evts_tot=0.
        for ii in range (6):
            nu_evts_tot+=get_n_events_solar(E_thr[var],ii)
        for ii in range (4):
            nu_evts_tot+=get_n_events_atmo(E_thr[var],ii)
        nT=1./T/nu_evts_tot
        M=nT*A/N0#g
        
        print '\n','threshold energy:',E_thr[var],'keV, detector mass',M/1.e3,'kg,'\
                ,'exposure time:',T/days,'days'
        mDM=np.logspace(0,3,Npoints)
        sig=[]
        plt.xscale('log')
        plt.xlabel('WIMP mass [GeV]')
        plt.yscale('log')
        plt.ylabel('WIMP nucleon cross section [cm^2]')    
        for jj in range (Npoints):
            if(v_min(mDM[jj],E_thr[var])>1.e9 and v_min(mDM[jj],upper_threshold)>1.e9):
                sig.append(1.e20)
                continue
            factor=2.*mDM[jj]*red_mass(mDM[jj])*red_mass(mDM[jj])/rho0/M/T/A/A*0.1*GeVkg*GeVkg/keVJ
            sig.append(factor/sigmaDM(mDM[jj],E_thr[var]))
        plt.plot(mDM,sig)
    plt.show(10)

def draw_spectra():
    b7_norm=4.84E9
    ax=plt.figure(1)
    plt.xlim(1.e-1,1.e3)
    plt.xscale('log')
    plt.xlabel(r'$E_{\nu}\,[MeV]$',fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(1.e-4,1.e13)
    plt.yscale('log')
    plt.ylabel(r'$\frac{dN}{dE_{\nu}}\, \left[(cm^2\, s\, MeV)^{-1}\right]$',fontsize=22)
    for jj in range(9):
        data=get_spectrum_solar(jj)
        energy,spectrum,norm=data
	if jj==0:
		plt.plot(energy[:],spectrum[:]*norm,color='0.8',label=r'$pp$')
	if jj==1:
		plt.plot(energy[:],spectrum[:]*norm,'r-.',linewidth=2,label=r'$hep$')
	if jj==2:
		plt.plot(energy[:],spectrum[:]*norm,'r-',linewidth=2,label=r'$^8B$')
	if jj==3:
		plt.plot(energy[:],spectrum[:]*norm,color='0.6',label=r'$^{13}N$')
	if jj==4:
		plt.plot(energy[:],spectrum[:]*norm,color='0.4',label=r'$^{15}O$')
	if jj==5:
		plt.plot(energy[:],spectrum[:]*norm,color='0.2',label=r'$^{17}F$')
	if jj==6:
	        plt.plot(energy[:],spectrum[:]*norm/1000.,linestyle='-',color='0.5',label=r'$^7Be_{861.3}$')
	if jj==7:
		plt.plot(energy[:],spectrum[:]*norm/1000.,linestyle='-.',color='0.5',label=r'$^7Be_{384.3}$')
	if jj==8:
	        plt.plot(energy[:],spectrum[:]*norm/1000.,linestyle=':',color='0.5',label=r'$pep$')

    for jj in range(4):
        data=get_spectrum_atmo(jj)
        energy,spectrum=data
	if jj==0:
	        plt.plot(energy[:],spectrum[:],color='b',linestyle='solid',linewidth=2,label=r'$e$')
	if jj==1:
                plt.plot(energy[:],spectrum[:],color='b',linestyle='dashed',linewidth=2,label=r'$\bar{e}$')
	if jj==2:
                plt.plot(energy[:],spectrum[:],color='b',linestyle='dashdot',linewidth=2,label=r'$\mu$')
	if jj==3:
                plt.plot(energy[:],spectrum[:],color='b',linestyle='dotted',linewidth=2,label=r'$\bar{\mu}$')
    for jj in range(3):
        data=get_spectrum_dsnb(jj)
        energy,spectrum=data
	if jj==0:
	        plt.plot(energy[:],spectrum[:],'g:',linewidth=2,label=r'$3\,MeV$')
	if jj==1:
                plt.plot(energy[:],spectrum[:],'g-.',linewidth=2,label=r'$5\,MeV$')
	if jj==2:
                plt.plot(energy[:],spectrum[:],'g-',linewidth=2,label=r'$8\,MeV$')
    plt.legend(fontsize=18,ncol=2)
    plt.show(1)

#draw_spectra()

def draw_diffrate(M,T):
    nT=M*N0/A
    plt.figure(2)
    plt.xlim(1.e-3,1.e2)
    plt.xscale('log')
    plt.xlabel('E_rec [keV]')
    plt.ylim(1.e-5,1.e8)
    plt.yscale('log')
    ylabel='differential rate [1/('+str(M/1.e6)+'t '+str(T/yrs)+' yrs keV)]'
    plt.ylabel(ylabel)
    for jj in range(6):
        data=get_rate_solar(jj)
        energy,diffrate,totrate=data
        plt.plot(energy[:],nT*T*diffrate[:],color=color[jj])
    for jj in range(4):
        data=get_rate_atmo(jj)
        energy,diffrate,totrate=data
        plt.plot(energy[:],nT*T*diffrate[:],color=color[jj])
    plt.show(2)

def draw_totrate(M,T):
    nT=M*N0/A
    plt.figure(3)
    plt.xlim(1.e-3,1.e2)
    plt.xscale('log')
    plt.xlabel(r'$E_{\rm thr} [keV]$',fontsize=35)
    plt.ylim(1.e-4,1.e4)
    plt.yscale('log')
    plt.ylabel(r'$b \, [(t\, yrs)^{-1}]$',fontsize=35)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    for jj in range(9):
        data=get_rate_solar(jj)
        energy,norm,spectrum=data
	if jj==0:
		plt.plot(energy[:],nT*T*spectrum[:],color='0.8',linewidth=2,label=r'$pp$')
	if jj==1:
		plt.plot(energy[:],nT*T*spectrum[:],'r-.',linewidth=3,label=r'$hep$')
	if jj==2:
		plt.plot(energy[:],nT*T*spectrum[:],'r-',linewidth=3,label=r'$^8B$')
	if jj==3:
		plt.plot(energy[:],nT*T*spectrum[:],color='0.6',linewidth=2,label=r'$^{13}N$')
	if jj==4:
		plt.plot(energy[:],nT*T*spectrum[:],color='0.4',linewidth=2,label=r'$^{15}O$')
	if jj==5:
		plt.plot(energy[:],nT*T*spectrum[:],color='0.2',linewidth=2,label=r'$^{17}F$')
	if jj==6:
	        plt.plot(energy[:],nT*T*spectrum[:],linestyle='-',color='0.5',linewidth=2,label=r'$^7Be_{861.3}$')
	if jj==7:
		plt.plot(energy[:],nT*T*spectrum[:],linestyle='-.',color='0.5',linewidth=2,label=r'$^7Be_{384.3}$')
	if jj==8:
	        plt.plot(energy[:],nT*T*spectrum[:],linestyle=':',color='0.5',linewidth=2,label=r'$pep$')

    for jj in range(4):
        data=get_rate_atmo(jj)
        energy,norm,spectrum=data
	if jj==0:
	        plt.plot(energy[:],nT*T*spectrum[:],color='b',linestyle='solid',linewidth=3,label=r'$e$')
	if jj==1:
                plt.plot(energy[:],nT*T*spectrum[:],color='b',linestyle='dashed',linewidth=3,label=r'$\bar{e}$')
	if jj==2:
                plt.plot(energy[:],nT*T*spectrum[:],color='b',linestyle='dashdot',linewidth=3,label=r'$\mu$')
	if jj==3:
                plt.plot(energy[:],nT*T*spectrum[:],color='b',linestyle='dotted',linewidth=3,label=r'$\bar{\mu}$')
    for jj in range(3):
        data=get_rate_dsnb(jj)
        energy,norm,spectrum=data
	if jj==0:
	        plt.plot(energy[:],nT*T*spectrum[:],'g:',linewidth=3,label=r'$3\,MeV$')
	if jj==1:
                plt.plot(energy[:],nT*T*spectrum[:],'g-.',linewidth=3,label=r'$5\,MeV$')
	if jj==2:
                plt.plot(energy[:],nT*T*spectrum[:],'g-',linewidth=3,label=r'$8\,MeV$')
    leg=plt.legend(bbox_to_anchor=(0, 0, 1, 1),loc=1,fontsize=20,ncol=6,mode="expand", borderaxespad=0.,fancybox='True')
    leg.get_frame().set_alpha(0.5)
    plt.show(3)

draw_spectra()
draw_totrate(1.e6,1.*yrs)
