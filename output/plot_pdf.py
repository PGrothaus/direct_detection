import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import matplotlib.cm as cm
import math

xmin=2.
xmax=100.
for kk in range (3):
	if kk==0:
		name='FINAL/pdfs/2dpdf_CF4_5.0keV_DM_6.0_6.txt'
		#name='FINAL/pdfs/2dpdf_Xe_2.0keV_no_nu_uncertain_DM_30.0_6.txt'
	if kk==1:
		#name='FINAL/pdfs/2dpdf_CF4_5.0keV_DM_6.0_0.txt'
		name='FINAL/pdfs/2dpdf_CF4_5.0keV_DM_40.0_6.txt'
		#name='FINAL/pdfs/2dpdf_Xe_2.0keV_no_nu_uncertain_DM_100.0_6.txt'
	if kk==2:
		#name='FINAL/pdfs/2dpdf_CF4_5.0keV_nu.txt'
		name='FINAL/pdfs/2dpdf_CF4_5.0keV_DM_100.0_6.txt'
		#name='FINAL/pdfs/2dpdf_Xe_2.0keV_no_nu_uncertain_DM_1000.0_6.txt'

	data=np.loadtxt(name)
	
	E_thr=5.
	upper_threshold=100.
	N_erec,N_theta=31,16
	Ne=500
	
	E_rec_bin=np.linspace(E_thr,upper_threshold,N_erec)
	theta_bin=np.linspace(-1.,1.,N_theta)
	
	x_edge,y_edge=[],[]
	for ii in range (N_erec-1):
	        x_edge.append(0.5*(E_rec_bin[ii]+E_rec_bin[ii+1]))
	for ii in range (N_theta-1):
	        y_edge.append(0.5*(theta_bin[ii]+theta_bin[ii+1]))
	
	###shift to cover the whole grid when interpolating
	###
	x_edge[0]=E_thr
	x_edge[-1]=upper_threshold
	y_edge[0]=-1.
	y_edge[-1]=1.
	dtheta=1.*(y_edge[-1]-y_edge[0])/N_theta
	derec=1.*(x_edge[-1]-x_edge[0])/N_erec
	#
	Erec=data[:,0]
	theta=data[:,1]
	pdf=np.log10(data[:,2])
	
	pdf=np.reshape(pdf,(N_erec-1,N_theta-1))
	f_ipl=RectBivariateSpline(x_edge,y_edge,pdf,kx=1,ky=1)
	ee=np.linspace(E_thr,xmax,Ne)
	tt=np.linspace(-1,1,Ne)
	ee=np.tile(ee,Ne)
	tt=np.tile(tt,Ne)
	np.random.shuffle(tt)
	pdf_val=f_ipl.ev(ee,tt)

	fig1=plt.figure(2*kk+1)
	im=plt.scatter(ee,tt,c=pdf_val,edgecolors='none',s=30,cmap=cm.YlOrRd)
	cbar=fig1.colorbar(im)
	if kk<2:
		cbar.set_label(r'$\log_{10} \, \rm{p}_{\rm DM}$',fontsize=20)
	if kk==2:
		cbar.set_label(r'$\log_{10} \, \rm{p}_\nu$',fontsize=20)
	cbar.ax.tick_params(labelsize=16)
	plt.clim(-5,0)
	
	Erec=data[:,0]
	theta=data[:,1]
	pdf=np.log10(data[:,2])

	plt.ylim(-1,1)
	plt.xlim(E_thr,xmax)
	plt.xlabel(r'$E_{\rm rec} [{\rm keV}]$',fontsize=20)
	plt.ylabel(r'$\cos\,\theta_{\rm sun}$',fontsize=20)
	CS=plt.tricontour(Erec,theta,pdf,[0,-1,-2,-3,-4],linewidth=1,colors='k',vmin=-6,vmax=1,linestyles='solid',alpha=0.5)
	plt.clabel(CS, inline=1, fontsize=18)
	plt.tick_params(axis='both',labelsize=16)

plt.show(all)

#
#
#
#
#
##for kk in range (2):
##	if kk==0:
##		name1='FINAL/2dpdf_CF4_5.0keV_DM_6.0_6.txt'
##	if kk==1:
##		name2='FINAL/2dpdf_CF4_5.0keV_nu.txt'
##
##data1=np.loadtxt(name1)
##data2=np.loadtxt(name2)
##
##E_thr=5.
##upper_threshold=100.
##N_erec,N_theta=31,16
##Ne=500
##
##s=10.
##b=500.
##
##E_rec_bin=np.linspace(E_thr,upper_threshold,N_erec)
##theta_bin=np.linspace(-1.,1.,N_theta)
##
##x_edge,y_edge=[],[]
##for ii in range (N_erec-1):
##        x_edge.append(0.5*(E_rec_bin[ii]+E_rec_bin[ii+1]))
##for ii in range (N_theta-1):
##        y_edge.append(0.5*(theta_bin[ii]+theta_bin[ii+1]))
##
#####shift to cover the whole grid when interpolating
#####
##x_edge[0]=E_thr
##x_edge[-1]=upper_threshold
##y_edge[0]=-1.
##y_edge[-1]=1.
##dtheta=1.*(y_edge[-1]-y_edge[0])/N_theta
##derec=1.*(x_edge[-1]-x_edge[0])/N_erec
###
##Erec=0.5*(data1[:,0]+data2[:,0])
##theta=0.5*(data1[:,1]+data2[:,1])
##pdf=np.log10((s*data1[:,2]+b*data2[:,2])/(b+s))
##
##pdf=np.reshape(pdf,(N_erec-1,N_theta-1))
##f_ipl=RectBivariateSpline(x_edge,y_edge,pdf,kx=1,ky=1)
##ee=np.linspace(E_thr,40,Ne)
##tt=np.linspace(-1,1,Ne)
##ee=np.tile(ee,Ne)
##tt=np.tile(tt,Ne)
##np.random.shuffle(tt)
##pdf_val=f_ipl.ev(ee,tt)
##
##fig1=plt.figure(1000)
##im=plt.scatter(ee,tt,c=pdf_val,edgecolors='none',s=30,cmap=cm.YlOrRd)
##cbar=fig1.colorbar(im)
##cbar.set_label(r'$\log_{10} \, \rm{p}_{\rm S+B}$',fontsize=20)
##cbar.ax.tick_params(labelsize=16)
##plt.clim(-5,0)
##
##Erec=data1[:,0]
##theta=data1[:,1]
##pdf=np.log10((s*data1[:,2]+b*data2[:,2])/(s+b))
#
#plt.ylim(-1,1)
#plt.xlim(E_thr,40)
#plt.xlabel(r'$E_{\rm rec} [\rm{keV}]$',fontsize=20)
#plt.ylabel(r'$\cos\,\theta_{\rm sun}$',fontsize=20)
#CS=plt.tricontour(Erec,theta,pdf,[0,-1,-2,-3,-4],linewidth=1,colors='k',vmin=-6,vmax=1,linestyles='solid',alpha=0.5)
#plt.clabel(CS, inline=1, fontsize=18)
#plt.tick_params(axis='both',labelsize=16)
#plt.show(all)
 
