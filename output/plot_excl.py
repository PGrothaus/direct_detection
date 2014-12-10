import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.interpolate

###Load data
###(we want to plot an exclusion band rather than one single line
### because we have background uncertainties;
### i.e. make two different runs with the flux values
### shifted up and down by one sigma to find the maximal sensitivty.
### This then gives an exclusion band)
###
data_0=np.loadtxt('June4/Xe_down_erec_1sigma.txt')
data_1=np.loadtxt('June4/Xe_up_erec_1sigma.txt')

data_2=np.loadtxt('June4/Xe_down_erec_3sigma.txt')
data_3=np.loadtxt('June4/Xe_up_erec_3sigma.txt')

data_4=np.loadtxt('June4/Xe_down_angle_1sigma.txt')
data_5=np.loadtxt('June4/Xe_up_angle_1sigma.txt')

data_6=np.loadtxt('June4/Xe_down_angle_3sigma.txt')
data_7=np.loadtxt('June4/Xe_up_angle_3sigma.txt')

data_10=np.loadtxt('../disclimit.txt')

mm=data_10[:,0]
ss=data_10[:,1]

for jj in range (8):
	if jj==0:
		data=data_0
	elif jj==1:
		data=data_1
	elif jj==2:
		data=data_2
	elif jj==3:
		data=data_3
	elif jj==4:
		data=data_4
	elif jj==5:
		data=data_5
	elif jj==6:
		data=data_6
	elif jj==7:
		data=data_7

	mass=data[:,2]
	sigma=data[:,3]
	cl1=data[:,5]
	cl2=data[:,6]

	sigma1_out,sigma2_out,mass_out=np.array([]),np.array([]),np.array([])
	
	mem1,mem2=1,1
	ii=0
	while ii<len(mass):
		cl1_arr,cl2_arr=np.array([]),np.array([])
		sig1,sig2=np.array([]),np.array([])
		mass0,mass_new=mass[ii],mass[ii]
		mem1,mem2=0,0
		spec1,spec2=False,False
		mass_out=np.concatenate((mass_out,np.array([mass_new])))
        #for each dark matter mass, sort the cross sections that were scanned
        #Then, interpolate the overlap to find exclusion limit
		while mass_new==mass0:
			cl1_arr=np.concatenate((cl1_arr,np.array([cl1[ii]])))
			cl2_arr=np.concatenate((cl2_arr,np.array([cl2[ii]])))
			sig1=np.concatenate((sig1,np.array([sigma[ii]])))
			tmp=zip(cl1_arr,sig1)
			tmp.sort(key= lambda tup:tup[1])
			cl1_arr,sig1=zip(*tmp)
			sig2=np.concatenate((sig2,np.array([sigma[ii]])))
			tmp=zip(cl2_arr,sig2)
			tmp.sort(key=lambda tup:tup[1])
			cl2_arr,sig2=zip(*tmp)
			ii+=1
			if ii==len(mass):
				break
			mass_new=mass[ii]

		print mass0	
		cl1_arr=cl1_arr[::-1]
		cl2_arr=cl2_arr[::-1]
		sig1=sig1[::-1]
		sig2=sig2[::-1]

		#print cl1_arr,sig1
		if jj<2:
			sig_ipl=scipy.interpolate.interp1d(cl2_arr,sig2,kind='linear')
			sigma2_out=np.concatenate((sigma2_out,np.array([sig_ipl(0.1)])))
		if jj>1 and jj<4:
			sig_ipl=scipy.interpolate.interp1d(cl2_arr,sig2,kind='linear')
			sigma2_out=np.concatenate((sigma2_out,np.array([sig_ipl(0.00135)])))
		if jj>3 and jj<6:
			sig_ipl=scipy.interpolate.interp1d(cl1_arr,sig2,kind='linear')
			sigma2_out=np.concatenate((sigma2_out,np.array([sig_ipl(0.1)])))
		if jj>5 and jj<8:
			sig_ipl=scipy.interpolate.interp1d(cl1_arr,sig2,kind='linear')
			sigma2_out=np.concatenate((sigma2_out,np.array([sig_ipl(0.00135)])))

	
	fig=plt.subplot(111)

	if jj==0:
		plt.plot(mm,ss,'-k',alpha=0.2,linewidth=8,color='0.2')
		sigma2_out=0.9*sigma2_out
		plt.plot(mass_out,sigma2_out,linestyle='dashed',color='0.3')
	if jj==1:
		sigma2_out=1.1*sigma2_out
		plt.plot(mass_out,sigma2_out,linestyle='dashed',color='0.3')
	if jj==2:
		sigma2_out=0.9*sigma2_out
		plt.plot(mass_out,sigma2_out,linestyle='dashed',color='0.1')
	if jj==3:
		sigma2_out=1.1*sigma2_out
		plt.plot(mass_out,sigma2_out,linestyle='dashed',color='0.1')
	if jj==4:
		sigma2_out=0.9*sigma2_out
		plt.plot(mass_out,sigma2_out,linestyle='dotted',color='0.3')
	if jj==5:
		sigma2_out=1.1*sigma2_out
		plt.plot(mass_out,sigma2_out,linestyle='dotted',color='0.3')
	if jj==6:
		sigma2_out=0.9*sigma2_out
		plt.plot(mass_out,sigma2_out,linestyle='dotted',color='0.1')
	if jj==7:
		sigma2_out=1.1*sigma2_out
		plt.plot(mass_out,sigma2_out,linestyle='dotted',color='0.1')
	plt.xscale('log')
	plt.xlabel(r'$m_{\rm DM} \, [ \rm{GeV} ] $',fontsize=22)
	plt.yscale('log')
	plt.ylabel(r'$ \sigma_p\, [ \rm{cm}^2 ]$',fontsize=22)
	plt.xlim(3.5,1000)
	plt.ylim(1.E-50,1.E-44)


	if jj==0:
		fig.text(100,1.E-45,r'$\rm{Xe}$',fontsize=25)
		fig.text(70,3.5E-46,r'$E_{\rm thr}=2\,{\rm keV}$',fontsize=20)
	if jj==0:
		mass_out1=mass_out
		upe1=sigma2_out
	if jj==1:
		mass_out2=mass_out
		downe1=sigma2_out
	if jj==2:
		mass_out3=mass_out
		upe3=sigma2_out
	if jj==3:
		mass_out4=mass_out
		downe3=sigma2_out
	if jj==4:
		mass_out5=mass_out
		upa1=sigma2_out
	if jj==5:
		mass_out6=mass_out
		downa1=sigma2_out
	if jj==6:
		mass_out7=mass_out
		upa3=sigma2_out
	if jj==7:
		mass_out8=mass_out
		downa3=sigma2_out

fig.fill_between(mass_out1,downe1,upe1,facecolor='red',interpolate=True,alpha=0.25,edgecolor='None')
fig.fill_between(mass_out3,downe3,upe3,facecolor='red',interpolate=True,alpha=0.75,edgecolor='red')
fig.fill_between(mass_out5,downa1,upa1,facecolor='green',interpolate=True,alpha=0.25,edgecolor='None')
fig.fill_between(mass_out7,downa3,upa3,facecolor='green',interpolate=True,alpha=0.75,edgecolor='green')
fig.tick_params(axis='both',labelsize=18)

plt.show()










