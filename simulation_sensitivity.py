import numpy as np
import math
from math import pi,log
import sys
#import matplotlib.pyplot as plt
import ephem as eph
import random as rnd
from dm_functions import *
from neutrino_functions import *
from constants import *
from create_lookupfiles import *
import scipy 
from scipy.interpolate import RectBivariateSpline,interp1d,SmoothBivariateSpline
from scipy.stats import norm

###############################################################################################################

###comparing my errorfunciotn against the implemented one
###
#aa=np.arange(-10.,10.,0.1)
#for ii in range(len(aa)):
#	print math.erf(aa[ii]),erf(aa[ii])

###For test modus of the 2dimensional pdf's
###	(test=1 tests the pdf's)
test=0
show_hist=0
print_info=0
if test==1:
        print ''
        print '#########################################################################'
        print 'TEST MODE'
        print '#########################################################################'
        print ''

###initialise observer, CygnusA and the Sun
###
line = "Cygnus A,f|J,19:59:28.4,40:44:01.0,1260,2000"
cygnA = ephem.readdb(line)
sun=ephem.Sun()
obs=eph.Observer()
obs.pressure=0.
obs.temp=0.
obs.lon='0.'
obs.lat='0.'
obs.elevation=0.
sun.compute(obs)
rnd.seed()

###flux uncertainties
###	(extrapolated ten years)
nu_sigma=np.array([0.000000001,0.0000001,0.00000001])#in fraction of N

###Choose values for simulation
###
N_sim=100				#num of sets of events to be generated for simulation
N_Q=1					#num of times to vary the fluxes when evaluating Q
factor=50				#num of loops when generating the toy models
steps=2500				#num of steps for integrating the pdf's
accuracy=0.005				#amount by how much both Q_pdf integrals are required to be similar
N_tt=10					#num of lookup tables for DM, i.e. time bins
N_min_nu=75000				#num of created models to create 2d pdf
N_min_DM=50000				#num of created models to create 2d pdf
source_length=10000			#maximal number of toy models in pool to draw from per factor
					#total maximal pool size=factor*source_length
mode_max_Mdet=False
if mode_max_Mdet==True: gain_direction=True

###choose detector set-up
###	(choose Z in constants.py)
M_det=1.e6#g
t0=eph.date('2014/01/01')
t1=eph.date('2015/01/01')
T_det=(float(t1)-float(t0))
t1_f=float(t1)-float(t0)

###specify mass and cross section range
###
#m_DM_array=np.array([1000.,500.,250.,100.,50.,40.,30.,20.,15.,12.,11.,10.,9.,8.,7.,6.])
#m_DM_array=np.array([6.,7.,8.,9.,10.,11.,12.,15.,20.,30.,40.,50.,100.,250.,500.,1000.])
m_DM_array=np.array([250.])
sigma=np.logspace(-40,-52,120)#one sigma takes approx 1min with N_sim==25.000
filename_dm='sensitivity_250GeV_CF4_5keV_part2.txt'
f=open(filename_dm,'w')
f.close()

np.random.seed()
###############################################################################################################

if channel_Erec==True and channel_time==False:
	print ''
	print 'including recoil energy without considering time information is not possible'
	print '(because that would be stupid)'
	print ''
	sys.exit()
if test==0:
	print ''
	print '#########################################################################'
	print 'START'
	print '#########################################################################'
	print ''
	print '####################################'
	print '#detector specifications:'
	print '#mass target material: A=', A,'(N',N,' Z',Z,')'
	print '#detector mass = ',M_det/1000.,'kg'
	print '#exposure time = ', T_det,' live days'
	print '#lower threshold: ',E_thr
	print '#upper threshold: ',upper_threshold
	print '####################################'
	print ''
	print 'NEUTRINOS'
	print ''
	print 'creating lookup tables...'
	#create_all_neutrino_lookuptables(Npoints=400,Nsteps=800)
	print '			DONE!'
	print ''
	print 'calculating expected events...'

#mass=np.array([10.,25.,50.,100.,500.,1000.,2500.,5000.,10000.,25000.,50000.])
mass=np.array([100000.,500000])

mm=0
###calculate the general annual modulation for that DM mass
###
print 'calculating expected events...'
time_array_DM,rate_array_DM,N_DM0,dt_DM=tot_rate_dm(M_det,E_thr,m_DM_array[mm],t0,t1,steps=100)
print '			DONE!'

bin_prob_DM_t=rate_array_DM/N_DM0
acc_prob_DM_t=np.cumsum(bin_prob_DM_t)
rnd_DM_t=interp1d(acc_prob_DM_t,time_array_DM,kind='linear')
rate_array_DM[0]=rate_array_DM[1]#because the differential rate is non zero also in the first bin	
pdf_DM_t=interp1d(time_array_DM,rate_array_DM/N_DM0/dt_DM,kind='linear')

if test==1:
	N_DM_exp=10
s_int=0
###get 2d pdf for recoil energy and cos_theta_sun for DM
###
H_out=np.zeros((N_erec-1,N_theta-1))
t0_f=float(t0)
t1_f=float(t1)-t0_f
t0_f=0.
t_edges=np.linspace(t0_f,t1_f,N_tt)

###define the grid
###
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


if channel_Erec==True:
	if test==0:
		print ''
		print '2 dimensional E_rec-cos_theta pdf for different times of the year'
	        try:
			name=basenamepdf+'_DM_'+str(m_DM_array[mm])+'_'+str(0)+'.txt'
	                with open(name):
				print '			Files for 2d pdf exist!'
				print '			All files read!'
		except:
			print '			Creating the pdf...'
			for ff in range (factor):
				H_tmp=recoil_dm(E_thr,m_DM_array[mm],t0,t1,N_t=N_tt,N_min=N_min_DM,mode=0)
				H_out=H_out+H_tmp
			print '			File created!'
			H_out=1.*H_out/factor		
			for hh in range (len(H_out)):
				f=open(basenamepdf+'_DM_'+str(m_DM_array[mm])+'_'+str(hh)+'.txt','w')
				f.close()
				f=open(basenamepdf+'_DM_'+str(m_DM_array[mm])+'_'+str(hh)+'.txt','a')

				HHnz=np.where(H_out[hh]>0,H_out[hh],10.)
				HHmin=np.min(HHnz)
				for ii in range (len(x_edge)):
					for jj in range (len(y_edge)):
				        	if H_out[hh][ii][jj]<HHmin:
	 	                			H_out[hh][ii][jj]=1./(factor*N_min_DM)
						f.write(str(x_edge[ii])+' '+str(y_edge[jj])\
								+' '+str(H_out[hh][ii][jj])+'\n')	
				f.close()	
		
		###Normalise properly
		###
		f_array=[]
		for tt in range (N_tt):
			name=basenamepdf+'_DM_'+str(m_DM_array[mm])+'_'+str(tt)+'.txt'
		        data=np.loadtxt(name)
		        pdf_val=data[:,2]
			pdf_val=np.reshape(pdf_val,(N_erec-1,N_theta-1))
			f_ipl=RectBivariateSpline(x_edge,y_edge,pdf_val,kx=1,ky=1)
			norm_Psb=f_ipl.integral(min(x_edge),max(x_edge),min(y_edge),max(y_edge))
			f_ipl=RectBivariateSpline(x_edge,y_edge,pdf_val/norm_Psb,kx=1,ky=1)
                        f_array.append(f_ipl)

		###marginalise over the angle
		###	(if only recoil information should be used)
		if channel_angle==False or gain_direction==True:
			f_array_noangle=[]
			for tt in range (N_tt):
                	        name=basenamepdf+'_DM_'+str(m_DM_array[mm])+'_'+str(tt)+'.txt'
                                data=np.loadtxt(name)
	                        pdf_val=data[:,2]
        	                pdf_val=np.reshape(pdf_val,(N_erec-1,N_theta-1))
				p_margin=[]
				for ii in range(len(x_edge)):
					p_margin_tmp=0.
					for jj in range (len(y_edge)-1):
						p_margin_tmp+=0.5*(pdf_val[ii][jj]+pdf_val[ii][jj+1])*dtheta
					p_margin.append(p_margin_tmp)
				norm=0.
				for ii in range (len(x_edge)-1):
					derec=x_edge[ii+1]-x_edge[ii]
			                norm+=0.5*(p_margin[ii]+p_margin[ii+1])*derec
				f_ipl=interp1d(x_edge,p_margin/norm,kind='linear')
				f_array_noangle.append(f_ipl)
				
		###Test interpolation
		###
		#ee=np.linspace(E_thr,upper_threshold,100)
		#plt.plot(ee,f_array_noangle[6](ee),'ro')
		#plt.plot(ee,Pb_noangle_ipl(ee),'bo')
		#plt.show()

		#name=basenamepdf+'_DM_'+str(m_DM_array[mm])+'_'+str(8)+'.txt'
	        #data=np.loadtxt(name)
		#x_val=data[:,0]
	        #pdf_val=data[:,2]		
		#ee=np.linspace(E_thr,upper_threshold,100)
		#thth=-1.
		#plt.plot(x_val,pdf_val,'bo')
		#for ii in range (len(ee)):
		#	plt.plot(ee[ii],f_array[8](ee[ii],thth),'ro')
		#plt.yscale('log')
		#plt.show()

if test==1:
	t0_f=float(t0)
	t1_f=float(t1)-t0_f
	t0_f=0.
	t_edges=np.linspace(t0_f,t1_f,N_tt)
	f_array,f_array_noangle=[],[]
	for ii in range (N_tt):
		pdf_dm=5.5/(D_Erec*D_cos)*np.ones(4)
		x_test=np.array([E_thr,E_thr,upper_threshold,upper_threshold])
		y_test=np.array([-1.,1.,-1.,1.])
		f_ipl=SmoothBivariateSpline(x_test,y_test,pdf_dm,kx=1,ky=1)
                f_array.append(f_ipl)
		
		pdf_dm=np.array([pdf_dm[0],pdf_dm[2]])
		f_ipl=interp1d(np.array([E_thr,upper_threshold]),pdf_dm)
		f_array_noangle.append(f_ipl)

	kk=0
	pdf_array=5./D_time*np.ones(len(time_array_nu))
	pdf_DM_t=interp1d(time_array_DM,pdf_array,kind='linear')

###simulate DM events
###
print 'simulate DM events...'
source_length=5000
t_src_DM,E_rec_src_DM,cos_src_DM=[],[],[]
void_source=np.zeros(source_length)
for ff in range (factor):
	t_src_DM.append(void_source)
	E_rec_src_DM.append(void_source)
        cos_src_DM.append(void_source)

if test==0:
        if channel_time==True:
		t_src_DM=[]
		for ff in range (factor):
			r_array=np.random.uniform(0.,1.,source_length)
	                t_tmp=rnd_DM_t(r_array)
			t_src_DM.append(t_tmp)

	if channel_Erec==True:
	        t_src_DM,E_rec_src_DM,cos_src_DM=[],[],[]
	        for ff in range (factor):
	                t_tmp,E_rec_tmp,cos_tmp=recoil_dm(E_thr,m_DM_array[mm],t0,t1,N_tt,\
								N_evt=source_length,mode=1)
	                t_src_DM.append(t_tmp)
	                E_rec_src_DM.append(E_rec_tmp)
	                cos_src_DM.append(cos_tmp)
if test==1:
	N_DM=10
	n_source_DM=N_DM*np.ones(N_sim,dtype=int)
	E_rec_src_DM=np.ones((factor,N_sim*N_DM))*25.
	t_src_DM=np.ones((factor,N_sim*N_DM))*100.
	cos_src_DM=np.ones((factor,N_sim*N_DM))*0.1
	src_DM=np.array(zip(t_src_DM,E_rec_src_DM,cos_src_DM))
print '			DONE!'

for ard in range (len(mass)):

	###calculate number of expected events
	###	(assuming flux at central value)
	time_array_nu,rate_array_nu,dt_nu,N_sun_arr,N_atmo_arr,N_dsnb_arr=tot_rate_nu(M_det,E_thr,t0,t1,steps=100)
	
	N_sun0=np.sum(N_sun_arr)
	N_atmo0=np.sum(N_atmo_arr)
	N_dsnb0=np.sum(N_dsnb_arr)
	mu_nu_0=N_sun0+N_atmo0+N_dsnb0
	N_nu0=mu_nu_0
	N_tot=mu_nu_0
	print mu_nu_0
	M_det=mass[ard]/mu_nu_0*M_det
	print 'again'
	time_array_nu,rate_array_nu,dt_nu,N_sun_arr,N_atmo_arr,N_dsnb_arr=tot_rate_nu(M_det,E_thr,t0,t1,steps=100)
	
	print N_sun_arr
	print N_atmo_arr
	print N_dsnb_arr
	N_sun0=np.sum(N_sun_arr)
	N_atmo0=np.sum(N_atmo_arr)
	N_dsnb0=np.sum(N_dsnb_arr)
	mu_nu_0=N_sun0+N_atmo0+N_dsnb0
	N_nu0=mu_nu_0
	N_tot=mu_nu_0
	
	N_nu_avg=int(mu_nu_0)
	rest=mu_nu_0-N_nu_avg
	if rest>0.5:
		N_nu_avg+=1
	if test==1:
		N_nu_avg=39

	print 'calculating expected events...'
	time_array_DM,rate_array_DM,N_DM0,dt_DM=tot_rate_dm(M_det,E_thr,m_DM_array[mm],t0,t1,steps=100)
	print '			DONE!'

	bin_prob_DM_t=rate_array_DM/N_DM0
	acc_prob_DM_t=np.cumsum(bin_prob_DM_t)
	rnd_DM_t=interp1d(acc_prob_DM_t,time_array_DM,kind='linear')
	rate_array_DM[0]=rate_array_DM[1]#because the differential rate is non zero also in the first bin	
	pdf_DM_t=interp1d(time_array_DM,rate_array_DM/N_DM0/dt_DM,kind='linear')

	###create the pdf's for the neutrino signals
	###
	bin_prob_B_t=rate_array_nu/N_nu0
	acc_prob_B_t=np.cumsum(bin_prob_B_t)
	rnd_B_t=interp1d(acc_prob_B_t,time_array_nu,kind='linear')
	rate_array_nu[0]=rate_array_nu[1]
	pdf_nu_t=interp1d(time_array_nu,rate_array_nu/N_nu0/dt_nu,kind='linear')
	
	###get the relative parts of neutrino sources
	###	(assuming central flux values)
	
	sun_ratio=N_sun0/N_tot
	atmo_ratio=N_atmo0/N_tot
	dsnb_ratio=N_dsnb0/N_tot
	ratio_array=np.array([sun_ratio,atmo_ratio,dsnb_ratio])
	N_nu_arr=np.array([N_sun0,N_atmo0,N_dsnb0])
	mu_nu=N_tot
	
	if channel_Erec==True:
		###define the grid
		###
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
	
		###create 2d pdf
		###
		if test==0:
			print ''
			print '2 dimensional E_rec-cos_theta pdf'
	                try:
	                        name=basenamepdf+'_nu.txt'
	                        with open(name):
	                                print '			File for 2d neutrino pdf exist!'
					print '			File read!'
	                except:
				print '			Creating the pdf...'
				pdf_nu=np.zeros((N_erec-1,N_theta-1))
				for ff in range (factor):
					N_arr=N_nu_arr
					ratio_arr=N_arr/(np.sum(N_arr))
					for ii in range (3):
						pdf_nu_tmp=recoil_nu(E_thr,ii,N_min=int(N_min_nu*ratio_arr[ii]),mode=0)
						pdf_nu=pdf_nu+pdf_nu_tmp*ratio_arr[ii]
				                pdf_nu_tmp=[]
	
				pdf_nu=1.*pdf_nu/factor
				f=open(basenamepdf+'_nu.txt','w')
				f.close()
				tmp=np.where(pdf_nu>0,pdf_nu,10.)
				tmp_min=np.min(tmp)
				f=open(basenamepdf+'_nu.txt','a')
				for ii in range (len(x_edge)):
					for jj in range (len(y_edge)):
						if pdf_nu[ii][jj]<tmp_min:
							pdf_nu[ii][jj]=1./(factor*N_min_nu)
						f.write(str(x_edge[ii])+' '+str(y_edge[jj])+' '+str(pdf_nu[ii][jj])+'\n')
				f.close()
				print '			File created!'
	
			###Normalise properly
		        ###	(because of shifting edges and assigning non-zeros values to bins with no events)
		        name=basenamepdf+'_nu.txt'
		        data=np.loadtxt(name)
		        pdf_val=data[:,2]
			pdf_val=np.reshape(pdf_val,(N_erec-1,N_theta-1))
			Pb_ipl=RectBivariateSpline(x_edge,y_edge,pdf_val,kx=1,ky=1)
			norm_Pb=Pb_ipl.integral(min(x_edge),max(x_edge),min(y_edge),max(y_edge))
			Pb_ipl=RectBivariateSpline(x_edge,y_edge,pdf_val/norm_Pb,kx=1,ky=1)
	
			###marginalise over angle if only recoil information should be used
			###
			p_margin=[]
			for ii in range (len(x_edge)):
				p_margin_tmp=0.
				for jj in range (len(y_edge)-1):
					p_margin_tmp+=0.5*(pdf_val[ii][jj]+pdf_val[ii][jj+1])*dtheta
				p_margin.append(p_margin_tmp)
			norm=0.
			for ii in range (len(x_edge)-1):
				derec=x_edge[ii+1]-x_edge[ii]
				norm+=0.5*(p_margin[ii]+p_margin[ii+1])*derec
			Pb_noangle_ipl=interp1d(x_edge,p_margin/norm,kind='linear')
	
		if test==1:
			D_Erec=upper_threshold-E_thr
			D_cos=2.
			D_time=365.25
	
			x_test=np.array([E_thr,E_thr,upper_threshold,upper_threshold])
			y_test=np.array([-1.,1.,-1.,1.])
	
			pdf_nu=9.54/(D_Erec*D_cos)*np.ones(4)
		        Pb_ipl=SmoothBivariateSpline(x_test,y_test,pdf_nu,kx=1,ky=1)
		
			pdf_margin=np.array([pdf_nu[0],pdf_nu[2]])
			Pb_noangle_ipl=interp1d(np.array([E_thr,upper_threshold]),pdf_margin)
	
			pdf_nu_t=interp1d(time_array_nu,1./D_time*np.ones(len(time_array_nu)),kind='linear')
	
			#E_test=np.arange(E_thr,upper_threshold,1.)
			#theta_test=np.arange(-1.,1.,0.2)
			#f=open('pdfs/test_nu.txt','w')
			#f.close()
			#for ii in range (len(E_test)):
			#	for jj in range (len(theta_test)):
			#		f=open('pdfs/test_nu.txt','a')
			#		f.write(str(E_test[ii])+' '+str(theta_test[jj])\
			#			+' '+str(float(Pb_ipl(E_test[ii],theta_test[jj])))+'\n')
	
	###simulate neutrino events
	###
	if test==0:
		print ''
		print 'simulating neutrino events...'
	
	###solar neutrinos
	###
	print '			solar...'
	t_src_solar_nu=[]
	E_rec_src_solar_nu=[]
	cos_src_solar_nu=[]
	
	solar_length=max(10,min(int(1.*N_sun0*N_sim),source_length))
	void_source=np.zeros(solar_length)
	for ff in range (factor):
		t_src_solar_nu.append(void_source)
		E_rec_src_solar_nu.append(void_source)
		cos_src_solar_nu.append(void_source)
	if channel_time==True:
		t_src_solar_nu=[]
		for ff in range (factor):
			r_array=np.random.uniform(0.,1.,solar_length)
			t_src_solar_nu.append(rnd_B_t(r_array))
	if channel_Erec==True:
		E_rec_src_solar_nu,cos_src_solar_nu=[],[]
		for ff in range (factor):
			E_rec_tmp,cos_tmp=recoil_nu(E_thr,0,N_evt=solar_length,mode=1)
			E_rec_src_solar_nu.append(E_rec_tmp)
			cos_src_solar_nu.append(cos_tmp)
	
	###atmospheric neutrinos
	### (minimum number of events to be gerated for the pool of neutrinois is 10 for each species)
	print '			atmospheric...'
	t_src_atmo_nu=[]
	E_rec_src_atmo_nu=[]
	cos_src_atmo_nu=[]
	
	atmo_length=max(10,min(int(1.*N_atmo0*N_sim),source_length))
	void_source=np.zeros(atmo_length)
	for ff in range (factor):
		t_src_atmo_nu.append(void_source)
		E_rec_src_atmo_nu.append(void_source)
		cos_src_atmo_nu.append(void_source)
	if channel_time==True:
		t_src_atmo_nu=[]
		for ff in range (factor):
			r_array=np.random.uniform(0.,1.,atmo_length)
			t_src_atmo_nu.append(r_array*t1_f)
	if channel_Erec==True:
		E_rec_src_atmo_nu,cos_src_atmo_nu=[],[]
		for ff in range (factor):
			E_rec_tmp,cos_tmp=recoil_nu(E_thr,1,N_evt=atmo_length,mode=1)
			E_rec_src_atmo_nu.append(E_rec_tmp)
			cos_src_atmo_nu.append(cos_tmp)
	
	###DSNB neutrinos
	###
	print '			DSNB...'
	t_src_dsnb_nu=[]
	E_rec_src_dsnb_nu=[]
	cos_src_dsnb_nu=[]
	
	dsnb_length=max(10,min(int(N_dsnb0*N_sim),source_length))
	void_source=np.zeros(dsnb_length)
	
	for ff in range (factor):
		t_src_dsnb_nu.append(void_source)
		E_rec_src_dsnb_nu.append(void_source)
		cos_src_dsnb_nu.append(void_source)
	if channel_time==True:
		t_src_dsnb_nu=[]
		for ff in range (factor):
			r_array=np.random.uniform(0.,1.,dsnb_length)
			t_src_dsnb_nu.append(t1_f*r_array)
	
	if channel_Erec==True:
		E_rec_src_dsnb_nu,cos_src_dsnb_nu=[],[]
		for ff in range (factor):
			E_rec_tmp,cos_tmp=recoil_nu(E_thr,2,N_evt=dsnb_length,mode=1)
			E_rec_src_dsnb_nu.append(E_rec_tmp)
			cos_src_dsnb_nu.append(cos_tmp)
	
	del E_rec_tmp
	del cos_tmp
	
	if test==0:
		print '			DONE!'
		print ''
		print 'DARK MATTER'
	
	count=0
	
	###loop over DM mass
	###
	#s_int=0
	for mm in range (len(m_DM_array)):
	
		if test==0:
			print ''
			print 'loop over cross section...'
			print ''
			print "DM_mass        sigma_SI       N_dm         N_nu"
	
		###loop over cross section
		###
	
		#for ss in range (len(sigma)):
		scan=True
		waszero=False
		wasbig=False
		mem1=0
		mem2=0
		scanned=[]
		jump=False
	
		while scan==True:
			jump=False
			if s_int in scanned:
				jump=True
	
			if jump==False:
				if s_int<0:
					print 'break'
					break
				ss=s_int
				
				###initialise arrays
				###	
				Q_B=[]
				Q_SB=[]
				t_array_SB=[]
		
				###calculate number of expected dark matter events by multiplying with sigma
				###
				if test==0:
					N_DM1=N_DM0*sigma[ss]
					N_DM=int(N_DM1)
					rest=N_DM1-N_DM
					if rest>0.5:
						N_DM+=+1
		
				if N_DM<1. and waszero==False and wasbig==False:
					s_int-=1 
					continue
		
				if N_DM<1:
					break
		
				if N_DM>isoevents*N_nu_avg:
					s_int+=1
					continue
		
				N_DM_exp=N_DM
				mu_DM=N_DM_exp
		
				print ("%2E   %2E   %i        %i " % (m_DM_array[mm],sigma[ss],N_DM_exp,N_nu_avg))
		
				Q_B_angle,Q_B_erec=np.array([]),np.array([])
				Q_SB_angle,Q_SB_erec=np.array([]),np.array([])
		
				angle_info_DM,erec_info_DM=[],[]
		
				for ff in range (factor):
					print ff	
					#######################
					###B only hypothesis###
					#######################
					N_nu_exp=N_sun0+N_atmo0+N_dsnb0
		
					ratio_solar=1.*N_sun0/N_nu_exp
					ratio_atmo=1.*N_atmo0/N_nu_exp
					ratio_dsnb=1.*N_dsnb0/N_nu_exp
		
					N_nu=N_nu_exp
					n_nu_arr=np.random.poisson(N_nu,N_sim)
					NN_nu=np.sum(n_nu_arr)
		
					if print_info==1:
						print 'B only hypothesis'
						print ''
						print '(mu_DM_0, mu_nu_0) ~ (center of Poisson) '
						print N_DM_exp,N_nu_exp
						print ''
						print 'n_nu_arr (drawn from the Poisson)'
						print n_nu_arr[:12]
						print ''				
					del N_nu
		
					r_solar=ratio_solar*np.ones(NN_nu)
					r_atmo=ratio_atmo*np.ones(NN_nu)
					r_dsnb=ratio_dsnb*np.ones(NN_nu)
		
					###split up in solar, atmo and dsnb neutrinos
					###
					r_array=np.random.uniform(0.,1.,NN_nu)
					N_solar=np.where(r_array<r_solar,1,0)
					N_dsnb=np.where(r_array>(r_solar+r_atmo),1,0)
					N_atmo=1-N_solar-N_dsnb	
					NN_solar=np.sum(N_solar)
					NN_atmo=np.sum(N_atmo)
					NN_dsnb=np.sum(N_dsnb)
					NN_tot=NN_solar+NN_atmo+NN_dsnb
		
					#print 1.*NN_solar/NN_tot,1.*NN_atmo/NN_tot,1.*NN_dsnb/NN_tot
		
					sample_prob_nu_B=np.zeros(N_sim)
					t_array_solar,t_array_atmo,t_array_dsnb=[],[],[]
					E_rec_array_solar,E_rec_array_atmo,E_rec_array_dsnb=[],[],[]
					cos_array_solar,cos_array_atmo,cos_array_dsnb=[],[],[]
		
					###simulate neutrino events
					###
					if channel_time==True:
		
						prob_nu_B=np.zeros(NN_nu)
		
						###if recoil energy information is used, need to do proper simulation
						###
						if channel_Erec==True:
							###solar
							###
							if solar_length>1:
								while ((len(E_rec_array_solar))<NN_solar):
									jj=np.random.randint(0,solar_length-1,solar_length)
						        	        t_array_tmp=t_src_solar_nu[ff][jj]
									E_rec_array_tmp=E_rec_src_solar_nu[ff][jj]
									cos_array_tmp=cos_src_solar_nu[ff][jj]
									t_array_solar=np.concatenate((t_array_solar,t_array_tmp))
									E_rec_array_solar=np.concatenate((E_rec_array_solar,E_rec_array_tmp))
									cos_array_solar=np.concatenate((cos_array_solar,cos_array_tmp))
								t_array_solar=t_array_solar[:NN_solar]
								E_rec_array_solar=E_rec_array_solar[:NN_solar]
								cos_array_solar=cos_array_solar[:NN_solar]
		
							###atmo
							###
							if atmo_length>1:
								while ((len(E_rec_array_atmo))<NN_atmo):
									jj=np.random.randint(0,atmo_length-1,atmo_length)
						        	        t_array_tmp=t_src_atmo_nu[ff][jj]
									E_rec_array_tmp=E_rec_src_atmo_nu[ff][jj]
									cos_array_tmp=cos_src_atmo_nu[ff][jj]
									t_array_atmo=np.concatenate((t_array_atmo,t_array_tmp))
									E_rec_array_atmo=np.concatenate((E_rec_array_atmo,E_rec_array_tmp))
									cos_array_atmo=np.concatenate((cos_array_atmo,cos_array_tmp))
								t_array_atmo=t_array_atmo[:NN_atmo]
								E_rec_array_atmo=E_rec_array_atmo[:NN_atmo]
								cos_array_atmo=cos_array_atmo[:NN_atmo]
		
							###dsnb
							###
							if dsnb_length>1:
								while ((len(E_rec_array_dsnb))<NN_dsnb):
									jj=np.random.randint(0,dsnb_length-1,dsnb_length)
						        	        t_array_tmp=t_src_dsnb_nu[ff][jj]
									E_rec_array_tmp=E_rec_src_dsnb_nu[ff][jj]
									cos_array_tmp=cos_src_dsnb_nu[ff][jj]
									t_array_dsnb=np.concatenate((t_array_dsnb,t_array_tmp))
									E_rec_array_dsnb=np.concatenate((E_rec_array_dsnb,E_rec_array_tmp))
									cos_array_dsnb=np.concatenate((cos_array_dsnb,cos_array_tmp))
								t_array_dsnb=t_array_dsnb[:NN_dsnb]
								E_rec_array_dsnb=E_rec_array_dsnb[:NN_dsnb]
								cos_array_dsnb=cos_array_dsnb[:NN_dsnb]
		
						###stick together in correct order
						###
						E_rec_nu=np.zeros(NN_nu)
						cos_nu=np.zeros(NN_nu)
						t_nu=np.zeros(NN_nu)
		
						if solar_length>1:
							ij=np.nonzero(N_solar>0.5)
							E_rec_nu[ij]=E_rec_array_solar
							cos_nu[ij]=cos_array_solar
							t_nu[ij]=t_array_solar
						if atmo_length>1:
							ij=np.nonzero(N_atmo>0.5)
		                                        E_rec_nu[ij]=E_rec_array_atmo
							cos_nu[ij]=cos_array_atmo
							t_nu[ij]=t_array_atmo
						if dsnb_length>1:
							ij=np.nonzero(N_dsnb>0.5)
		                                        E_rec_nu[ij]=E_rec_array_dsnb
							cos_nu[ij]=cos_array_dsnb
		                                	t_nu[ij]=t_array_dsnb
		
		                                del t_array_solar
						del t_array_atmo
						del t_array_dsnb
		                                del E_rec_array_solar
						del E_rec_array_atmo
						del E_rec_array_dsnb
			                        del cos_array_solar
						del cos_array_atmo
						del cos_array_dsnb
		
						for ii in range (len(t_nu)):
							if E_rec_nu[ii]<E_thr:
								print E_rec_nu[ii],ii
			
						###calculate Pb_B
						###
						mu_nu=np.zeros((N_sim,N_Q))
						for ii in range (9):
							if N_sun_arr[ii]>0.:
								N_nu_solar=np.random.normal(N_sun_arr[ii],nu_sigma[0]*N_sun_arr[ii],(N_sim,N_Q))
								N_nu_solar=np.where(N_nu_solar<0.,0.,N_nu_solar)
								mu_nu+=N_nu_solar
						for ii in range (4):
							if N_atmo_arr[ii]>0.:
								N_nu_atmo=np.random.normal(N_atmo_arr[ii],nu_sigma[1]*N_atmo_arr[ii],(N_sim,N_Q))
								N_nu_atmo=np.where(N_nu_atmo<0.,0.,N_nu_atmo)
								mu_nu+=N_nu_atmo
						for ii in range (3):
							if N_dsnb_arr[ii]>0.:
								N_nu_dsnb=np.random.normal(N_dsnb_arr[ii],nu_sigma[2]*N_dsnb_arr[ii],(N_sim,N_Q))
								N_nu_dsnb=np.where(N_nu_dsnb<0.,0.,N_nu_dsnb)
								mu_nu+=N_nu_dsnb
						
						if test==1:
							n_nu_arr=N_nu_avg*np.ones(N_sim,dtype=int)
		
							E_rec_nu=2.*E_thr*np.ones(N_sim*N_nu_avg)
							cos_nu=0.5*np.ones(N_sim*N_nu_avg)
							t_nu=100.*np.ones(N_sim*N_nu_avg)
							mu_nu=N_nu_avg*np.ones((N_sim,N_Q))
							NN_nu=N_nu_avg*N_sim
		
						mu_nu=np.hstack(mu_nu)
						n_nu_arr=np.tile(n_nu_arr,N_Q)
		
					if channel_time==True:
						if channel_Erec==True:
							dif=np.zeros((len(t_edges),len(t_nu)))
							for ii in range(len(t_edges)):
								dif[ii]=abs(t_edges[ii]-t_nu)
							dif=np.reshape(dif,(len(t_nu),len(t_edges)))
							id1_nu=np.argmin(dif,axis=1)
							t0_nu=t_edges[id1_nu]
							id2_nu=np.where(id1_nu==N_tt-1,id1_nu-1,0)
							id2_nu=np.where(id1_nu==0,1,id2_nu)
							
							id2_nu_tmp1=np.where(id2_nu==0,id1_nu+1,id2_nu)
							t1_nu=t_edges[id2_nu_tmp1]
							
							id2_nu_tmp2=np.where(id2_nu==0,id1_nu-1,id2_nu)
							t2_nu=t_edges[id2_nu_tmp2]
							
							id2_nu=np.where(abs(t1_nu-t_nu)>abs(t2_nu-t_nu),id2_nu_tmp2,id2_nu)
							id2_nu=np.where(abs(t1_nu-t_nu)<abs(t2_nu-t_nu),id2_nu_tmp1,id2_nu)
					                d1=abs(t_nu-t_edges[id1_nu])
					                d2=abs(t_nu-t_edges[id2_nu])
							pdf1,pdf2=np.zeros(NN_nu),np.zeros(NN_nu)
							for ii in range (N_tt):
								pdf1=np.where(id1_nu==ii,f_array[ii].ev(E_rec_nu,cos_nu),pdf1)
								pdf2=np.where(id2_nu==ii,f_array[ii].ev(E_rec_nu,cos_nu),pdf2)
							prob_nu_B_angle=Pb_ipl.ev(E_rec_nu,cos_nu)
							prob_nu_S_angle=pdf1+d1/(d1+d2)*(pdf2-pdf1)
							for ii in range (N_tt):
		                                                pdf1=np.where(id1_nu==ii,f_array_noangle[ii](E_rec_nu),pdf1)
		                                                pdf2=np.where(id2_nu==ii,f_array_noangle[ii](E_rec_nu),pdf2)
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
						prob_nu_B_time=np.split(prob_nu_B_time,n_split)
						prob_nu_S_time=np.split(prob_nu_S_time,n_split)
						prob_nu_B_angle=np.split(prob_nu_B_angle,n_split)
						prob_nu_S_angle=np.split(prob_nu_S_angle,n_split)
						prob_nu_B_erec=np.split(prob_nu_B_erec,n_split)
						prob_nu_S_erec=np.split(prob_nu_S_erec,n_split)
					
						for ii in range (N_sim*N_Q):
							time_info=np.log(1.+1.*mu_DM/mu_nu[ii]*prob_nu_S_time[ii]/prob_nu_B_time[ii])
							angle_info=np.log(1.+1.*mu_DM/mu_nu[ii]*prob_nu_S_angle[ii]/prob_nu_B_angle[ii])
							erec_info=np.log(1.+1.*mu_DM/mu_nu[ii]*prob_nu_S_erec[ii]/prob_nu_B_erec[ii])
		
							pre=-mu_DM+n_nu_arr[ii]*(np.log(mu_nu[ii])-np.log(mu_DM+mu_nu[ii]))
		
							angle_info=pre+np.sum(time_info)+np.sum(angle_info)
							erec_info=pre+np.sum(time_info)+np.sum(erec_info)
		
							Q_B_angle=np.concatenate((Q_B_angle,np.array([angle_info])))
							Q_B_erec=np.concatenate((Q_B_erec,np.array([erec_info])))
		
					#########################
					###Now, S+B hypothesis###
					#########################
					
		                        N_SB=N_DM_exp+N_nu_exp
		                        if print_info==1:
						print 'S+B hypothesis'
						print ''
		                                print 'N_DM_exp, N_nu_exp, N_SB'
		                                print N_DM_exp,N_nu_exp
		                                print ''
		
					NN_SB=np.random.poisson(N_SB,N_sim)
					ratio_solar=1.*N_sun0/N_SB
					ratio_atmo=1.*N_atmo0/N_SB
					ratio_dsnb=1.*N_dsnb0/N_SB
					ratio_dm=1.*N_DM/N_SB
					
					###get number of DM and neutrino events
					###
					n_arr=np.sum(NN_SB)
					r_dm=ratio_dm*np.ones(n_arr)
		                        r_solar=ratio_solar*np.ones(n_arr)
		                        r_atmo=ratio_atmo*np.ones(n_arr)
		                        r_dsnb=ratio_dsnb*np.ones(n_arr)
		
					###split up in solar, atmo and dsnb neutrinos
		                        ###
		                        r_array=np.random.uniform(0.,1.,n_arr)
		                        N_solar=np.where(r_array<r_solar,1,0)
		                        N_dm=np.where(r_array>(r_solar+r_atmo+r_dsnb),1,0)
		                        N_atmo=1-N_solar-N_dm
					N_atmo=np.random.uniform(0.,r_atmo+r_dsnb)*N_atmo
					N_atmo=np.where(N_atmo>r_dsnb,1,0)
					N_dsnb=1-N_dm-N_solar-N_atmo
		                        NN_solar=np.sum(N_solar)
		                        NN_atmo=np.sum(N_atmo)
		                        NN_dsnb=np.sum(N_dsnb)
					NN_DM=np.sum(N_dm)
		                        NN_tot=NN_solar+NN_atmo+NN_dsnb+NN_DM
					NN_nu=NN_tot-NN_DM
		
					###simulate DM events
					###
					tmp=np.cumsum(NN_SB)
					i_max=len(tmp)
					tmp=np.split(N_dm,tmp)
					n_array_DM=np.zeros(i_max,dtype=int)
					for ii in range (i_max):
						n_array_DM[ii]=int(sum(tmp[ii]))
					n_nu_arr=NN_SB-n_array_DM
					
					if test==1:
						n_array_DM=N_DM_exp*np.ones(N_sim,dtype=int)
		
					if channel_time==True:
						t_array_DM,E_rec_array_DM,cos_array_DM=[],[],[]
						while ((len(t_array_DM))<NN_DM):
							jj=np.random.randint(0,source_length-1,source_length)
							t_array_tmp=t_src_DM[ff][jj]
							E_rec_array_tmp=E_rec_src_DM[ff][jj]
							cos_array_tmp=cos_src_DM[ff][jj]
							t_array_DM=np.concatenate((t_array_DM,t_array_tmp))
							E_rec_array_DM=np.concatenate((E_rec_array_DM,E_rec_array_tmp))
							cos_array_DM=np.concatenate((cos_array_DM,cos_array_tmp))
						t_array_DM=t_array_DM[:NN_DM]
						E_rec_array_DM=E_rec_array_DM[:NN_DM]
						cos_array_DM=cos_array_DM[:NN_DM]
					
					###simulate neutrino events
					###
					t_array_solar,t_array_atmo,t_array_dsnb=[],[],[]
					E_rec_array_solar,E_rec_array_atmo,E_rec_array_dsnb=[],[],[]
					cos_array_solar,cos_array_atmo,cos_array_dsnb=[],[],[]
		
					if channel_time==True:
						###if recoil energy information is used, need to do proper simulation
						###
						if channel_Erec==True:
							###solar
							###
							if solar_length>1:
								while ((len(E_rec_array_solar))<NN_solar):
									jj=np.random.randint(0,solar_length-1,solar_length)
						        	        t_array_tmp=t_src_solar_nu[ff][jj]
									E_rec_array_tmp=E_rec_src_solar_nu[ff][jj]
									cos_array_tmp=cos_src_solar_nu[ff][jj]
									t_array_solar=np.concatenate((t_array_solar,t_array_tmp))
									E_rec_array_solar=np.concatenate((E_rec_array_solar,E_rec_array_tmp))
									cos_array_solar=np.concatenate((cos_array_solar,cos_array_tmp))
								t_array_solar=t_array_solar[:NN_solar]
								E_rec_array_solar=E_rec_array_solar[:NN_solar]
								cos_array_solar=cos_array_solar[:NN_solar]
		
							###atmo
							###
							if atmo_length>1:
								while ((len(E_rec_array_atmo))<NN_atmo):
									jj=np.random.randint(0,atmo_length-1,atmo_length)
						        	        t_array_tmp=t_src_atmo_nu[ff][jj]
									E_rec_array_tmp=E_rec_src_atmo_nu[ff][jj]
									cos_array_tmp=cos_src_atmo_nu[ff][jj]
									t_array_atmo=np.concatenate((t_array_atmo,t_array_tmp))
									E_rec_array_atmo=np.concatenate((E_rec_array_atmo,E_rec_array_tmp))
									cos_array_atmo=np.concatenate((cos_array_atmo,cos_array_tmp))
								t_array_atmo=t_array_atmo[:NN_atmo]
								E_rec_array_atmo=E_rec_array_atmo[:NN_atmo]
								cos_array_atmo=cos_array_atmo[:NN_atmo]
		
							###dsnb
							###
							if dsnb_length>1:
								while ((len(E_rec_array_dsnb))<NN_dsnb):
									jj=np.random.randint(0,dsnb_length-1,dsnb_length)
						        	        t_array_tmp=t_src_dsnb_nu[ff][jj]
									E_rec_array_tmp=E_rec_src_dsnb_nu[ff][jj]
									cos_array_tmp=cos_src_dsnb_nu[ff][jj]
									t_array_dsnb=np.concatenate((t_array_dsnb,t_array_tmp))
									E_rec_array_dsnb=np.concatenate((E_rec_array_dsnb,E_rec_array_tmp))
									cos_array_dsnb=np.concatenate((cos_array_dsnb,cos_array_tmp))
								t_array_dsnb=t_array_dsnb[:NN_dsnb]
								E_rec_array_dsnb=E_rec_array_dsnb[:NN_dsnb]
								cos_array_dsnb=cos_array_dsnb[:NN_dsnb]
		
						###stick together in correct order
						###
						E_rec_nu=np.zeros(NN_tot)
						cos_nu=np.zeros(NN_tot)
						t_nu=np.zeros(NN_tot)
		
						if solar_length>1:
							ij=np.nonzero(N_solar>0.5)
							E_rec_nu[ij]=E_rec_array_solar
							cos_nu[ij]=cos_array_solar
							t_nu[ij]=t_array_solar
						if atmo_length>1:
							ij=np.nonzero(N_atmo>0.5)
		                                        E_rec_nu[ij]=E_rec_array_atmo
							cos_nu[ij]=cos_array_atmo
							t_nu[ij]=t_array_atmo
						if dsnb_length>1:
							ij=np.nonzero(N_dsnb>0.5)
		                                        E_rec_nu[ij]=E_rec_array_dsnb
							cos_nu[ij]=cos_array_dsnb
		                                	t_nu[ij]=t_array_dsnb
		
						ij=np.nonzero(t_nu)
						t_nu=t_nu[ij]
						cos_nu=cos_nu[ij]
						E_rec_nu=E_rec_nu[ij]
		
		                                del t_array_solar
						del t_array_atmo
						del t_array_dsnb
		                                del E_rec_array_solar
						del E_rec_array_atmo
						del E_rec_array_dsnb
			                        del cos_array_solar
						del cos_array_atmo
						del cos_array_dsnb
		
					###calculate Pb_SB and Psb_SB
					###
					mu_nu=np.zeros((N_sim,N_Q))
					for ii in range (9):
						if N_sun_arr[ii]>0.:
							N_nu_solar=np.random.normal(N_sun_arr[ii],nu_sigma[0]*N_sun_arr[ii],(N_sim,N_Q))
							N_nu_solar=np.where(N_nu_solar<0.,0.,N_nu_solar)
							mu_nu+=N_nu_solar
					for ii in range (4):
						if N_atmo_arr[ii]>0.:
							N_nu_atmo=np.random.normal(N_atmo_arr[ii],nu_sigma[1]*N_atmo_arr[ii],(N_sim,N_Q))
							N_nu_atmo=np.where(N_nu_atmo<0.,0.,N_nu_atmo)
							mu_nu+=N_nu_atmo
					for ii in range (3):
						if N_dsnb_arr[ii]>0.:
							N_nu_dsnb=np.random.normal(N_dsnb_arr[ii],nu_sigma[2]*N_dsnb_arr[ii],(N_sim,N_Q))
							N_nu_dsnb=np.where(N_nu_dsnb<0.,0.,N_nu_dsnb)
							mu_nu+=N_nu_dsnb
					if test==1:
		                                n_nu_arr=N_nu_avg*np.ones(N_sim,dtype=int)
		
		                                E_rec_nu=2.*E_thr*np.ones(N_sim*N_nu_avg)
		                                cos_nu=0.5*np.ones(N_sim*N_nu_avg)
		                                t_nu=100.*np.ones(N_sim*N_nu_avg)
		                                mu_nu=N_nu_avg*np.ones((N_sim,N_Q))
		                                NN_nu=N_nu_avg*N_sim
			
					mu_nu=np.hstack(mu_nu)
					n_nu_arr=np.tile(n_nu_arr,N_Q)
					n_array_DM=np.tile(n_array_DM,N_Q)
		
		#			###Poisson distributions
		#			###
		#			poisson_B=-mu_nu+(n_nu_arr+n_array_DM)*np.log(mu_nu)\
		#					-scipy.special.gammaln(n_nu_arr+n_array_DM+1)
		#			poisson_SB=-(mu_DM+mu_nu)+(n_nu_arr+n_array_DM)*np.log(mu_DM+mu_nu)\
		#					-scipy.special.gammaln(n_nu_arr+n_array_DM+1)
		
					###Dark Matter
					###	(i.e. Signal)
					if print_info==1:
						print 'mu nu'
						print mu_nu[:12]
						print ''
		                                print 'n_DM'
		                                print n_array_DM[:12]
		                                print ''
		                                print 'cos_DM'
		                                print cos_array_DM[:12]
		                                print ''
		                                print 'E_rec_DM'
		                                print E_rec_array_DM[:12]
		                                print ''
					
					if channel_time==True:
						prob_DM_B=np.ones(NN_DM)
						prob_DM_SB=np.ones(NN_DM)
						if channel_Erec==True:
							dif=np.zeros((len(t_edges),len(t_array_DM)))
							for ii in range(len(t_edges)):
								dif[ii]=abs(t_edges[ii]-t_array_DM)
							dif=np.reshape(dif,(len(t_array_DM),len(t_edges)))
							id1_DM=np.argmin(dif,axis=1)
							t0_DM=t_edges[id1_DM]
							id2_DM=np.where(id1_DM==N_tt-1,id1_DM-1,0)
							id2_DM=np.where(id1_DM==0,1,id2_DM)
							
							id2_DM_tmp1=np.where(id2_DM==0,id1_DM+1,id2_DM)
							t1_DM=t_edges[id2_DM_tmp1]
							
							id2_DM_tmp2=np.where(id2_DM==0,id1_DM-1,id2_DM)
							t2_DM=t_edges[id2_DM_tmp2]
							
							id2_DM=np.where(abs(t1_DM-t_array_DM)>abs(t2_DM-t_array_DM),id2_DM_tmp2,id2_DM)
							id2_DM=np.where(abs(t1_DM-t_array_DM)<abs(t2_DM-t_array_DM),id2_DM_tmp1,id2_DM)
					                d1=abs(t_array_DM-t_edges[id1_DM])
					                d2=abs(t_array_DM-t_edges[id2_DM])
							pdf1,pdf2=np.zeros(NN_DM),np.zeros(NN_DM)
							#if channel_angle==True:
							for ii in range (N_tt):
								pdf1=np.where(id1_DM==ii,f_array[ii].ev(E_rec_array_DM,cos_array_DM),pdf1)
								pdf2=np.where(id2_DM==ii,f_array[ii].ev(E_rec_array_DM,cos_array_DM),pdf2)
							prob_DM_S_angle=pdf1+d1/(d1+d2)*(pdf2-pdf1)
							prob_DM_B_angle=Pb_ipl.ev(E_rec_array_DM,cos_array_DM)
		
							#if channel_angle==False:
							for ii in range (N_tt):
		                                                pdf1=np.where(id1_DM==ii,f_array_noangle[ii](E_rec_array_DM),pdf1)
		                                                pdf2=np.where(id2_DM==ii,f_array_noangle[ii](E_rec_array_DM),pdf2)
							prob_DM_S_erec=pdf1+d1/(d1+d2)*(pdf2-pdf1)
							prob_DM_B_erec=Pb_noangle_ipl(E_rec_array_DM)
		
						prob_DM_S_time=pdf_DM_t(t_array_DM)
						prob_DM_B_time=pdf_nu_t(t_array_DM)
		
		                                prob_DM_S_time=np.tile(prob_DM_S_time,N_Q)
		                                prob_DM_B_time=np.tile(prob_DM_B_time,N_Q)
		                                prob_DM_S_angle=np.tile(prob_DM_S_angle,N_Q)
		                                prob_DM_B_angle=np.tile(prob_DM_B_angle,N_Q)
		                                prob_DM_S_erec=np.tile(prob_DM_S_erec,N_Q)
		                                prob_DM_B_erec=np.tile(prob_DM_B_erec,N_Q)
		
						n_split=np.cumsum(n_array_DM)
						n_split=np.delete(n_split,-1)
						prob_DM_B_time=np.split(prob_DM_B_time,n_split)
						prob_DM_S_time=np.split(prob_DM_S_time,n_split)
						prob_DM_B_angle=np.split(prob_DM_B_angle,n_split)
						prob_DM_S_angle=np.split(prob_DM_S_angle,n_split)
						prob_DM_B_erec=np.split(prob_DM_B_erec,n_split)
						prob_DM_S_erec=np.split(prob_DM_S_erec,n_split)
						prob_sum_DM_SB_angle=np.zeros(N_sim*N_Q)
						prob_sum_DM_B_angle=np.zeros(N_sim*N_Q)
						prob_sum_DM_SB_erec=np.zeros(N_sim*N_Q)
						prob_sum_DM_B_erec=np.zeros(N_sim*N_Q)
		
		                                for ii in range (N_sim*N_Q):
		                                        time_info=1.+1.*mu_DM/mu_nu[ii]*prob_DM_S_time[ii]/prob_DM_B_time[ii]
		                                        angle_info=1.+1.*mu_DM/mu_nu[ii]*prob_DM_S_angle[ii]/prob_DM_B_angle[ii]
		                                        erec_info=1.+1.*mu_DM/mu_nu[ii]*prob_DM_S_erec[ii]/prob_DM_B_erec[ii]
		
		                                        time_info=np.log(time_info)
		                                        angle_info=np.log(angle_info)
		                                        erec_info=np.log(erec_info)
		
		                                        pre=-mu_DM+(n_nu_arr[ii]+n_array_DM[ii])*(np.log(mu_nu[ii])-np.log(mu_nu[ii]+mu_DM))
		
		                                        angle_info_DM.append(pre+np.sum(time_info)+np.sum(angle_info))
		                                        erec_info_DM.append(pre+np.sum(time_info)+np.sum(erec_info))
		
					###neutrinos
					###	(i.e. Background)
					if print_info==1:
		                                print 'n_nu'
		                                print n_nu_arr[:12]
		                                print ''
		                                print 'cos_nu'
		                                print cos_nu[:12]
		                                print ''
		                                print 'E_rec_nu'
		                                print E_rec_nu[:12]
		                                print ''
		
					if channel_time==True:
						prob_nu_SB=np.ones(NN_nu)
						if channel_Erec==True:
							dif=np.zeros((len(t_edges),len(t_nu)))
							for ii in range(len(t_edges)):
								dif[ii]=abs(t_edges[ii]-t_nu)
							dif=np.reshape(dif,(len(t_nu),len(t_edges)))
							id1_nu=np.argmin(dif,axis=1)
							t0_nu=t_edges[id1_nu]
							id2_nu=np.where(id1_nu==N_tt-1,id1_nu-1,0)
							id2_nu=np.where(id1_nu==0,1,id2_nu)
							
							id2_nu_tmp1=np.where(id2_nu==0,id1_nu+1,id2_nu)
							t1_nu=t_edges[id2_nu_tmp1]
							
							id2_nu_tmp2=np.where(id2_nu==0,id1_nu-1,id2_nu)
							t2_nu=t_edges[id2_nu_tmp2]
							
							id2_nu=np.where(abs(t1_nu-t_nu)>abs(t2_nu-t_nu),id2_nu_tmp2,id2_nu)
							id2_nu=np.where(abs(t1_nu-t_nu)<abs(t2_nu-t_nu),id2_nu_tmp1,id2_nu)
					                d1=abs(t_nu-t_edges[id1_nu])
					                d2=abs(t_nu-t_edges[id2_nu])
							pdf1,pdf2=np.zeros(NN_nu),np.zeros(NN_nu)
							for ii in range (N_tt):
								pdf1=np.where(id1_nu==ii,f_array[ii].ev(E_rec_nu,cos_nu),pdf1)
								pdf2=np.where(id2_nu==ii,f_array[ii].ev(E_rec_nu,cos_nu),pdf2)
							prob_nu_B_angle=Pb_ipl.ev(E_rec_nu,cos_nu)
							prob_nu_S_angle=pdf1+d1/(d1+d2)*(pdf2-pdf1)
		
							for ii in range (N_tt):
		                                                pdf1=np.where(id1_nu==ii,f_array_noangle[ii](E_rec_nu),pdf1)
		                                                pdf2=np.where(id2_nu==ii,f_array_noangle[ii](E_rec_nu),pdf2)
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
						prob_nu_B_time=np.split(prob_nu_B_time,n_split)
						prob_nu_S_time=np.split(prob_nu_S_time,n_split)
						prob_nu_B_angle=np.split(prob_nu_B_angle,n_split)
						prob_nu_S_angle=np.split(prob_nu_S_angle,n_split)
						prob_nu_B_erec=np.split(prob_nu_B_erec,n_split)
						prob_nu_S_erec=np.split(prob_nu_S_erec,n_split)
		
		                                for ii in range (N_sim*N_Q):
		                                        time_info=1.+1.*mu_DM/mu_nu[ii]*prob_nu_S_time[ii]/prob_nu_B_time[ii]
		                                        angle_info=1.+1.*mu_DM/mu_nu[ii]*prob_nu_S_angle[ii]/prob_nu_B_angle[ii]
		                                        erec_info=1.+1.*mu_DM/mu_nu[ii]*prob_nu_S_erec[ii]/prob_nu_B_erec[ii]
		
		                                        time_info=np.log(time_info)
		                                        angle_info=np.log(angle_info)
		                                        erec_info=np.log(erec_info)
		
		                                        angle_info=np.sum(time_info)+np.sum(angle_info)
		                                        erec_info=np.sum(time_info)+np.sum(erec_info)
		
		                                        Q_SB_angle=np.concatenate((Q_SB_angle,angle_info+angle_info_DM))
		                                        Q_SB_erec=np.concatenate((Q_SB_erec,erec_info+erec_info_DM))
		
				###calculate the values of the test statistics
				###
				for kk in range (2):
					if kk==0:
						Q_B=-2*Q_B_angle
						Q_SB=-2*Q_SB_angle
					if kk==1:
						Q_B=-2*Q_B_erec
						Q_SB=-2*Q_SB_erec
					
					if print_info==1:
						print 'Q B'
						print Q_B[:12]
						print ''
						print 'Q SB'
						print Q_SB[:12]
						print ''
						#plt.figure(1)
						#plt.hist(Q_B,bins=np.linspace(min(Q_B),max(Q_B),100))
						#plt.figure(3)
						#plt.hist(Q_SB,bins=np.linspace(min(Q_SB),max(Q_SB),100))
						#print 'Next two only non-zero if Q_B>75'
						#out1=np.where(Q_B>75,Psb_array_B,0)
						#print 'Psb array nu'
						#print out1
						#out1=np.where(Q_B>75,Pb_array_B,0)
						#print 'Pb array nu'
						#print out1
						#print''
					if test==1:
						if Q_B[1] < (-36.) and Q_B[1] > (-39.):
							print '		Q_B OK!'
						if Q_SB[1] < (-50.) and Q_SB[1] > (-53.):
							print '		Q_SB OK!'
							#print Q_B,Q_SB
						else : 
							print 'some error!'
							print Q_B[:10], Q_SB[:10]
							
						sys.exit()
			
					###Now, do statistics with it
					###
					hist_B,Q_grid_B=np.histogram(Q_B,bins=50,normed=True)
			                hist_SB,Q_grid_SB=np.histogram(Q_SB,bins=50,normed=True)
	
					Q_min=np.min(Q_grid_SB)
					Q_max=np.max(Q_grid_B)
		
					Q_int=np.linspace(Q_min,Q_max,steps)
					dQ_int=1.*(Q_max-Q_min)/steps
				
					###The distributions with correct normalisation
					###
					def pdf_B(Qv):
						if Qv<np.min(Q_grid_B) or Qv>np.max(Q_grid_B) or Qv==np.max(Q_grid_B):
							return 0.
						Qv=np.array([Qv,Qv])
						id=np.digitize(Qv,bins=Q_grid_B)
						id=id[0]-1
						return hist_B[id]
			
					def pdf_SB(Qv):
						if Qv<np.min(Q_grid_SB) or Qv>np.max(Q_grid_SB):
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
						
						if min(Q_grid_SB)<min(Q_grid_B):
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
					#	qq=np.linspace(min(Q_grid_SB),max(Q_grid_B),250)
					#	plt.figure(2)	
					#	plt.hist(Q_B,bins=Q_grid_B,facecolor='b',alpha=0.5,normed=1.)
					#	plt.hist(Q_SB,bins=Q_grid_SB,facecolor='r',alpha=0.5,normed=1.)
					#	#for ii in range (len(qq)):
					#	#	plt.plot(qq[ii],pdf_B(qq[ii]),'bo')
					#	#	plt.plot(qq[ii],pdf_SB(qq[ii]),'ro')
					#	plt.show(all)
		
					if(len(cl)==0):
						cl_avg=0.
					else:
						cl_avg=1.*np.sum(cl)/len(cl)
					if kk==0:
						cl_angle=cl_avg
					if kk==1:
						cl_erec=cl_avg
		
		
					
				###write to file
				###
				f=open(filename_dm,'a')
				f.write(str(M_det)+ ' '+str(N_nu_exp)+' '+str(m_DM_array[mm])+' '+str(sigma[ss])+\
						' '+str(N_DM_exp)+' '+str(cl_angle)+' '+str(cl_erec)+'\n')
				f.close()
			
			if cl_angle<0.0001 and mem1==0:#and cl_erec<0.1 
				waszero=True
				mem1=1
		
			if cl_angle>0.1 and mem2==0:#and cl_erec>0.1 
				wasbig=True
				mem2=1
	
			if cl_angle>0.1  and waszero==True:#and cl_erec>0.1
				scan=False
	
			if wasbig==True and waszero==True:
				scan=False

			scanned=np.concatenate((scanned,np.array([s_int])))
	
			if waszero==True:
				s_int+=1
	
			if waszero==False:
				s_int-=1
	
			if scan==False:
				if waszero==True and wasbig==False:
					s_int+=1
				else:
					s_int-=1

#			print s_int
#			print scanned
#			print scan
#			print ''
print ''
print ''
print ''
print '####################################################################'
print 'END'
print '####################################################################'
print ''
