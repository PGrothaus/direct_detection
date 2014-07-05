import numpy as np
from math import pi,sin,cos,exp,log10
import math
#import matplotlib.pyplot as plt
from scipy.interpolate import interp1d,RectBivariateSpline
from constants import *
import ephem
import random as rnd
	
#normalisations (no spectral lines yet)
pp_norm=5.99E10
pep_norm=1.42E8
hep_norm=7.93E3
b8_norm=5.69E6
n13_norm=3.07E8
o15_norm=2.33E8
f17_norm=3.25E6
b7_norm=4.84E9

###including the branching ratios for the 7Be de excitation
###and a factor 100 for keeping the pdf for the lines
###normalised when converting keV into MeV
###
norm_vector=[pp_norm,hep_norm,b8_norm,n13_norm,o15_norm,f17_norm,1000.*0.9*b7_norm,1000.*0.1*b7_norm,1000.*1.E7]

################
#cross sections#
################
def d_sigma_CNS(E_nu,rec_E):#CNS cross section for given neutrino and recoil energy
    sigma=GF**2./4./pi*Qw**2.*mT*(1.-mT*rec_E/2./E_nu/E_nu)*form2(rec_E)
    if (sigma<0.): return 0.
    return sigma/1.e6*GeVcm**2.#cm^2/keV
    
def Er_max(E_nu):#maximal recoil energy from CNS event
#    return 2.*E_nu**2./(mT*1.e3+2.*E_nu)*1.e3#in keV
	return 2.*E_nu**2./(mT*1.e3)*1.e3#in keV

def Enu_min_CNS(rec_E):#necessary neutrino energy to get a CNS recoil of rec_E
    return sqrt(mT*rec_E/2.)#MeV
    
########
#fluxes#
########
def get_spectrum_solar(typ):#reads in neutrino spectra
    name='data/solarspectrum'+str(typ)+'.txt'
    data=np.loadtxt(name)
    energy=data[:,0]
    spectrum=data[:,1]
    norm=norm_vector[typ]
    return energy,spectrum,norm
    
def get_flux_solar(E_nu,typ):#returns interpolated neutrino flux at E_nu from flux 
	energy,flux,norm=get_spectrum_solar(typ)
	Emin=min(energy)
	Emax=max(energy)
	flux_ipl=interp1d(energy,flux,kind='linear')
	t_var=np.where(E_nu<Emin,0.,1.)
	t_var=np.where(E_nu>Emax,0.,t_var)
	E_nu=np.where(E_nu<Emin,Emin,E_nu)
	E_nu=np.where(E_nu>Emax,Emax,E_nu)
	flux_out=t_var*flux_ipl(E_nu)
	return flux_out*norm#in (MeV cm^2 s)^-1

def solar_flux(E_nu):
	tot_flux=np.zeros(len(E_nu))
	for jj in range (9):
		tot_flux+=get_flux_solar(E_nu,jj)
	return tot_flux

def Enu_max_solar(typ):#finds maximal neutrino energy from neutrino flux 
    energy,flux,norm=get_spectrum_solar(typ)
    maximE=max(energy)
    return maximE#MeV

def get_spectrum_atmo(typ):#reads in neutrino spectra
    name='data/atmospectrum'+str(typ)+'.txt'
    data=np.loadtxt(name)
    return data[:,0],data[:,1]

def get_flux_atmo(E_nu,typ):
	energy,flux=get_spectrum_atmo(typ)
	Emin=min(energy)
	Emax=max(energy)
	flux_ipl=interp1d(energy,flux,kind='linear')
	t_var=np.where(E_nu<Emin,0.,1.)
	t_var=np.where(E_nu>Emax,0.,t_var)
	E_nu=np.where(E_nu<Emin,Emin,E_nu)
	E_nu=np.where(E_nu>Emax,Emax,E_nu)
	flux_out=t_var*flux_ipl(E_nu)
	return flux_out#in (MeV cm^2 s)^-1

def atmo_flux(E_nu):
        tot_flux=np.zeros(len(E_nu))
        for jj in range (4):
                tot_flux+=get_flux_atmo(E_nu,jj)
        return tot_flux

def Enu_max_atmo(typ):#finds maximal neutrino energy from neutrino flux 
    energy,flux=get_spectrum_atmo(typ)
    maximE=max(energy)
    return maximE#MeV

def get_spectrum_dsnb(typ):
	name='data/dsnbspectrum'+str(typ)+'.txt'
	data=np.loadtxt(name)
	return data[:,0],data[:,1]

def get_flux_dsnb(E_nu,typ):
	energy,flux=get_spectrum_dsnb(typ)
	Emin=min(energy)
	Emax=max(energy)
	norm=1.
	if typ==2: 
		norm=4.
	flux_ipl=interp1d(energy,flux,kind='linear')
	t_var=np.where(E_nu<Emin,0.,1.)
	t_var=np.where(E_nu>Emax,0.,t_var)
	E_nu=np.where(E_nu<Emin,Emin,E_nu)
	E_nu=np.where(E_nu>Emax,Emax,E_nu)
	flux_out=t_var*flux_ipl(E_nu)
	return norm*flux_out#in (MeV cm^2 s)^-1

def dsnb_flux(E_nu):
	tot_flux=np.zeros(len(E_nu))
        for jj in range (3):
                tot_flux+=get_flux_dsnb(E_nu,jj)
        return tot_flux


def tot_flux(E_nu):
        tot_flux=0.
        for jj in range (8):
                tot_flux+=get_flux_solar(E_nu,jj)
	for jj in range (4):
		tot_flux+=get_flux_atmo(E_nu,jj)
	for jj in range (3):
		tot_flux+=get_flux_dsnb(E_nu,jj)
        return tot_flux
	
###############
#recoil events#
###############
def get_rate_solar(typ):#reads in neutrino spectra
    name='data/solarrate'+str(typ)+'_'+str(Z)+'.txt'
    data=np.loadtxt(name)
    energy=data[:,0]
    diffrate=data[:,1]
    n_events=data[:,2]
    return energy,diffrate,n_events
    
def get_diff_rate_solar(E_rec,typ):#returns interpolated diff event rate from neutrino flux 'data'
    energy,diffrate,n_events=get_rate_solar(typ)
    rate_ipl=interp1d(energy,diffrate,kind='linear')
    Emin=min(energy)
    Emax=max(energy)
    t_var=np.where(E_rec<Emin,0.,1.)
    t_var=np.where(E_rec>Emax,0.,t_var)
    E_rec=np.where(E_rec<Emin,Emin,E_rec)
    E_rec=np.where(E_rec>Emax,Emax,E_rec)
    return t_var*rate_ipl(E_rec)

def get_n_events_solar(E_thr,typ):#returns total number of events per s from neutrino-CNS
                              #for a given threshold energy E_thr
                              #including all neutrino fluxes via an interpolation
    energy,diffrate,n_evt=get_rate_solar(typ)
    if(E_thr>max(energy)):
        return 0.
    evt_ipl=interp1d(energy,n_evt,kind='linear')
    return evt_ipl(E_thr)#per second
    
def get_rate_atmo(typ):#reads in neutrino spectra
    name='data/atmorate'+str(typ)+'_'+str(Z)+'.txt'
    data=np.loadtxt(name)
    energy=data[:,0]
    diffrate=data[:,1]
    n_events=data[:,2]
    return energy,diffrate,n_events
    
def get_diff_rate_atmo(E_rec,typ):#returns interpolated diff event rate from neutrino flux 'data'
    energy,diffrate,n_events=get_rate_atmo(typ)
    rate_ipl=interp1d(energy,diffrate,kind='linear')
    Emin=min(energy)
    Emax=max(energy)
    t_var=np.where(E_rec<Emin,0.,1.)
    t_var=np.where(E_rec>Emax,0.,t_var)
    E_rec=np.where(E_rec<Emin,Emin,E_rec)
    E_rec=np.where(E_rec>Emax,Emax,E_rec)
    return t_var*rate_ipl(E_rec)

def get_n_events_atmo(E_thr,typ):#returns total number of events per s from neutrino-CNS
                    #for a given threshold energy E_thr
                    #including all neutrino fluxes via an interpolation
    energy,diffrate,n_evt=get_rate_atmo(typ)
    if(E_thr>max(energy)):
        return 0.
    evt_ipl=interp1d(energy,n_evt,kind='linear')
    return evt_ipl(E_thr)#per second

def get_rate_dsnb(typ):#reads in neutrino spectra
    name='data/dsnbrate'+str(typ)+'_'+str(Z)+'.txt'
    data=np.loadtxt(name)
    energy=data[:,0]
    diffrate=data[:,1]
    n_events=data[:,2]
    return energy,diffrate,n_events
    
def get_diff_rate_dsnb(E_rec,typ):#returns interpolated diff event rate from neutrino flux 'data'
    energy,diffrate,n_events=get_rate_dsnb(typ)
    rate_ipl=interp1d(energy,diffrate,kind='linear')
    Emin=min(energy)
    Emax=max(energy)
    t_var=np.where(E_rec<Emin,0.,1.)
    t_var=np.where(E_rec>Emax,0.,t_var)
    E_rec=np.where(E_rec<Emin,Emin,E_rec)
    E_rec=np.where(E_rec>Emax,Emax,E_rec)
    return t_var*rate_ipl(E_rec)

def get_n_events_dsnb(E_thr,typ):#returns total number of events per s from neutrino-CNS
                    #for a given threshold energy E_thr
                    #including all neutrino fluxes via an interpolation
    energy,diffrate,n_evt=get_rate_dsnb(typ)
    if(E_thr>max(energy)):
        return 0.
    evt_ipl=interp1d(energy,n_evt,kind='linear')
    return evt_ipl(E_thr)#per second


def form2(E_rec):#returns form factor squared at a given recoil energy
    if E_rec==0.:
	return 0.
    q=sqrt(2.*mT*1.e6*E_rec)*keVfm
    j1=(sin(q*rn)-q*rn*cos(q*rn))/q/q/rn/rn
    ff2=((3.*j1/q/rn)*exp(-0.5*q**2*sf**2))**2
    return ff2

def form2_vector(E_rec):#returns form factor squared at a given recoil energy
	E_rec=np.where(E_rec<1.E-3,1.E-6,E_rec)
	q=np.sqrt(2.*mT*1.e6*E_rec)*keVfm
	j1=(np.sin(q*rn)-q*rn*np.cos(q*rn))/q/q/rn/rn
	ff2=(3.*j1/q/rn*np.exp(-0.5*q**2*sf**2))**2
	return ff2


def tot_rate_nu(M,E_thr,t0,t1,steps=365):
	###initialise observer
	###
	sun=ephem.Sun()
	obs=ephem.Observer()
	obs.pressure=0.
	obs.temp=0.
	obs.lon='0.'
	obs.lat='0.'
	obs.elevation=0.

        t0_f=float(t0)
        t1_f=float(t1)
        time_grid=np.linspace(t0_f,t1_f,steps)
	d_time=(t1_f-t0_f)/steps
        rate=np.zeros(steps)
        time=np.zeros(steps)
	time[0]=0.
	rate_temp0=0.
        for typ in range(4):
                rate_temp0+=get_n_events_atmo(E_thr,typ)
        atmorate=rate_temp0

        for typ in range(3):
                rate_temp0+=get_n_events_dsnb(E_thr,typ)
        dsnbrate=rate_temp0-atmorate

	###need this loop because ephem can't handle arrays
	###
	if nu_mod==True:
		if resolution_energy==False or resolution_energy==True:
		        ###Atmo and DSNB neutrino event rates are time independent
		        ###
			dsnbrate=np.zeros(3)
			atmorate=np.zeros(4)
		        
			for typ in range(4):
		                atmorate[typ]=get_n_events_atmo(E_thr,typ)*(t1_f-t0_f)*days*M/A*N0
		
		        for typ in range(3):
		                dsnbrate[typ]=get_n_events_dsnb(E_thr,typ)*(t1_f-t0_f)*days*M/A*N0

			rate[0]=0.
			alpha=np.zeros(steps)
			solarrate=np.zeros(9)
			for typ in range (9):
				for ii in range (steps-1):
			                time1_f=time_grid[ii+1]
			                dT=time_grid[ii+1]-time_grid[ii]
			                time1=ephem.date(time1_f)
					obs.date=time1_f
					sun.compute(obs)
					d=sun.earth_distance#in units of AE
				        alpha[ii+1]=1./(d**2)
					rate_tmp=get_n_events_solar(E_thr,typ)*alpha[ii+1]
			                rate[ii+1]+=rate_tmp*dT*days*M/A*N0
					solarrate[typ]+=rate_tmp*dT*days*M/A*N0
			                time[ii+1]=time1_f-t0_f
			rate=rate+(np.sum(atmorate)+np.sum(dsnbrate))/((t1_f-t0_f))*dT
			rate[0]=0.
			summed_rate=np.sum(rate)
			sum_sun=solarrate
			sum_atmo=atmorate
			sum_dsnb=dsnbrate
			alpha[0]=alpha[1]

	#	if resolution_energy==True:
	#		E_evt=np.logspace(log10(E_thr/10.),log10(10.*upper_threshold),steps)
	#                E_rec=np.logspace(log10(E_thr),log10(upper_threshold),steps)
	#
	#                sigmaE = lambda E: energy_res*(np.sqrt(E))
	#                smearing = lambda E,mu: 1./(sqrt(2.*pi)*sigmaE(E))*\
	#                                        np.exp(-(E-mu)**2/(2.*sigmaE(E))**2)
	#
	#                data_eff=np.loadtxt(eff_name)
	#                E_eff=data_eff[:,0]
	#                eff=data_eff[:,1]
	#                eff_ipl=interp1d(E_eff,eff,kind='linear')
	#
	#                dE_evt=np.delete(E_evt,0)-np.delete(E_evt,-1)
	#                E_evt=0.5*(np.delete(E_evt,0)+np.delete(E_evt,-1))
	#                dE_rec=np.delete(E_rec,0)-np.delete(E_rec,-1)
	#                E_rec=0.5*(np.delete(E_rec,0)+np.delete(E_rec,-1))

	#                if rec_eff==True:
	#                        eff1=eff_ipl(E_rec)
	#                else:
	#                        eff1=np.ones(len(E_rec))

	#                integral=0.
	#		rate=np.zeros(steps)
	#		###time independent DSNB and atmo rate
	#		###
	#		dsnbrate=np.zeros((3,len(E_evt)))
	#		atmorate=np.zeros((4,len(E_evt)))
	#		for jj in range (3):
	#			dsnbrate[jj]=get_diff_rate_dsnb(E_evt,jj)
	#		for jj in range (4):
        #                        atmorate[jj]=get_diff_rate_atmo(E_evt,jj)
	#		rate[0]=0.
	#		rate_solar=np.zeros(steps)
	#		alpha=np.zeros(steps)
	#		sum_sun,sum_atmo,sum_dsnb=np.zeros(9),np.zeros(4),np.zeros(3)
	#		for ii in range(steps-1):
	#			time1_f=time_grid[ii+1]
        #                        dT=time_grid[ii+1]-time_grid[ii]
        #                        time1=ephem.date(time1_f)
        #                        obs.date=time1_f
        #                        sun.compute(obs)
        #                        d=sun.earth_distance#in units of AE
        #                        alpha[ii+1]=1./(d**2)
	#			integral_atmo,integral_dsnb,integral_solar=np.zeros(4),np.zeros(3),np.zeros(9),
	#			solarrate=np.zeros((9,len(E_evt)))
	#			for jj in range (9):
        #        	                solarrate[jj]=get_diff_rate_solar(E_evt,jj)*alpha[ii+1]
	#			for kk in range (4):
	#				for jj in range (len(E_evt)):
	#		                        integral_atmo[kk]+=np.sum(dE_evt[jj]*dE_rec*eff1*smearing(E_rec,E_evt[jj])*atmorate[kk][jj])*dT*days*M/A*N0

	#			for kk in range (3):
	#				for jj in range (len(E_evt)):
	#		                        integral_dsnb[kk]+=np.sum(dE_evt[jj]*dE_rec*eff1*smearing(E_rec,E_evt[jj])*dsnbrate[kk][jj])*dT*days*M/A*N0

	#			for kk in range (9):
	#				for jj in range (len(E_evt)):
	#		                        integral_solar[kk]+=np.sum(dE_evt[jj]*dE_rec*eff1*smearing(E_rec,E_evt[jj])*solarrate[kk][jj])*dT*days*M/A*N0*alpha[ii+1]

	#			sum_sun+=integral_solar
	#			sum_atmo+=integral_atmo
	#			sum_dsnb+=integral_dsnb
	#			integral=np.sum(integral_atmo)+np.sum(integral_solar)+np.sum(integral_dsnb)
	#			rate_solar[ii+1]=np.sum(integral_solar)
	#			rate[ii+1]=integral
	#			time[ii+1]=time1_f-t0_f
	#		alpha[0]=alpha[1]
			
		return time,rate,d_time,sum_sun,sum_atmo,sum_dsnb

	###if no annual modulation is considered
	###
	if nu_mod==False:
		rate_temp=rate_temp0
		for typ in range(8):
			rate_temp=rate_temp+get_n_events_solar(E_thr,typ)
	        summed_rate=rate_temp*(t1_f-t0_f)*days*M/A*N0
		alpha=np.zeros(steps)
        return time,rate,summed_rate,d_time,atmorate*(t1_f-t0_f)*days*M/A*N0,dsnbrate*(t1_f-t0_f)*days*M/A*N0,alpha


def test_totrate_nu():
	t0=ephem.date('2014/01/01')
	t1=ephem.date('2015/01/01')
	M_det=500.e6
	time_arr,rate_arr,N_nu,dt,N_atmo,N_dsnb,alpha=tot_rate_nu(M_det,E_thr,t0,t1)
	N_solar=N_nu-N_atmo-N_dsnb
	print N_solar,N_atmo,N_dsnb,N_nu
#	plt.plot(time_arr,rate_arr,'ro')
#	plt.show()

#test_totrate_nu()

#tmp,rate,summed_rate,tmp,atmo,dsnb=tot_rate_nu(500.e6,3.,ephem.date('2014/01/01'),ephem.date('2015/01/01'))
#print summed_rate
#print atmo,dsnb

def get_rnd_E_nu(E_thr, N_points=500):
        ###to draw random neutrino energies
	###
        name='data/rnd_Enu_'+str(E_thr)+'keV_'+str(Z)+'.txt'
        try:
                with open(name):
                        #print "File exists!"
                        return
        except IOError:
                Emin=Enu_min_CNS(E_thr)
                Emax=1000.#manually set, flux gets too low,

                if tot_flux(Emin)==0.:
#                        print "no events from neutrinos!"
                        return -1
                E_grid=np.logspace(log10(Emin),log10(Emax),N_points)
                F_grid=np.zeros(N_points)
                bin_prob=np.zeros(N_points-1)
                prob_dis=np.zeros(N_points)
                acc_prob=np.zeros(N_points)
		
                test=0
                for var in range (N_points):
                        if test==1:
                                break
                        else:
                                F_grid[var]=tot_flux(E_grid[var])
				prob_dis[var]=F_grid[var]
                                if F_grid[var]==0:
                                        test=1
		F_tot=0.
                for var in range (N_points-1):
                        dE=E_grid[var+1]-E_grid[var]
                        bin_prob[var]=F_grid[var]*dE
                        acc_prob[var]=sum(bin_prob[:var])
			F_tot+=bin_prob[var]
		
		shift=np.zeros(1)
		bin_prob=np.concatenate((bin_prob,shift))
		bin_prob=bin_prob/F_tot
		acc_prob=acc_prob/F_tot
		acc_prob[-1]=1.
		prob_dis=prob_dis/F_tot
                for var in range (N_points):
                        f=open(name,"a")
                        f.write(str(E_grid[var])+' '+str(F_grid[var])+' '\
                                        +str(bin_prob[var])+' '+str(prob_dis[var])+' '+str(acc_prob[var])+'\n')
                        f.close()
###
#               plt.figure("E_nu prob distribution")
#               plt.yscale('log')
#               plt.xscale('log')
#               plt.plot(E_grid[:-1],bin_prob[:],'g.')
#               plt.plot(E_grid[:-1],prob_dis[:],'b.')
#               plt.plot(E_grid[:-1],acc_prob[:],'r.')
#               plt.show()              
###
                return


###create lookup table for E_nu and cos theta
###
def create_Enu_cos(N_E=300,N_cos=100,N_r=100):
	###get recoil energy from angle of recoiling nucleus
	###	(this is different from the one appearing in the differential cross section)
	E_array=np.logspace(0.,3.,N_E)
        cos_nu_array=np.linspace(-1.,1.,N_cos)#can't scatter in backward direction of incoming neutrino
	r_array=np.linspace(0.,1.,N_r)
	d_cos=(max(cos_nu_array)-min(cos_nu_array))/N_cos

	E_cos_grid=np.meshgrid(E_array,cos_nu_array)
	shift=np.zeros(1)

	f=open('data/cos_theta_table.txt','w')
	f.close()
	f=open('data/cos_theta_table.txt','a')
	cos_out=np.zeros((N_E,N_r))
	###need this loop because my functions can't handle meshgrids
	###
	for ii in range (N_E):
		E_nu=E_array[ii]
#		erec_array=erec(cos_array,E_nu)
		erec_array=E_nu**2/mT/1.e3*(1.-cos_nu_array)*1.e3
		form2_array=form2_vector(erec_array)
		###need angle between in and outgoing neutrino direction here!
		###	(Drukier and Stodolsky)
#		cos_nu_array=(E_nu**2-erec_array*mT)/(E_nu**2)
		sigma_array=E_nu**2*(1.+cos_nu_array)*form2_array
		
		bin_array=sigma_array*d_cos
#		plt.plot(cos_nu_array,bin_array/sum(bin_array)/d_cos,'ro')
#		plt.show()
		bin_array=np.delete(bin_array,0)
		norm=np.sum(bin_array)
		acc_prob_array=np.cumsum(bin_array)/norm
		acc_prob_array=np.concatenate((shift,acc_prob_array))
		cos_ipl=interp1d(acc_prob_array,cos_nu_array,kind='linear')
		cos_out_tmp=cos_ipl(r_array)
		nan_array=np.isnan(cos_out_tmp)
		cos_out_tmp=np.where(nan_array==True,0.,cos_out_tmp)
		for jj in range (N_r):
			cos_out[ii][jj]=cos_out_tmp[jj]
#		for jj in range (N_r):
#			f.write(str(cos_out[jj])+' ')
#	f.close()
#	for ii in range (len(E_array)-1):
#		dE=E_array[ii+1]-E_array[ii]
#		E_array[ii]=E_array[ii]+0.5*dE
#	r_array=r_array+0.5*1./N_r
#	E_out=np.delete(E_array,-1)
#	r_out=np.delete(r_array,-1)
		
	return E_array,r_array,cos_out


###probability of E_nu coming from the sun
###
def prob_being_solar(E_thr,array_length=5000):
	Emin=Enu_min_CNS(E_thr)
        E_grid=np.logspace(log10(Emin),3,array_length)
        flux_s=solar_flux(E_grid)
        flux_a=atmo_flux(E_grid)
        flux_d=dsnb_flux(E_grid)
        prob=flux_s/(flux_s+flux_a+flux_d)
        return E_grid,prob 
#	E_grid=np.logspace(-3,3,array_length)
#	flux_s=solar_flux(E_grid)
#	flux_a=atmo_flux(E_grid)
#	flux_d=dsnb_flux(E_grid)
#	prob=[]#flux_s/(flux_s+flux_a+flux_d)
#	alpha_grid=np.linspace(0.9,1.1,100)
#	for ii in range (len(alpha_grid)):
#		prob.append(flux_s/(flux_s*alpha_grid[ii]+flux_a+flux_d)*alpha_grid[ii])
#	prob=np.hstack(prob)
#	prob=np.reshape(prob,(6000,100))
#	return E_grid,alpha_grid,prob 



#ee,prob=prob_being_solar()
#plt.plot(ee,1.-prob,'ro')
##plt.yscale('log')
#plt.xscale('log')
#plt.show()

###mode=0: creating pdf's
###mode=1: simulating events
###
def recoil_nu(E_thr,typ,N_min=750000,array_length=1000,N_evt=1,mode=0,):
	if N_min==0: N_min=1
	###	(typ=0 solar B8, typ=1 atmo, typ=2 dsnb, typ=3 solar hep)
        ###Euler Rodrigues rotation
        ###
	t0=ephem.date('2014/01/01')
	t1=ephem.date('2015/01/01')
	t0_f=float(t0)
	t1_f=float(t1)
	t1_f-=t0_f
	np.random.seed()

        def rotation_matrix(axis,theta):
            axis = axis/math.sqrt(np.dot(axis,axis))
            a = math.cos(theta/2)
            b,c,d = -axis*math.sin(theta/2)
            return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                             [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                             [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

        def erec(cos_theta,Enu):
                erec1=2.*A*1.e3*Enu**2*cos_theta**2/((A*1.e3+Enu)**2-Enu**2*cos_theta**2)*1.e3
                return erec1


        sun=ephem.Sun()
        obs=ephem.Observer()
        obs.pressure=0.
        obs.temp=0.
        obs.lon='0.'
        obs.lat='0.'
        obs.elevation=0.
	obs.date=0.#everything for the solar neutrinos is time independent

        sun.compute(obs)
        ra_sun=sun.ra
        dec_sun=sun.dec

	phi_sun=ra_sun
	theta_sun=0.5*pi-dec_sun

        x_sun=sin(theta_sun)*cos(phi_sun)
        y_sun=sin(theta_sun)*sin(phi_sun)
        z_sun=cos(theta_sun)
	v_sun=np.array([x_sun,y_sun,z_sun])
	axis=np.array([1.,0.,-1.*x_sun/z_sun])

	data_eff=np.loadtxt(eff_name)
	E_eff=data_eff[:,0]
	eff=data_eff[:,1]
	eff_ipl=interp1d(E_eff,eff,kind='linear')

	E_nu=np.zeros(array_length)
	get_rnd_E_nu(E_thr)

#	name='data/rnd_Enu_'+str(E_thr)+'keV_'+str(Z)+'.txt'
#	data_E_nu=np.loadtxt(name)
#	E_nu_grid=data_E_nu[:,0]
#	acc_prob_E_nu=data_E_nu[:,4]
#	E_nu_ipl=interp1d(acc_prob_E_nu,E_nu_grid,kind='linear')#checked

	###create cpdf for neutrino energies
	###	(do for all three types seperately)
	if typ==0:
		E_grid=np.logspace(log10(Enu_min_CNS(E_thr)),log10(20.),1000)
		flux_grid=get_flux_solar(E_grid,1)
		dE_grid=np.delete(E_grid,0)-np.delete(E_grid,-1)
		avg_flux=0.5*(np.delete(flux_grid,0)+np.delete(flux_grid,-1))
		norm=np.sum(avg_flux*dE_grid)	
		bin_prob=avg_flux*dE_grid/norm
		acc_prob=np.cumsum(bin_prob)
		acc_prob=np.insert(acc_prob,0,0.)
		E_nu_ipl=interp1d(acc_prob,E_grid,kind='linear')
	if typ==1:
		E_grid=np.logspace(log10(Enu_min_CNS(E_thr)),log10(1000.),1000)
		flux_grid=atmo_flux(E_grid)
		dE_grid=np.delete(E_grid,0)-np.delete(E_grid,-1)
		avg_flux=0.5*(np.delete(flux_grid,0)+np.delete(flux_grid,-1))
		norm=np.sum(avg_flux*dE_grid)	
		bin_prob=avg_flux*dE_grid/norm
		acc_prob=np.cumsum(bin_prob)
		acc_prob=np.insert(acc_prob,0,0.)
		E_nu_ipl=interp1d(acc_prob,E_grid,kind='linear')
	if typ==2:
		E_grid=np.logspace(log10(Enu_min_CNS(E_thr)),log10(100.),1000)
		flux_grid=dsnb_flux(E_grid)
		dE_grid=np.delete(E_grid,0)-np.delete(E_grid,-1)
		avg_flux=0.5*(np.delete(flux_grid,0)+np.delete(flux_grid,-1))
		norm=np.sum(avg_flux*dE_grid)	
		bin_prob=avg_flux*dE_grid/norm
		acc_prob=np.cumsum(bin_prob)
		acc_prob=np.insert(acc_prob,0,0.)
		E_nu_ipl=interp1d(acc_prob,E_grid,kind='linear')
	if typ==3:
		E_grid=np.logspace(log10(Enu_min_CNS(E_thr)),log10(20.),1000)
		flux_grid=get_flux_solar(E_grid,2)
		dE_grid=np.delete(E_grid,0)-np.delete(E_grid,-1)
		avg_flux=0.5*(np.delete(flux_grid,0)+np.delete(flux_grid,-1))
		norm=np.sum(avg_flux*dE_grid)	
		bin_prob=avg_flux*dE_grid/norm
		acc_prob=np.cumsum(bin_prob)
		acc_prob=np.insert(acc_prob,0,0.)
		E_nu_ipl=interp1d(acc_prob,E_grid,kind='linear')
	x_Enu,y_r,values=create_Enu_cos()
	cos_ipl=RectBivariateSpline(x_Enu,y_r,values,kx=1,ky=1)#checked via angular distribution

	E_rec_out=[]
	cos_out=[]
	no_solar_out=[]
	theta_atmo_out=[]
	phi_atmo_out=[]

	cos_array=np.zeros(array_length)

	if mode==0:
		N_loop=N_min
	if mode==1:
		N_loop=N_evt
	rr=1.

	while(len(E_rec_out)<N_loop):

		r_array=np.random.uniform(0.,1.,array_length)
		E_nu_array=E_nu_ipl(r_array)

		r_array=np.random.uniform(0.,1.,array_length)
		cos_nu_array=cos_ipl.ev(E_nu_array,r_array)

		E_rec_array=E_nu_array**2/mT/1.e3*(1.-cos_nu_array)*1.e3
		cos_array=(E_nu_array+mT*1.e3)/E_nu_array*np.sqrt(E_rec_array/2/mT/1.e6)

                if resolution_energy==True:
			E_rec_array=np.where(E_rec_array<E_thr-1.,E_thr-1.,np.random.normal(E_rec_array,0.1*np.sqrt(E_rec_array)))

		###energy thresholds
		###
		E_rec_array=np.where(E_rec_array<E_thr,0.001,E_rec_array)
		E_rec_array=np.where(E_rec_array>upper_threshold,0.001,E_rec_array)
		E_rec_array=np.where(cos_array>1.,0.001,E_rec_array)

		###include detector efficency
		###
		if rec_eff==True:
	               r_eff=np.random.uniform(0.,1.,array_length)
	               eff_array=eff_ipl(E_rec_array)
	               eff_array=np.where(r_eff<eff_array,1.,0.)
		else:
	               eff_array=np.ones(array_length)

		E_rec_array=E_rec_array*eff_array
                E_rec_array=np.where(E_rec_array<E_thr,0.,E_rec_array)

		cos_array=np.where(E_rec_array>E_thr,cos_array,-10.)

		E_rec_out=np.concatenate((E_rec_out,E_rec_array[E_rec_array>E_thr]))
		cos_out=np.concatenate((cos_out,cos_array[cos_array>-9.]))

#                if mode==0:
#  	              if(1.*(len(E_rec_out))/N_min)>rr*0.1:
#        		        print str(100.*(len(E_rec_out))/N_min)+'%'
#		                rr+=1
#                if mode==1:
#                	if(1.*(len(E_rec_out))/N_evt)>rr*0.1:
#		                print str(100.*(len(E_rec_out))/N_evt)+'%'
#                		rr+=1

#	print 'models simulated!'
	N_sim=len(E_rec_out)
	cos_rel=np.zeros(N_sim)

	###rotation to get the recoilig nucleus vector
	###
	angle_out=np.arccos(cos_out)
	phi_out=np.random.uniform(0.,2*pi,N_sim)

	###If non-solar, get random direction
	###
	if (typ==1 or typ ==2):
		theta_nonsolar=np.random.uniform(-1.,1.,N_sim)
		phi_nonsolar=np.random.uniform(0,2.*pi,N_sim)
	        x_dir=np.sin(theta_nonsolar)*np.cos(phi_nonsolar)
	        y_dir=np.sin(theta_nonsolar)*np.sin(phi_nonsolar)
	        z_dir=np.cos(theta_nonsolar)
	
	###Now, rotate...
	###
	for ii in range (N_sim):
	        axis=np.array([1.,0.,-1.*x_sun/z_sun])
		direction=-1.*v_sun
		if (typ==1 or typ==2):
			direction=np.array([x_dir[ii],y_dir[ii],z_dir[ii]])
			axis=np.array([1.,0.,-1.*x_dir[ii]/z_dir[ii]])
		v_recoil=np.dot(rotation_matrix(axis,angle_out[ii]),direction)
		v_recoil=np.dot(rotation_matrix(direction,phi_out[ii]),v_recoil)
	
		###get relativ angle to the sun
		###   should be the same, no?!
		cos_rel[ii]=np.dot(v_recoil,v_sun)/\
			(sqrt(np.dot(v_recoil,v_recoil))*sqrt(np.dot(v_sun,v_sun)))

	if resolution_angle==True:
		angle_rel=np.arccos(cos_rel)
		angle_rel=np.random.normal(angle_rel,angle_res/np.sqrt(E_rec_out))
		cos_rel=np.cos(angle_rel)

#	print '	DONE!'
#	print 'all angles calculated'
	if mode==1:
#		print 'Now, zip it!'
		sample=zip(E_rec_out,cos_rel)
#		print 'zipped'
		np.random.shuffle(sample)
#		print 'shuffled'
		sample=sample[:N_evt]
#		print 'sampled drawn.'
		E_rec_out,cos_rel=np.array(zip(*sample))
#		print 'unzipped'
		return E_rec_out,cos_rel


	E_rec_out=E_rec_out[:N_min]
	cos_rel=cos_rel[:N_min]

	###make 2 dimensional histograms
	###
	H=np.histogram2d(E_rec_out,cos_rel,bins=(E_rec_bin,theta_bin),normed=True)[0]

#	print 'pdf generated'
#	H=np.reshape(H,(N_erec-1)*(N_theta-1))
#	plt.figure('E_rec_nu 75000')
#	bins1=np.linspace(min(E_rec_out),max(E_rec_out),25)
#	plt.hist(E_rec_out,bins=bins1,normed=True,log=True)
#	E_rec_solar=E_rec_out[no_solar_out==0]
#	print max(E_rec_solar)
#	plt.figure('cos_theta_sun_nu 75000')
#	plt.hist(cos_rel,25,normed=True,log=True)
#	plt.show(all)
#	E_rec_bin=E_rec_bin+0.5*(10.-E_thr)/N_erec
#	theta_bin=theta_bin+0.5*2./N_theta
#	E_rec_bin=np.delete(E_rec_bin,-1)
#	theta_bin=np.delete(theta_bin,-1)

	return H

def test_recoil_nu():
	t0=ephem.date('2014/01/01')
        t1=ephem.date('2015/01/01')
        M_det=1.e6
	print 'calculating rate...'
	time_arr,rate_arr,N_nu,dt,N_atmo,N_dsnb,alpha=tot_rate_nu(M_det,E_thr,t0,t1,steps=200)
	N_sun=N_nu-N_atmo-N_dsnb
	print N_nu,N_sun,N_atmo,N_dsnb,(N_atmo+N_dsnb)/N_nu
	print 'simulating events...'
	for ii in range (1,3):
		col=['ro','bo','go']
		#E_rec_out,cos_rel=recoil_nu(E_thr,N_sun,N_atmo,N_dsnb,ii,N_min=7500,array_length=1000,N_evt=10000,mode=0)
		tmp=recoil_nu(E_thr,ii,N_min=7500,array_length=1000,N_evt=10000,mode=0)
		#plt.plot(E_rec_out,cos_rel,col[ii])
	#E_rec_out,cos_rel=recoil_nu(E_thr,N_sun,N_atmo,N_dsnb,0,N_min=7500,array_length=1000,N_evt=1000,mode=1)
	tmp=recoil_nu(E_thr,0,N_min=7500,array_length=1000,N_evt=1000,mode=0)
	#plt.plot(E_rec_out,cos_rel,col[0])
	#plt.show()
	

#test_recoil_nu()
######
######
###create_Enu_cos(.)
#HH,EE,TT=recoil_nu(3.,N_min=100000,N_evt=1000000,mode=0)
#HHnz=np.where(HH>0,HH,1.)
#HHmin=np.min(HHnz)
#
#f=open('nu_pdf.txt','w')
#f.close()
#f=open('nu_pdf.txt','a')
#for ii in range (len(EE)):
#	for jj in range (len(TT)):
#		if HH[ii][jj]<HHmin:
#			HH[ii][jj]=HHmin
#		f.write(str(EE[ii])+' '+str(TT[jj])+' '+str(HH[ii][jj])+'\n')
#f.close()
#
######
######
