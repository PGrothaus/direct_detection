import numpy as np
from math import log10
#import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from neutrino_functions_upperflux import *
from dm_functions import *
from constants import *
import ephem



def create_all_neutrino_lookuptables(Npoints=500,Nsteps=1000):
    for var in range(9):
        create_lookup_solarneutrino(var,N_points=Npoints,steps=Nsteps)
    for var in range(4):
        create_lookup_atmoneutrino(var,N_points=Npoints,steps=Nsteps)
    for var in range (3):
	create_lookup_dsnbneutrino(var,N_points=Npoints,steps=Nsteps)
    print 'DONE! All files created.'



def create_lookup_solarneutrino(typ,N_points=500,steps=1000):
   
    print 'creating lookup table solar neutrino rates typ ',str(typ),' ...'
    name='data/solarrate'+str(typ)+'_'+str(Z)+'.txt'
    f = open(name, 'w')
    f.close()
    
    E_rec_max=min(Er_max(Enu_max_solar(typ)),upper_threshold)

    diffrate=np.zeros(N_points)
    n_event=np.zeros(N_points)
    E_rec=np.logspace(-3,log10(E_rec_max+1.),N_points)
    #need 'max_recE+1.' to allow for interpolation

    energy_list,flux_list,norm=get_spectrum_solar(typ)
    Enu_max=max(energy_list)
    Enu_min0=min(energy_list)
    flux_ipl=interp1d(energy_list,flux_list,kind='linear')
    
    print 'calculate differential event rate for solar type ',str(typ),' ...'
    for var in range (N_points):
	if ((var+1)*100.0/N_points)%20==0:
		print str((var+1)*100.0/N_points)+'%'

        Enu_min=Enu_min_CNS(E_rec[var])
        energy=np.logspace(log10(Enu_min),log10(Enu_max),steps)
        if Enu_min>Enu_max:
		diffrate[var]=0.
		continue

        integral=0.
	if energy[0]<Enu_min0:
		flux0=0.
	else:
		flux0=flux_ipl(energy[0])
	if flux0<0.:
		flux0=0.
	sigma0=d_sigma_CNS(energy[0],E_rec[var])
        for ii in range (len(energy)-1):
		energy1=energy[ii+1]
                dE=energy[ii+1]-energy[ii]
		if energy1>Enu_max or energy1<Enu_min or energy1<Enu_min0:
			flux1=0.
		else:
	                flux1=1.*norm*flux_ipl(energy1)
			if flux1<0.:
				flux1=0.
                sigma1=d_sigma_CNS(energy1,E_rec[var])
                integral=integral+0.5*(flux0+flux1)*0.5*(sigma0+sigma1)*dE
		flux0=flux1
		sigma0=sigma1
        diffrate[var]=integral
 
    print 'calculate total event rate for solar type ',str(typ),' ...'
    rate_ipl=interp1d(E_rec,diffrate,kind='linear')

    data_eff=np.loadtxt(eff_name)
    E_eff=data_eff[:,0]
    eff=data_eff[:,1]
    eff_ipl=interp1d(E_eff,eff,kind='linear')

    for var in range (N_points):
        if((var+1)*100.0/N_points)%20==0:
			print str((var+1)*100.0/N_points)+'%'

	E_rec_max=min(Er_max(Enu_max_solar(typ)),upper_threshold)
        if(E_rec[var]>E_rec_max):
            n_event[var]=0.
            continue
        energy=np.logspace(log10(E_rec[var]),log10(E_rec_max),steps)
        integral=0.
	diffrate0=rate_ipl(energy[0])
        for ii in range (len(energy)-1):
#                energy1=0.5*(energy[ii]+energy[ii+1])
		energy1=energy[ii+1]
                dE=energy[ii+1]-energy[ii]
                diffrate1=rate_ipl(energy1)
		if rec_eff==True:
			if energy1<upper_threshold:
				eff1=eff_ipl(energy1)
			else:
				eff1=0.
		if rec_eff==False:
			eff1=1.
                integral=integral+0.5*(diffrate0+diffrate1)*dE*eff1
		diffrate0=diffrate1
        n_event[var]=integral

    #write everything to file
    for var in range (N_points):
        f = open(name, 'a')
        f.write(str(E_rec[var])+' '+str(diffrate[var])+' '+str(n_event[var])+'\n')
        f.close()

def create_lookup_atmoneutrino(typ,N_points=500,steps=1000):
   
    print 'creating lookup table atmo neutrino rates typ ',str(typ),' ...'
    name='data/atmorate'+str(typ)+'_'+str(Z)+'.txt'
    f = open(name, 'w')
    f.close()
    
    E_rec_max=min(Er_max(Enu_max_atmo(typ)),upper_threshold)

    diffrate=np.zeros(N_points)
    n_event=np.zeros(N_points)
    E_rec=np.logspace(-3,log10(E_rec_max+1.),N_points)
    #need 'max_recE+1.' to allow for interpolation

    energy_list,flux_list=get_spectrum_atmo(typ)
    Enu_max=max(energy_list)
    Enu_min0=min(energy_list)
    flux_ipl=interp1d(energy_list,1.1*flux_list,kind='linear')
    print 'calculate differential event rate for atmo type ',str(typ),' ...'
    for var in range (N_points):
	if ((var+1)*100.0/N_points)%20==0:
		print str((var+1)*100.0/N_points)+'%'

        Enu_min=Enu_min_CNS(E_rec[var])
        energy=np.logspace(log10(Enu_min),log10(Enu_max),steps)
        if Enu_min>Enu_max:
		diffrate[var]=0.
		continue

        integral=0.
	if energy[0]<Enu_min0:
		flux0=0.
	else:
		flux0=flux_ipl(energy[0])
	if flux0<0.:
		flux0=0.
	sigma0=d_sigma_CNS(energy[0],E_rec[var])
        for ii in range (len(energy)-1):
#                energy1=0.5*(energy[ii]+energy[ii+1])
		energy1=energy[ii+1]
                dE=energy[ii+1]-energy[ii]
		if energy1>Enu_max or energy1<Enu_min or energy1<Enu_min0:
			flux1=0.
		else:
	                flux1=1.*flux_ipl(energy1)
			if flux1<0.:
				flux1=0.
                sigma1=d_sigma_CNS(energy1,E_rec[var])
                integral=integral+0.5*(flux0+flux1)*0.5*(sigma0+sigma1)*dE
		flux0=flux1
		sigma0=sigma1
        diffrate[var]=integral
 
    print 'calculate total event rate for atmo type ',str(typ),' ...'
    rate_ipl=interp1d(E_rec,diffrate,kind='linear')
    data_eff=np.loadtxt(eff_name)
    E_eff=data_eff[:,0]
    eff=data_eff[:,1]
    eff_ipl=interp1d(E_eff,eff,kind='linear')

    for var in range (N_points):
        if((var+1)*100.0/N_points)%20==0:
			print str((var+1)*100.0/N_points)+'%'

	E_rec_max=min(Er_max(Enu_max_atmo(typ)),upper_threshold)
        if(E_rec[var]>E_rec_max):
            n_event[var]=0.
            continue
        energy=np.logspace(log10(E_rec[var]),log10(E_rec_max),steps)
        integral=0.
	diffrate0=rate_ipl(energy[0])
        for ii in range (len(energy)-1):
#                energy1=0.5*(energy[ii]+energy[ii+1])
		energy1=energy[ii+1]
                dE=energy[ii+1]-energy[ii]
                diffrate1=rate_ipl(energy1)
		if rec_eff==True:
			if energy1<upper_threshold:
				eff1=eff_ipl(energy1)
			else:
				eff1=0.
		if rec_eff==False:
			eff1=1.
                integral=integral+0.5*(diffrate0+diffrate1)*dE*eff1
		diffrate0=diffrate1
        n_event[var]=integral

    #write everything to file
    for var in range (N_points):
        f = open(name, 'a')
        f.write(str(E_rec[var])+' '+str(diffrate[var])+' '+str(n_event[var])+'\n')
        f.close()

def create_lookup_dsnbneutrino(typ,N_points=500,steps=1000):
   
    print 'creating lookup table DSNB neutrino rates typ ',str(typ),' ...'
    if typ==0:
	Temp=3
	norm=1.1*1.
    if typ==1:
	Temp=5
	norm=1.1*1.
    if typ==2:
	Temp=8
	norm=1.1*4.

    name='data/dsnbspectrum'+str(typ)+'.txt'
    print name
    name_write='data/dsnbrate'+str(typ)+'_'+str(Z)+'.txt'
    f=open(name_write, 'w')
    f.close()

    data=np.loadtxt(name)
    energy_list=data[:,0]
    flux_list=data[:,1]
    Enu_max=max(energy_list)
    Enu_min0=min(energy_list)

    E_rec_max=min(Er_max(Enu_max),upper_threshold)

    diffrate=np.zeros(N_points)
    n_event=np.zeros(N_points)
    E_rec=np.logspace(-3,log10(E_rec_max+1.),N_points)
    #need 'max_recE+1.' to allow for interpolation

    flux_ipl=interp1d(energy_list,flux_list,kind='linear')
    
    print 'calculate differential event rate for DSNB type ',str(typ),' ...'
    for var in range (N_points):
	if ((var+1)*100.0/N_points)%20==0:
		print str((var+1)*100.0/N_points)+'%'

        Enu_min=Enu_min_CNS(E_rec[var])
        energy=np.logspace(log10(Enu_min),log10(Enu_max),steps)
        if Enu_min>Enu_max:
		diffrate[var]=0.
		continue

        integral=0.
	if energy[0]<Enu_min0:
		flux0=0.
	else:
		flux0=1.*norm*flux_ipl(energy[0])
	if flux0<0.:
		flux0=0.
	sigma0=d_sigma_CNS(energy[0],E_rec[var])
        for ii in range (len(energy)-1):
#                energy1=0.5*(energy[ii]+energy[ii+1])
		energy1=energy[ii+1]
                dE=energy[ii+1]-energy[ii]
		if energy1>Enu_max or energy1<Enu_min or energy1<Enu_min0:
			flux1=0.
		else:
	                flux1=1.*norm*flux_ipl(energy1)
			if flux1<0.:
				flux1=0.
                sigma1=d_sigma_CNS(energy1,E_rec[var])
                integral=integral+0.5*(flux0+flux1)*0.5*(sigma0+sigma1)*dE
		flux0=flux1
		sigma0=sigma1
       	diffrate[var]=integral
 
    print 'calculate total event rate for DSNB type ',str(typ),' ...'
    rate_ipl=interp1d(E_rec,diffrate,kind='linear')

    data_eff=np.loadtxt(eff_name)
    E_eff=data_eff[:,0]
    eff=data_eff[:,1]
    eff_ipl=interp1d(E_eff,eff,kind='linear')

    for var in range (N_points):
        if((var+1)*100.0/N_points)%20==0:
			print str((var+1)*100.0/N_points)+'%'

	E_rec_max=min(Er_max(Enu_max),upper_threshold)
        if(E_rec[var]>E_rec_max):
            n_event[var]=0.
            continue
        energy=np.logspace(log10(E_rec[var]),log10(E_rec_max),steps)
        integral=0.
	diffrate0=rate_ipl(energy[0])
        for ii in range (len(energy)-1):
#                energy1=0.5*(energy[ii]+energy[ii+1])
		energy1=energy[ii+1]
                dE=energy[ii+1]-energy[ii]
                diffrate1=rate_ipl(energy1)
		if rec_eff==True:
			if energy1<upper_threshold:
				eff1=eff_ipl(energy1)
			else:
				eff1=0.
		if rec_eff==False:
			eff1=1.
                integral=integral+0.5*(diffrate0+diffrate1)*dE*eff1
		diffrate0=diffrate1
        n_event[var]=integral

    #write everything to file
    for var in range (N_points):
        f = open(name_write, 'a')
        f.write(str(E_rec[var])+' '+str(diffrate[var])+' '+str(n_event[var])+'\n')
        f.close()


#create_all_neutrino_lookuptables(Npoints=50,Nsteps=100)
