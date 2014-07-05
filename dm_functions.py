import numpy as np
from scipy.integrate import quad,dblquad
from scipy import stats
from scipy import special
from constants import *
import math
from math import log10
from math import pi,acos,atan2,sin,cos,exp
from neutrino_functions import form2
import random
#import matplotlib.pyplot as plt
import ephem
from scipy.interpolate import interp1d
###################################
#velocity distribution and related#
###################################
def vel_earth(time):
	time_f=float(time)
	#from 1312.1355
	t0=ephem.date('2000/01/01 12:00:00')
	t0_f=float(t0)
	n=time_f-t0_f
	#print n
	T=n/36525

	b1 = 5.536+0.013*T
	b2 = -59.574+0.002*T
	b3 = -29.811+0.001*T
	
	l1 = 266.840+1.397*T
	l2 = 347.340+1.375*T
	l3 = 180.023+1.404*T

	#everything in degrees, rpd means radian per degree
	e = 0.9574 #eccentric
	L = 280.460+0.9856474*n #mean longitude
	g = 357.528+0.9856003*n #longitude of perihelion
	w = 282.932+0.0000417*n #longitude of perihelion

	uE_av=29.79#km/s
	
	uE_x=uE_av*cos(b1*rpd)*(sin(L*rpd-l1*rpd)+e*rpd*sin(2.*L*rpd-l1*rpd-w*rpd))
	uE_y=uE_av*cos(b2*rpd)*(sin(L*rpd-l2*rpd)+e*rpd*sin(2.*L*rpd-l2*rpd-w*rpd))
	uE_z=uE_av*cos(b3*rpd)*(sin(L*rpd-l3*rpd)+e*rpd*sin(2.*L*rpd-l3*rpd-w*rpd))
	u_E2=uE_x**2+uE_y**2+uE_z**2

	#all in km/s
	v_LSR_x=0.
	v_LSR_y=220.
	v_LSR_z=0.

	v_pec_x=11.1
	v_pec_y=12.2
	v_pec_z=7.3
	
	vE_x=uE_x+v_LSR_x+v_pec_x
	vE_y=uE_y+v_LSR_y+v_pec_y
	vE_z=uE_z+v_LSR_z+v_pec_z
	
	v_E2=(vE_x)**2+(vE_y)**2+(vE_z)**2

	return [vE_x*1.e5,vE_y*1.e5,vE_z*1.e5],sqrt(v_E2)*1.e5#cm/s

def vel_earth_array(time_f):
	#from 1312.1355
	t0=ephem.date('2000/01/01 12:00:00')
	t0_f=float(t0)
	n=time_f-t0_f
	#print n
	T=n/36525

	b1 = 5.536+0.013*T
	b2 = -59.574+0.002*T
	b3 = -29.811+0.001*T
	
	l1 = 266.840+1.397*T
	l2 = 347.340+1.375*T
	l3 = 180.023+1.404*T

	#everything in degrees, rpd means radian per degree
	e = 0.9574 #eccentric
	L = 280.460+0.9856474*n #mean longitude
	g = 357.528+0.9856003*n #longitude of perihelion
	w = 282.932+0.0000417*n #longitude of perihelion

	uE_av=29.79#km/s
	
	uE_x=uE_av*np.cos(b1*rpd)*(np.sin(L*rpd-l1*rpd)+e*rpd*np.sin(2.*L*rpd-l1*rpd-w*rpd))
	uE_y=uE_av*np.cos(b2*rpd)*(np.sin(L*rpd-l2*rpd)+e*rpd*np.sin(2.*L*rpd-l2*rpd-w*rpd))
	uE_z=uE_av*np.cos(b3*rpd)*(np.sin(L*rpd-l3*rpd)+e*rpd*np.sin(2.*L*rpd-l3*rpd-w*rpd))
	u_E2=uE_x**2+uE_y**2+uE_z**2

	#all in km/s
	v_LSR_x=0.
	v_LSR_y=220.
	v_LSR_z=0.

	v_pec_x=11.1
	v_pec_y=12.2
	v_pec_z=7.3
	
	vE_x=uE_x+v_LSR_x+v_pec_x
	vE_y=uE_y+v_LSR_y+v_pec_y
	vE_z=uE_z+v_LSR_z+v_pec_z
	
	v_E2=(vE_x)**2+(vE_y)**2+(vE_z)**2

	return [vE_x*1.e5,vE_y*1.e5,vE_z*1.e5],np.sqrt(v_E2)*1.e5#cm/s


def vel_dis(v,theta,time):
	info=vel_earth(time)#in cm/s
	v_E=info[1]

	z=v_esc/v0
	sigma_v=sqrt(3./2.)*v0
	N_esc=erf(z)-2.*z*exp(-z**2)/sqrt(pi)

	f_v=1./N_esc*(3./(2.*pi*sigma_v**2))**(3./2)\
			*exp(-3.*(v**2+2.*v*v_E*cos(theta)+v_E**2)/(2*sigma_v**2))

	return f_v

def vel_int(m_DM,E_thr,time):
	v_low=v_min(m_DM,E_thr)
        info=vel_earth(time)#in cm/s
	v_obs=info[1]

	x=v_low/v0
	y=v_obs/v0
        z=v_esc/v0
        sigma_v=sqrt(3./2.)*v0
        N_esc=erf(z)-2.*z*exp(-z**2)/sqrt(pi)
#	print 'x',x
#	print 'y-z',abs(y-z)
#	print 'y',y,'z',z
	if z<y and x<abs(y-z):
		integral=1./v0/y
	if z>y and x<abs(y-z):
		integral=1./(2.*N_esc*v0*y)\
			*(erf(x+y)-erf(x-y)-4.*y/sqrt(pi)*exp(-z**2))
	if abs(y-z)<x and x<(y+z):
		integral=1./(2.*N_esc*v0*y)\
			*(erf(z)-erf(x-y)-2./sqrt(pi)*(y+x-z)*exp(-z**2))
	if x>y+z:
		integral=0.

	if integral<0.:
		return 0.
	else:
		return integral

def v_min(m_DM,rec_E):#returns minimal velocity to get a collision of rec_E cm/s
	r=4.*m_DM*mT/(m_DM+mT)**2.
	E_min=rec_E*1.e-6/r
	return sqrt(2.*E_min/m_DM)*c0#in cm/s

########################
##for the cross-section#
########################

#def form2(E_rec):#returns form factor squared at a given recoil energy
#	q=sqrt(2.*A*1.e6*E_rec)*keVfm
#	j1=sin(q*rn)/q/q/rn/rn-cos(q*rn)/q/rn
#	ff2=(3.*j1/q/rn)**2.*exp(-1.*q**2.*s**2.)
#	return ff2

def sigma_p(m_DM,E_thr,time):#looks up integral over recoil energy of 
                      #form factor and velocity distribution for E_thr
	day,month=int(time.tuple()[2]),int(time.tuple()[1])
	name='data/dm_sigma'+str(E_thr)+'keV_'+str(Z)+'_'+str(day)+'_'+str(month)+'.txt'
	data=np.loadtxt(name)
	mass=data[:,0]
	sigma_int=data[:,1]
	sigma_ipl=interp1d(mass,sigma_int,kind='linear')
	return sigma_ipl(m_DM)#in s keV/cm

def rec_int(m_DM,E_thr,time,steps=200):
	#this is the integral over the recoil energy
#the quad routine is just not precise enough and misses possible recoils above threshold
#for the very light DM candidates around m_DM~1GeV.
#Therefore, we need to do a Riemannian sum to not miss those events.
	if resolution_energy==False or resolution_energy==True:
	        integral=0.
	        E_range=np.logspace(log10(E_thr),log10(upper_threshold),steps)
	
		data_eff=np.loadtxt(eff_name)
		E_eff=data_eff[:,0]
		eff=data_eff[:,1]
		eff_ipl=interp1d(E_eff,eff,kind='linear')
		
	        for var in range (steps-1):
	                delta_E=E_range[var+1]-E_range[var]
			energy1=E_range[var]+0.5*delta_E
			if rec_eff==True:
		                if energy1<upper_threshold:
	        	                eff1=eff_ipl(energy1)
		                else:
	        	                eff1=0.
			if rec_eff==False:
				eff1=1.
	                integral+=delta_E*vel_int(m_DM,energy1,time)*form2(energy1)*eff1

#	if resolution_energy==True:
#	        integral=0.
#	        E_evt=np.logspace(log10(E_thr/10.),log10(10.*upper_threshold),steps)
#		E_rec=np.logspace(log10(E_thr),log10(upper_threshold),steps)
#
#		sigmaE = lambda E: energy_res*(np.sqrt(E))
#		smearing = lambda E,mu: 1./(sqrt(2.*pi)*sigmaE(E))*\
#					np.exp(-(E-mu)**2/(2.*sigmaE(E))**2)
#
#		data_eff=np.loadtxt(eff_name)
#		E_eff=data_eff[:,0]
#		eff=data_eff[:,1]
#		eff_ipl=interp1d(E_eff,eff,kind='linear')
#
#		dE_evt=np.delete(E_evt,0)-np.delete(E_evt,-1)
#		E_evt=0.5*(np.delete(E_evt,0)+np.delete(E_evt,-1))
#		dE_rec=np.delete(E_rec,0)-np.delete(E_rec,-1)
#                E_rec=0.5*(np.delete(E_rec,0)+np.delete(E_rec,-1))
#		if rec_eff==True:
#			eff1=eff_ipl(E_rec)
#		else:
#			eff1=np.ones(len(E_rec))
#		integral=0.
#		for ii in range (len(E_evt)):
#			integral+=np.sum(dE_evt[ii]*dE_rec*eff1*smearing(E_rec,E_evt[ii])*form2(E_evt[ii])*vel_int(m_DM,E_evt[ii],time))
#		print integral
		
        return integral#s*keV/cm

def tot_rate_dm(M,E_thr,m_DM,t0,t1,steps=365):
        #this is the time integral over the recoil rate
	#NO SIGMA IN HERE AS EVERYTHING SCALES WITH IT!!!!
	t0_f=float(t0)
	t1_f=float(t1)
	time=np.zeros(steps)
	rate=np.zeros(steps)
	time[0]=0.

	if dm_mod==True:
		time_grid=np.linspace(t0_f,t1_f,steps)
		dT=(t1_f-t0_f)/steps
		rate[0]=0.
		for ii in range (steps-1):
			time1_f=time_grid[ii+1]
			time1=ephem.date(time1_f)
		        mu_p=m_DM*mP/(m_DM+mP)
		        rate[ii+1]=(1./(2.*m_DM*mu_p**2)*rho0*M*dT*days*A**2/0.1/GeVkg/GeVkg*keVJ*\
					rec_int(m_DM,E_thr,time1))
			time[ii+1]=(time1_f-t0_f)
		summed_rate=sum(rate)

	if dm_mod==False:
		mu_p=m_DM*mP/(m_DM+mP)
		time0=ephem.date(t0_f)
		summed_rate=(1./(2.*m_DM*mu_p**2)*rho0*M*(t1_f-t0_f)*days*A**2/0.1/GeVkg/GeVkg*keVJ*\
                                        rec_int(m_DM,E_thr,time0))
		
	return time,rate,summed_rate,dT

def test_rate_DM():
	M=500.e6
	m_DM=10.
	sigma=1.e-47
        t0=ephem.date('2014/01/01')
        t1=ephem.date('2015/01/01')
	time_arr,rate_arr,N_DM,dt=tot_rate_dm(M,E_thr,m_DM,t0,t1,steps=365)
	print N_DM*sigma
#        plt.plot(time_arr,rate_arr,'ro')
#        plt.show()

#test_rate_DM()

#tmp,rate_arr,rate,tmp=tot_rate_dm(1.e6,3.,100.,ephem.date('2014/01/01'),ephem.date('2015/01/01'))
#print rate_arr*1.E-45
#print rate*1.E-45
###mode=0 -> create pdf's
###mmode=1 -> simulate DM events
###
def recoil_dm(E_thr,m_DM,t0,t1,N_t=10,N_min=750000,array_length=1000,steps=10000,N_evt=1,mode=0):
	###Euler Rodrigues rotation
	###
	def rotation_matrix(axis,theta):
	    axis = axis/math.sqrt(np.dot(axis,axis))
	    a = math.cos(theta/2)
	    b,c,d = -axis*math.sin(theta/2)
	    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
	                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
	                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

	###get cpd for v
	###
	v_grid=np.linspace(0,v_esc,steps)
	fv_norm=erf(v_esc/v0)-2.*v_esc/sqrt(pi)/v0*exp(-(v_esc/v0)**2)
	Iv_grid=(special.erf(v_grid/v0)-2.*v_grid/sqrt(pi)/v0*np.exp(-(v_grid/v0)**2))/fv_norm
	rnd_v=interp1d(Iv_grid,v_grid,kind='linear')
	
	###specify date
	###
	t0_f=float(t0)
	t1_f=float(t1)
	t_grid=np.linspace(t0_f,t1_f,N_t)

	###minimal velocity for recoil energy
	###
	vel_min=v_min(m_DM,E_thr)
	
	data_eff=np.loadtxt(eff_name)
	E_eff=data_eff[:,0]
	eff=data_eff[:,1]
	eff_ipl=interp1d(E_eff,eff,kind='linear')
	
	###initialise arrays
	###
	v_dm_earth=np.zeros(array_length)
	E_rec_array=np.zeros(array_length)
	E_DM_array=np.zeros(array_length)
	cos_scat_angle_dm=np.zeros(array_length)

	###final arrays for output
	###
	cos_angle_out=[]
	E_rec_out=[]
	theta_out=[]
	phi_out=[]
	E_DM_out=[]
	time_out=[]
	x_sun_out=[]
	y_sun_out=[]
	z_sun_out=[]
	H_out=[]

	sun=ephem.Sun()
        obs=ephem.Observer()
        obs.pressure=0.
        obs.temp=0.
        obs.lon='0.'
        obs.lat='0.'
        obs.elevation=0.

	for tt in range (N_t):
		###draw velocity vecor of the earth in galactic coordinates
                ###
		N_loop=N_evt
		if mode==0:
	#		print '			',ephem.date(t_grid[tt])

	                t_array=t_grid[tt]*np.ones(array_length)
			t_f=float(t_array[tt])
	
			name='data/dm_lookup_'+str(tt)
			f=open(name,'w')
			f.close()
	                v_earth_array=vel_earth_array(t_array)[0]

			N_loop=N_min

		rr=1.
		while len(E_DM_out)<N_loop:
			if mode==1:
		                t_array=np.random.uniform(t0_f,t1_f,array_length)
        		        v_earth_array=vel_earth_array(t_array)[0]

			###draw random DM velocity vector in galactic coordinates
			###
			r=np.random.uniform(0.,1.,array_length)
			v_array=rnd_v(r)
			cos_theta_array=np.random.uniform(-1.,1.,array_length)
			theta_array=np.arccos(cos_theta_array)
			phi_array=np.random.uniform(0.,2*pi,array_length)

			###transform into cartesian coordinates
			###
			vel_x=v_array*np.sin(theta_array)*np.cos(phi_array)
			vel_y=v_array*np.sin(theta_array)*np.sin(phi_array)
			vel_z=v_array*np.cos(theta_array)

			###transform into earth's frame of reference
			###
			vel_dm_x=vel_x-v_earth_array[0]
			vel_dm_y=vel_y-v_earth_array[1]
			vel_dm_z=vel_z-v_earth_array[2]
			v_dm_earth=np.sqrt(vel_dm_x**2+vel_dm_y**2+vel_dm_z**2)

			###energy of DM particle in earths frame of reference
			###
			E_DM_array=0.5*m_DM*v_dm_earth**2/c0**2
		
			###get recoil energy assuming isotropic scattering in cos theta
			###
			cos_scat_angle_dm=np.random.uniform(-1.,1.,array_length)
			E_rec_array=E_DM_array*4*m_DM*mT/(m_DM+mT)**2*0.5*(1.-cos_scat_angle_dm)*1.E6

			if resolution_energy==True:
				E_rec_array=np.where(E_rec_array<E_thr-1.,E_thr-1.,np.random.normal(E_rec_array,0.1*np.sqrt(E_rec_array)))

			###energy thresholds
			###
			E_rec_array=np.where(E_rec_array<E_thr,0.0001,E_rec_array)
	                E_rec_array=np.where(E_rec_array>upper_threshold,0.0001,E_rec_array)

			###include detector efficency
			###
			if rec_eff==True:
				r2=np.random.uniform(0.,1.,array_length)
				eff_array=eff_ipl(E_rec_array)
				eff_array=np.where(eff_array<0.,0.,eff_array)
				eff_array=np.where(r2<eff_array,1.,0.)
			else:
				eff_array=np.ones(array_length)

			eff_array=np.where(eff_array<0.,0.,eff_array)
			E_rec_array=E_rec_array*eff_array
	                E_rec_array=np.where(E_rec_array<E_thr,0.,E_rec_array)
			###get angles of DM velocity vector
			###		(make sure these are in galactic coordinates)
			theta_array=np.arccos(vel_dm_z/v_dm_earth)
			phi_array=np.arctan2(vel_dm_y,vel_dm_x)

			###include energy threshold
			###
			v_dm_earth=np.where(E_rec_array<E_thr,-10,v_dm_earth)
			v_dm_earth=np.where(E_rec_array>upper_threshold,-10,v_dm_earth)
			v_dm_earth=np.where(E_DM_array<E_rec_array*1.E-6,-10,v_dm_earth)

			###shift the time array
			###	(do I want this?!?!?)
			t_array=t_array-t0_f

			###create final arrays by storing those events that are OK
			###
			c_tmp=np.where(v_dm_earth>vel_min,cos_scat_angle_dm,-10)
			Er_tmp=np.where(v_dm_earth>vel_min,E_rec_array,-10)
			t_tmp=np.where(v_dm_earth>vel_min,theta_array,-10)
			p_tmp=np.where(v_dm_earth>vel_min,phi_array,-10)
			E_tmp=np.where(v_dm_earth>vel_min,E_DM_array,-10)
			time_tmp=np.where(v_dm_earth>vel_min,t_array,-10)

			cos_angle_out=np.concatenate((cos_angle_out,c_tmp[c_tmp>-9]))
			E_rec_out=np.concatenate((E_rec_out,Er_tmp[Er_tmp>-9]))
			theta_out=np.concatenate((theta_out,t_tmp[t_tmp>-9]))
			phi_out=np.concatenate((phi_out,p_tmp[p_tmp>-9]))
			E_DM_out=np.concatenate((E_DM_out,E_tmp[E_tmp>-9]))
			time_out=np.concatenate((time_out,time_tmp[time_tmp>-9]))

#			if mode==0:
#			        if(1.*(len(E_rec_out))/N_min)>rr*0.1:
#        	        	        print str(100.*(len(E_rec_out))/N_min)+'%'
#					rr+=1
#			if mode==1:
#				if(1.*(len(E_rec_out))/N_evt)>rr*0.1:
#        	        	        print str(100.*(len(E_rec_out))/N_evt)+'%'
#					rr+=1

		N_sim=len(E_DM_out)
		
		###calculate scattering angle of nucleus by looking at
		###momentum of DM particle after collision (momentum p prime))
		###
		scat_angle_dm=np.arccos(cos_angle_out)
		p_DM_p=np.sqrt(2.*m_DM*E_DM_out-2.*m_DM*E_rec_out*1.E-6)
		p_DM_p=np.where(p_DM_p<0.,1.E-10,p_DM_p)
		p_DM_p_x=p_DM_p*np.cos(scat_angle_dm)
		p_DM_p_y=p_DM_p*np.sin(scat_angle_dm)
		
		theta_nuc=np.arctan2(-1*p_DM_p_y,np.sqrt(2.*m_DM*E_DM_out)-p_DM_p_x)
		phi_nuc=np.random.uniform(0.,2.*pi,N_sim)

		###transform into equatorial coordinates
		###
		dm_ra=np.zeros(N_sim)
		dm_dec=np.zeros(N_sim)
#		print 'calculate ra and dec'
		for ii in range(N_sim):
			dm_v=ephem.Galactic(phi_out[ii],0.5*pi-theta_out[ii])
			dm_v=ephem.Equatorial(dm_v)
			dm_ra[ii]=dm_v.ra
			dm_dec[ii]=dm_v.dec
#		print'	DONE!'

		###transform into cartesian coordinates
		###
		dm_phi=dm_ra
		dm_theta=0.5*pi-dm_dec
	
		x_dm=np.sin(dm_theta)*np.cos(dm_phi)
		y_dm=np.sin(dm_theta)*np.sin(dm_phi)
		z_dm=np.cos(dm_theta)

		###get direction of the sun
		###
		if mode==0:
			obs.date=t_f
	
			sun.compute(obs)
			ra_sun=sun.ra
			dec_sun=sun.dec
	
			phi_sun=ra_sun*np.ones(N_sim)
		        theta_sun=(0.5*pi-dec_sun)*np.ones(N_sim)

		if mode==1:
			ra_sun=np.zeros(N_sim)
			dec_sun=np.zeros(N_sim)
			for ii in range (N_sim):
	                        obs.date=time_out[ii]
	                        sun.compute(obs)
	                        ra_sun[ii]=sun.ra
	                        dec_sun[ii]=sun.dec
	
			phi_sun=ra_sun
	                theta_sun=0.5*pi-dec_sun
	
                x_sun_out=np.sin(theta_sun)*np.cos(phi_sun)
                y_sun_out=np.sin(theta_sun)*np.sin(phi_sun)
                z_sun_out=np.cos(theta_sun)
		
		###define a perpendicular axis to rotate around with theta
		###
		cos_rel=np.zeros(N_sim)
#		print 'Now, rotate...'
		for ii in range (N_sim):
			v_dm=np.array([x_dm[ii],y_dm[ii],z_dm[ii]])
			axis=np.array([1,0,-1.*x_dm[ii]/z_dm[ii]])
		
			v_recoil=np.dot(rotation_matrix(axis,theta_nuc[ii]),v_dm)

			###rotate around DM velocity vector with random phi
			###
			v_recoil=np.dot(rotation_matrix(v_dm,phi_nuc[ii]),v_recoil)	
	
			###calculate angle relative to the sun
			###
			cos_rel[ii]=\
			(v_recoil[0]*x_sun_out[ii]+v_recoil[1]*y_sun_out[ii]+v_recoil[2]*z_sun_out[ii])/\
			(np.sqrt(v_recoil[0]**2+v_recoil[1]**2+v_recoil[2]**2)\
			*np.sqrt(x_sun_out[ii]**2+y_sun_out[ii]**2+z_sun_out[ii]**2))
#		print '	DONE!'
		
	        if resolution_angle==True:
			angle_rel=np.arccos(cos_rel)
	                angle_rel=np.random.normal(angle_rel,angle_res/np.sqrt(E_rec_out))
			cos_rel=np.cos(angle_rel)

		if mode==1:
			sample=zip(time_out,E_rec_out,cos_rel,x_dm,y_dm,z_dm)
			np.random.shuffle(sample)
			sample=sample[:N_evt]
			time_out,E_rec_out,cos_rel,x_dm_out,y_dm_out,z_dm_out=np.array(zip(*sample))
			return time_out,E_rec_out,cos_rel#,x_dm_out,y_dm_out,z_dm_out,theta_nuc,phi_nuc

		###make histograms to get pdf
		###
#		plt.figure('E_rec_DM 10GeV')
#		plt.hist(E_rec_out,25,normed=True,log=True)
#		plt.figure('cos_theta_sun DM 10GeV')
#		plt.hist(cos_rel,25,normed=True,log=True)
#		plt.show(all)

		###create histogram
		###

	        H1=np.histogram2d(E_rec_out,cos_rel,bins=(E_rec_bin,theta_bin),normed=True)[0]
		H_out.append(H1)

		###write the pdf into file
		###
#		f=open(name,'a')
#		for ii in range (len(E_rec_bin)-1):
#			for jj in range (len(theta_bin)-1):
#				f.write(str(H[ii][jj])+' ')
#		f.close()

	return H_out

def get_recoil(m_DM,E_thr,time=ephem.date('2014/01/01')):
	name="data/dm_recE_"+str(m_DM)+"_"+str(E_thr)+"keV_"+str(Z)
	try:
		with open(name):
			print "File exists"
			return
			
	except IOError:
		N_points=100
	    	E_grid=np.logspace(log10(E_thr),log10(upper_threshold),N_points)
    		R_tilde=np.zeros(N_points)
	    	for ii in range (N_points):
	    		R_tilde[ii]=vel_int(m_DM,E_grid[ii],time)*form2(E_grid[ii])
	    	for jj in range (N_points-1):
	    		R_tilde[jj]=0.5*(R_tilde[jj]+R_tilde[jj+1])*(E_grid[jj+1]-E_grid[jj])
	    	R_tilde[-1]=0.
	    	prob=np.zeros(N_points-1)
	    	norm=sum(R_tilde)
	    	for ii in range (N_points-1):
	    		E_grid[ii]=0.5*(E_grid[ii]+E_grid[ii+1])
	    	E_grid[-1]=0.
	    	for kk in range (N_points-1):
	    		prob[kk]=R_tilde[kk]/norm
	    	acc_prob=np.zeros(N_points-1)
	    	for ll in range (N_points-1):
    			acc_prob[ll]=sum(prob[:ll])
	    		f=open(name,"a")
	    		f.write(str(E_grid[ll])+' '+str(acc_prob[ll])+'\n')
	    		f.close()

#time0=ephem.date('2014/01/01')
#time1=ephem.date('2015/01/01')
##recoil_dm(3.,10.,time0,time1,N_t=1,N_min=1000000,N_evt=1000000,mode=0)
#
#HH,EE,TT,tmp=recoil_dm(3.,100.,time0,time1,N_t=1,N_min=1000000,N_evt=1000000,mode=0)
#HHnz=np.where(HH[0]>0,HH[0],1.)
#HHmin=np.min(HHnz)
#
#f=open('dm_pdf.txt','w')
#f.close()
#f=open('dm_pdf.txt','a')
#for ii in range (len(EE)):
#        for jj in range (len(TT)):
#                if HH[0][ii][jj]<HHmin:
#                        HH[0][ii][jj]=HHmin
#                f.write(str(EE[ii])+' '+str(TT[jj])+' '+str(HH[0][ii][jj])+'\n')
#f.close()
#
