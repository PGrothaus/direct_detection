import numpy as np
import math
from math import pi,log
import scipy 
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d,SmoothBivariateSpline
from scipy.stats import norm
import sys
import random as rnd
import ephem as eph
from modules.dm_functions import *
from modules.neutrino_functions import *
from modules.constants import *
from modules.create_lookupfiles import *
import modules.statistic as statistic
import modules.event_simulation as event_simulation
#If libastro is locally installed, add it to the path variable
#sys.path.append('/home/pg3/Packages/pyephem-3.7.5.3/libastro-3.7.5')
##############################################################################


###For test modus of the 2dimensional pdf's
### (test=1 tests the pdf's)
test=0
show_hist=0
print_info=0

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
### (extrapolated ten years)
nu_sigma=np.array([0.0001,0.0001,0.0001])#in fraction of N

###Choose values for simulation
###
steps=4000              #num of steps for integrating the pdf's
N_bins=75               #num of bins for distributions of test-statistic
accuracy=0.0005         #amount by how much both Q_pdf integrals 
                        #are required to be similar when calculating
                        #the overlap

factor=10               #num of loops when generating the toy models/events
###For the pdf's:
###
N_tt=10                 #num of lookup tables for DM, i.e. time bins
N_min_nu=150000          #num of created events to create 2d pdf neutrinos
N_min_DM=100000          #num of created events to create 2d pdf dark matter
                        #total number of events to generate 2d pdf=
                        #factor * N_min_(nu/DM)

###For the pseudo experiments:
###
source_length=10000      #number of events in event pool
                        #total pool size = factor * source_length
N_Q = 1                #num of times to vary the fluxes when evaluating Q
                        #important for neutrino flux uncertainties:
                        #we have to vary the expectations
N_sim = 1250             #num of pseudo experiments generated in simulation
                        #total number of pseudo experiments = 
                        #factor * N_sim * N_Q

M_det0=10.e6#g

SENSITIVITY_SCAN = 1

if(1 == SENSITIVITY_SCAN):
    #quantify the number of neutrino events
    M_det_array = np.array( [10., 50., 100., 500., 1000., 5000.,\
                             10000., 50000., 100000.] ) 
    M_det0 = 1.e6#leave this unchanged! Used for rescaling
else:
    M_det_array = np.array( [M_det0] )

t0=eph.date('2014/01/01')
t1=eph.date('2015/01/01')
T_det=(float(t1)-float(t0))
t1_f=float(t1)-float(t0)

###specify mass and cross section range
###
m_DM_array=np.array( [1000.] )
sigma=np.logspace(-40,-52, 120)
filename_dm='test.txt'
filename_dm = 'output/' + filename_dm
f=open(filename_dm,'w')
f.close()

np.random.seed()
##############################################################################

if channel_Erec==True and channel_time==False:
    print ''
    print 'including recoil energy without time information is not possible'
    print ''
    sys.exit()
if test==0:
    print ''
    print '###################################################################'
    print 'START'
    print '###################################################################'
    print ''
    print '####################################'
    print '#detector specifications:'
    print '#mass target material: A=', A,'(N',N,' Z',Z,')'
    print '#detector mass = ',M_det0/1000.,'kg'
    print '#exposure time = ', T_det,' live days'
    print '#lower threshold: ',E_thr
    print '#upper threshold: ',upper_threshold
    print '####################################'
    print ''
    print 'NEUTRINOS'
    print ''
    print 'creating lookup tables...'
    create_all_neutrino_lookuptables(Npoints=500,Nsteps=1000)
    print '         DONE!'
    print ''
    print 'calculating expected neutrino events...'

###calculate number of expected events
### (assuming flux at central value)
time_array_nu,rate_array_nu,dt_nu,N_sun_arr,N_atmo_arr,N_dsnb_arr=\
    tot_rate_nu(M_det0,E_thr,t0,t1,steps=100)

N_sun0=np.sum(N_sun_arr)
N_atmo0=np.sum(N_atmo_arr)
N_dsnb0=np.sum(N_dsnb_arr)
mu_nu_ini=N_sun0+N_atmo0+N_dsnb0
N_nu0=mu_nu_ini
N_tot=mu_nu_ini

print N_tot

N_nu_avg=int(mu_nu_ini)
rest=mu_nu_ini-N_nu_avg
if rest>0.5:
    N_nu_avg+=1

###create the pdf's for the neutrino signals
###
bin_prob_B_t=rate_array_nu/N_nu0
acc_prob_B_t=np.cumsum(bin_prob_B_t)
rnd_B_t=interp1d(acc_prob_B_t,time_array_nu,kind='linear')
rate_array_nu[0]=rate_array_nu[1]
pdf_nu_t=interp1d(time_array_nu,rate_array_nu/N_nu0/dt_nu,kind='linear')

###get the relative parts of neutrino sources
### (assuming central flux values)
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
                print '         2d neutrino pdf exist!'
                print '         File read!'
        except:
            print '         Creating the pdf...'
            pdf_nu=np.zeros((N_erec-1,N_theta-1))
            for ff in range (factor):
                N_arr=N_nu_arr
                ratio_arr=N_arr/(np.sum(N_arr))
                for ii in range (3):
                    pdf_nu_tmp=recoil_nu(E_thr,ii,\
                                    N_min=int(N_min_nu*ratio_arr[ii]),mode=0)
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
                    f.write(str(x_edge[ii])+' '+\
                            str(y_edge[jj])+' '+str(pdf_nu[ii][jj])+'\n')
            f.close()
            print '         File created!'

        ###Normalise properly
        ### (because of shifting edges and
        ###  assigning non-zeros values to bins with no events)
        name=basenamepdf+'_nu.txt'
        data=np.loadtxt(name)
        pdf_val=data[:,2]
        pdf_val=np.reshape(pdf_val,(N_erec-1,N_theta-1))
        Pb_ipl=RectBivariateSpline(x_edge,y_edge,pdf_val,kx=1,ky=1)
        norm_Pb=Pb_ipl.integral(min(x_edge),max(x_edge),\
                                min(y_edge),max(y_edge))
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

###simulate neutrino events
###
if test==0:
    print ''
    print 'simulating neutrino events...'

lin_B_t = lambda r: r * t1_f

if test==0:
    print '         DONE!'
    print ''
    print ''
    print 'DARK MATTER'

count=0

###loop over DM mass
###
s_int = 0
for mm in range (len(m_DM_array)):
    if( 1 == SENSITIVITY_SCAN):
        s_int = 0
    print ''
    print 'Going to next dark matter mass!'
    print '   ',m_DM_array[mm],'GeV'
    ###get 2d pdf for recoil energy and cos_theta_sun for DM
    ###
    H_out=np.zeros((N_erec-1,N_theta-1))
    t0_f=float(t0)
    t1_f=float(t1)-t0_f
    t0_f=0.
    t_edges=np.linspace(t0_f,t1_f,N_tt)
 
    if channel_Erec==True:
        if test==0:
            print ''
            print '2d E_rec-cos_theta pdf for different times of the year'
            try:
                name=basenamepdf+'_DM_'+str(m_DM_array[mm])+'_'+str(0)+'.txt'
                with open(name):
                    print '         Files for 2d pdf exist!'
                    print '         All files read!'
            except:
                print '         Creating the pdf...'
                for ff in range (factor):
                    H_tmp=recoil_dm(E_thr,m_DM_array[mm],\
                                    t0,t1,N_t=N_tt,N_min=N_min_DM,mode=0)
                    H_out=H_out+H_tmp
                print '         File created!'
                H_out=1.*H_out/factor       
                for hh in range (len(H_out)):
                    f=open(basenamepdf+'_DM_'+str(m_DM_array[mm])+'_'+\
                                              str(hh)+'.txt','w')
                    f.close()
                    f=open(basenamepdf+'_DM_'+str(m_DM_array[mm])+'_'+\
                                              str(hh)+'.txt','a')
 
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
                norm_Psb=f_ipl.integral(min(x_edge),max(x_edge),\
                                        min(y_edge),max(y_edge))
                f_ipl=RectBivariateSpline(x_edge,y_edge,\
                                          pdf_val/norm_Psb,kx=1,ky=1)
                f_array.append(f_ipl)
 
            ###marginalise over the angle
            ### (if only recoil information should be used)
            f_array_noangle=[]
            for tt in range (N_tt):
                name=basenamepdf+'_DM_'+str(m_DM_array[mm])+'_'+\
                                        str(tt)+'.txt'
                data=np.loadtxt(name)
                pdf_val=data[:,2]
                pdf_val=np.reshape(pdf_val,(N_erec-1,N_theta-1))
                p_margin=[]
                for ii in range(len(x_edge)):
                    p_margin_tmp=0.
                    for jj in range (len(y_edge)-1):
                        p_margin_tmp+=0.5*\
                                (pdf_val[ii][jj]+pdf_val[ii][jj+1])*dtheta
                    p_margin.append(p_margin_tmp)
                norm=0.
                for ii in range (len(x_edge)-1):
                    derec=x_edge[ii+1]-x_edge[ii]
                    norm+=0.5*(p_margin[ii]+p_margin[ii+1])*derec
                f_ipl=interp1d(x_edge,p_margin/norm,kind='linear')
                f_array_noangle.append(f_ipl)
                    
    for mass_N in M_det_array:
        if 1 == SENSITIVITY_SCAN:
            if(mass_N>5):
                N_sim  = 1250
                N_Q    = 1
                factor = 10
            if(mass_N>2500):
                N_sim  = 625
                N_Q    = 1
                factor = 20
            if(mass_N>7500):
                N_sim  = 125
                N_Q    = 1
                factor = 100
            if(mass_N>12500):
                N_sim  = 25
                N_Q    = 1
                factor = 500
            if(mass_N>55000):
                N_sim  = 5
                N_Q    = 1
                factor = 2500
            if(mass_N > 105000):
                print 'avoid memory overload!'
                sys.exit()
            print 'N_sim', N_sim
            M_det = M_det0 * mass_N / mu_nu_ini
            time_array_nu,rate_array_nu,dt_nu,N_sun_arr,N_atmo_arr,N_dsnb_arr=\
                tot_rate_nu(M_det,E_thr,t0,t1,steps=100)
            
            N_sun0=np.sum(N_sun_arr)
            N_atmo0=np.sum(N_atmo_arr)
            N_dsnb0=np.sum(N_dsnb_arr)
            mu_nu_0=N_sun0+N_atmo0+N_dsnb0
            N_nu0=mu_nu_0
            N_tot=mu_nu_0
            
            print ''
            print 'Going to next detector mass!'
            print 'expected number of neutrino events:',N_tot
            
            N_nu_avg=int(mu_nu_0)
            rest=mu_nu_0-N_nu_avg
            if rest>0.5:
                N_nu_avg+=1
                    
            sun_ratio=N_sun0/N_tot
            atmo_ratio=N_atmo0/N_tot
            dsnb_ratio=N_dsnb0/N_tot
            ratio_array=np.array([sun_ratio,atmo_ratio,dsnb_ratio])
            N_nu_arr=np.array([N_sun0,N_atmo0,N_dsnb0])
            mu_nu=N_tot

        else:
            M_det = M_det0
        ###calculate the general annual modulation for that DM mass
        ###
        print 'calculating expected dark matter events...'
        time_array_DM,rate_array_DM,N_DM0,dt_DM = \
                        tot_rate_dm(M_det,E_thr,m_DM_array[mm],t0,t1,steps=100)
        print '         DONE!   (depends on cross section)'
        if N_DM0==0:
            continue
        bin_prob_DM_t=rate_array_DM/N_DM0
        acc_prob_DM_t=np.cumsum(bin_prob_DM_t)
        rnd_DM_t=interp1d(acc_prob_DM_t,time_array_DM,kind='linear')
        rate_array_DM[0]=rate_array_DM[1]
        pdf_DM_t=interp1d(time_array_DM,rate_array_DM/N_DM0/dt_DM,kind='linear')
    
        if test==0:
            print 'loop over cross section...'
            print ''
            print "DM_mass        sigma_SI       N_dm         N_nu"
        ###loop over cross section
        ###
        scan=True
        waszero=False
        wasbig=False
        mem1=0
        mem2=0
        scanned=np.array([])
    
        while scan==True:
            jump=False
            if s_int<0:
                break
    
            if s_int in scanned:
                jump=True
            
            if jump==False:
                ss=s_int
                
                ###initialise arrays
                ### 
                Q_B=[]
                Q_SB=[]
                t_array_SB=[]
        
                ###calculate expected dark matter events, multiply with sigma
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
        
                print ("%2E   %2E   %i        %i " % (\
                        m_DM_array[mm], sigma[ss],N_DM_exp,N_nu_avg))
        
                Q_B_angle,Q_B_erec=[],[]
                Q_SB_angle,Q_SB_erec=[],[]
        
                angle_info_DM,erec_info_DM=[],[]
    
                binsQBAngle = np.array( [] )
                binsQBErec  = np.array( [] )
                binsQSBAngle = np.array( [] )
                binsQSBErec = np.array( [] )
                FIX_BIN_SIZES_B = 0
                FIX_BIN_SIZES_SB = 0
    
                for ff in range (factor):
                    print 'loop', ff+1,'of', factor
                    ###simulate neutrino events
                    ###
                    t_src_solar_nu, E_rec_src_solar_nu, cos_src_solar_nu = \
                        event_simulation.simulate_events( 0, N_sun0, N_sim,\
                        source_length, rnd_B_t)
                    t_src_atmo_nu, E_rec_src_atmo_nu, cos_src_atmo_nu = \
                        event_simulation.simulate_events(1, N_atmo0, N_sim,\
                        source_length, lin_B_t )
                    t_src_dsnb_nu, E_rec_src_dsnb_nu, cos_src_dsnb_nu = \
                        event_simulation.simulate_events(2, N_dsnb0, N_sim,\
                        source_length, lin_B_t )
                    ###simulate DM events
                    ###
                    t_src_DM, E_rec_src_DM, cos_src_DM = \
                        event_simulation.simulate_dm_events( m_DM_array[mm],\
                                        source_length, t0, t1, N_tt, rnd_DM_t)
    
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
                        print 'get neutrino events...'          
            
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
        
                    sample_prob_nu_B=np.zeros(N_sim)
                    t_array_solar,t_array_atmo,t_array_dsnb=[],[],[]
                    E_rec_array_solar,E_rec_array_atmo,E_rec_array_dsnb=[],[],[]
                    cos_array_solar,cos_array_atmo,cos_array_dsnb=[],[],[]
        
                    ###simulate neutrino events
                    ###
                    if channel_time==True:
                        prob_nu_B=np.zeros(NN_nu)
                        ###if energy information is used, do proper simulation
                        ###
                        if channel_Erec==True:
                            ###solar
                            ###
                            solar_length=max(10,min(int(1.*N_sun0*N_sim),\
                                             source_length))
                            atmo_length=max(10,min(int(1.*N_atmo0*N_sim),\
                                             source_length))
                            dsnb_length=max(10,min(int(1.*N_dsnb0*N_sim),\
                                             source_length))
                            if solar_length>1:
                                t_array_solar, E_rec_array_solar,\
                                cos_array_solar = \
                                    event_simulation.get_event_array(\
                                       NN_solar, solar_length, \
                                       t_src_solar_nu,\
                                       E_rec_src_solar_nu,\
                                       cos_src_solar_nu )
                            ###atmo
                            ###
                            if atmo_length>1:
                                t_array_atmo, E_rec_array_atmo,\
                                cos_array_atmo =\
                                    event_simulation.get_event_array(\
                                       NN_atmo, atmo_length, \
                                       t_src_atmo_nu,\
                                       E_rec_src_atmo_nu,\
                                       cos_src_atmo_nu )
                            ###dsnb
                            ###
                            if dsnb_length>1:
                                t_array_dsnb, E_rec_array_dsnb,\
                                cos_array_dsnb =\
                                    event_simulation.get_event_array(\
                                       NN_dsnb, dsnb_length, \
                                       t_src_dsnb_nu,\
                                       E_rec_src_dsnb_nu,\
                                       cos_src_dsnb_nu )
        
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
        
                        if print_info==1:
                            print 'DONE!'
                            print 'evaluate Q for B only'
                        
                        ###calculate Pb_B
                        ###
                        mu_nu=np.zeros((N_sim,N_Q))
                        for ii in range (9):
                            if N_sun_arr[ii]>0.:
                                N_nu_solar=np.random.normal(N_sun_arr[ii],\
                                        nu_sigma[0]*N_sun_arr[ii],(N_sim,N_Q))
                                N_nu_solar=np.where(N_nu_solar<0.,0.,N_nu_solar)
                                mu_nu+=N_nu_solar
                        for ii in range (4):
                            if N_atmo_arr[ii]>0.:
                                N_nu_atmo=np.random.normal(N_atmo_arr[ii],\
                                        nu_sigma[1]*N_atmo_arr[ii],(N_sim,N_Q))
                                N_nu_atmo=np.where(N_nu_atmo<0.,0.,N_nu_atmo)
                                mu_nu+=N_nu_atmo
                        for ii in range (3):
                            if N_dsnb_arr[ii]>0.:
                                N_nu_dsnb=np.random.normal(N_dsnb_arr[ii],\
                                        nu_sigma[2]*N_dsnb_arr[ii],(N_sim,N_Q))
                                N_nu_dsnb=np.where(N_nu_dsnb<0.,0.,N_nu_dsnb)
                                mu_nu+=N_nu_dsnb
                        
                        mu_nu=np.hstack(mu_nu)
                        n_nu_arr=np.tile(n_nu_arr,N_Q)
        
                    
                    if channel_time==True:
                        if channel_Erec==True:
                            prob_nu_B_time, prob_nu_S_time,\
                            prob_nu_B_erec, prob_nu_S_erec,\
                            prob_nu_B_angle, prob_nu_S_angle =\
                            statistic.evaluate_Q( 
                                t_edges, t_nu, E_rec_nu, cos_nu,\
                                f_array, Pb_ipl,\
                                f_array_noangle, Pb_noangle_ipl,\
                                pdf_DM_t, pdf_nu_t,\
                                NN_nu, n_nu_arr, N_Q, N_tt)
                   
                        QBrangeAngle = []
                        QBrangeErec = []
    
                        for ii in range (N_sim*N_Q):
                            time_info=np.log(1.+1.*mu_DM/mu_nu[ii]*\
                                        prob_nu_S_time[ii]/prob_nu_B_time[ii])
                            angle_info=np.log(1.+1.*mu_DM/mu_nu[ii]*\
                                        prob_nu_S_angle[ii]/prob_nu_B_angle[ii])
                            erec_info=np.log(1.+1.*mu_DM/mu_nu[ii]*\
                                        prob_nu_S_erec[ii]/prob_nu_B_erec[ii])
                            pre=-mu_DM+n_nu_arr[ii]*(np.log(mu_nu[ii])-\
                                        np.log(mu_DM+mu_nu[ii]))
                            
                            angle_info=pre+np.sum(time_info)+np.sum(angle_info)
                            erec_info=pre+np.sum(time_info)+np.sum(erec_info)
                            Q_B_angle = -2. * np.array( [angle_info] )
                            Q_B_erec  = -2. * np.array( [erec_info] )
                            
                            if( 0 == FIX_BIN_SIZES_B):
                                QBrangeAngle.append(Q_B_angle)
                                QBrangeErec.append(Q_B_erec)
    
                            else:
                                Q_B_angle_histo += np.histogram( Q_B_angle,\
                                        bins = binsQBAngle )[0]
                                Q_B_erec_histo  += np.histogram( Q_B_erec,\
                                        bins = binsQBErec )[0]

                        del prob_nu_B_time
                        del prob_nu_S_time
                        del prob_nu_B_angle
                        del prob_nu_S_angle
                        del prob_nu_B_erec
                        del prob_nu_S_erec
                        del Q_B_angle
                        del Q_B_erec

                        if( 0 == FIX_BIN_SIZES_B):
                            minR = min(QBrangeAngle)
                            maxR = max(QBrangeAngle)
                            width = 0.5 * math.fabs( maxR - minR )
                            minR = minR - width
                            maxR = maxR + width
                            binsQBAngle = np.linspace( minR, maxR, N_bins )

                            minR = min(QBrangeErec)
                            maxR = max(QBrangeErec)
                            width = 0.5 * math.fabs( maxR - minR )
                            minR = minR - width
                            maxR = maxR + width
                            binsQBErec = np.linspace( minR, maxR, N_bins )
                            Q_B_angle_histo = np.histogram( QBrangeAngle,\
                                bins = binsQBAngle )[0]
                            Q_B_erec_histo  = np.histogram( QBrangeErec,\
                                bins = binsQBErec )[0]
                            FIX_BIN_SIZES_B = 1
    
                        if print_info==1:
                            print 'DONE!'
        
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
                        print 'get neutrino events'
                        
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
                    
                    if channel_time==True:
                        t_array_DM,E_rec_array_DM,cos_array_DM=[],[],[]
                        while ((len(t_array_DM))<NN_DM):
                            jj=np.random.randint(0,source_length-1,source_length)
                            t_array_tmp=t_src_DM[jj]
                            E_rec_array_tmp=E_rec_src_DM[jj]
                            cos_array_tmp=cos_src_DM[jj]
                            t_array_DM=np.concatenate((t_array_DM,t_array_tmp))
                            E_rec_array_DM=np.concatenate(\
                                    (E_rec_array_DM,E_rec_array_tmp))
                            cos_array_DM=np.concatenate(\
                                    (cos_array_DM,cos_array_tmp))
                        t_array_DM=t_array_DM[:NN_DM]
                        E_rec_array_DM=E_rec_array_DM[:NN_DM]
                        cos_array_DM=cos_array_DM[:NN_DM]
                    
                    ###simulate neutrino events
                    ###
                    t_array_solar,t_array_atmo,t_array_dsnb=[],[],[]
                    E_rec_array_solar,E_rec_array_atmo,E_rec_array_dsnb=[],[],[]
                    cos_array_solar,cos_array_atmo,cos_array_dsnb=[],[],[]
        
                    if channel_time==True:
                        ###if energy information is used, do proper simulation
                        ###
                        if channel_Erec==True:
                            ###solar
                            ###
                            if solar_length>1:
                                t_array_solar, E_rec_array_solar,\
                                cos_array_solar = \
                                    event_simulation.get_event_array(\
                                       NN_solar, solar_length, \
                                       t_src_solar_nu,\
                                       E_rec_src_solar_nu,\
                                       cos_src_solar_nu )
                            ###atmo
                            ###
                            if atmo_length>1:
                                t_array_atmo, E_rec_array_atmo,\
                                cos_array_atmo =\
                                    event_simulation.get_event_array(\
                                       NN_atmo, atmo_length, \
                                       t_src_atmo_nu,\
                                       E_rec_src_atmo_nu,\
                                       cos_src_atmo_nu )
                            ###dsnb
                            ###
                            if dsnb_length>1:
                                t_array_dsnb, E_rec_array_dsnb,\
                                cos_array_dsnb =\
                                    event_simulation.get_event_array(\
                                       NN_dsnb, dsnb_length, \
                                       t_src_dsnb_nu,\
                                       E_rec_src_dsnb_nu,\
                                       cos_src_dsnb_nu )
    
        
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
        
                        if print_info==1:
                            print 'DONE!'
        
                    ###calculate Pb_SB and Psb_SB
                    ###
                    mu_nu=np.zeros((N_sim,N_Q))
                    for ii in range (9):
                        if N_sun_arr[ii]>0.:
                            N_nu_solar=np.random.normal(N_sun_arr[ii],\
                                       nu_sigma[0]*N_sun_arr[ii],(N_sim,N_Q))
                            N_nu_solar=np.where(N_nu_solar<0.,0.,N_nu_solar)
                            mu_nu+=N_nu_solar
                    for ii in range (4):
                        if N_atmo_arr[ii]>0.:
                            N_nu_atmo=np.random.normal(N_atmo_arr[ii],\
                                      nu_sigma[1]*N_atmo_arr[ii],(N_sim,N_Q))
                            N_nu_atmo=np.where(N_nu_atmo<0.,0.,N_nu_atmo)
                            mu_nu+=N_nu_atmo
                    for ii in range (3):
                        if N_dsnb_arr[ii]>0.:
                            N_nu_dsnb=np.random.normal(N_dsnb_arr[ii],\
                                      nu_sigma[2]*N_dsnb_arr[ii],(N_sim,N_Q))
                            N_nu_dsnb=np.where(N_nu_dsnb<0.,0.,N_nu_dsnb)
                            mu_nu+=N_nu_dsnb
            
                    mu_nu=np.hstack(mu_nu)
                    n_nu_arr=np.tile(n_nu_arr,N_Q)
                    n_array_DM=np.tile(n_array_DM,N_Q)
        
                    ###Dark Matter
                    ### (i.e. Signal)
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
                        print 'calculate S+B dark matter'
                    
                    if channel_time==True:
                        prob_DM_B=np.ones(NN_DM)
                        prob_DM_SB=np.ones(NN_DM)
                        if channel_Erec==True:
                            prob_DM_B_time, prob_DM_S_time,\
                            prob_DM_B_erec, prob_DM_S_erec,\
                            prob_DM_B_angle, prob_DM_S_angle =\
                            statistic.evaluate_Q( t_edges, \
                                t_array_DM, E_rec_array_DM, cos_array_DM,\
                                f_array, Pb_ipl,\
                                f_array_noangle, Pb_noangle_ipl,\
                                pdf_DM_t, pdf_nu_t,\
                                NN_DM, n_array_DM, N_Q, N_tt)
                        
                        if print_info==1:
                            print 'DONE!'
        
                    ###neutrinos
                    ### (i.e. Background)
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
                        print 'calculate S+B neutrinos'
        
                    if channel_time==True:
                        prob_nu_SB=np.ones(NN_nu)
                        if channel_Erec==True:
                            prob_nu_B_time, prob_nu_S_time,\
                            prob_nu_B_erec, prob_nu_S_erec,\
                            prob_nu_B_angle, prob_nu_S_angle =\
                            statistic.evaluate_Q( 
                                t_edges, t_nu, E_rec_nu, cos_nu,\
                                f_array, Pb_ipl,\
                                f_array_noangle, Pb_noangle_ipl,\
                                pdf_DM_t, pdf_nu_t,\
                                NN_nu, n_nu_arr, N_Q, N_tt)
            
                        if print_info==1:
                            print 'DONE!'
                            print 'put everything together'
        
                        angle_info_nu,erec_info_nu=[],[]
                        angle_info_DM,erec_info_DM=[],[]
    
                        QSBrangeAngle = []
                        QSBrangeErec = []
    
                        for ii in range (N_sim*N_Q):
                            time_info_nu=np.array([np.sum(np.log(\
                                    1.+1.*mu_DM/mu_nu[ii]*prob_nu_S_time[ii]/\
                                    prob_nu_B_time[ii]))])
    
                            angle_info_nu=np.array([np.sum(np.log(\
                                    1.+1.*mu_DM/mu_nu[ii]*prob_nu_S_angle[ii]/\
                                    prob_nu_B_angle[ii]))])
    
                            erec_info_nu=np.array([np.sum(np.log(\
                                    1.+1.*mu_DM/mu_nu[ii]*prob_nu_S_erec[ii]/\
                                    prob_nu_B_erec[ii]))])
    
                            angle_info_nu = time_info_nu + angle_info_nu
    
                            erec_info_nu  = time_info_nu + erec_info_nu
    
                            time_info_DM=np.array([np.sum(np.log(\
                                1.+1.*mu_DM/mu_nu[ii]*prob_DM_S_time[ii]/\
                                prob_DM_B_time[ii]))])
    
                            angle_info_DM=np.array([np.sum(np.log(\
                                1.+1.*mu_DM/mu_nu[ii]*prob_DM_S_angle[ii]/\
                                prob_DM_B_angle[ii]))])
    
                            erec_info_DM=np.array([np.sum(np.log(\
                                1.+1.*mu_DM/mu_nu[ii]*prob_DM_S_erec[ii]/\
                                prob_DM_B_erec[ii]))])
    
                            pre=np.array([-mu_DM+\
                                (n_nu_arr[ii]+n_array_DM[ii])*\
                                (np.log(mu_nu[ii])-np.log(mu_nu[ii]+mu_DM))])
                            
                            angle_info_DM = pre + time_info_DM + angle_info_DM
                            erec_info_DM = pre + time_info_DM + erec_info_DM
    
                            Q_SB_angle = -2. * ( angle_info_DM + angle_info_nu )
                            Q_SB_erec  = -2. * ( erec_info_DM + erec_info_nu )
    
                            if( 0 == FIX_BIN_SIZES_SB):
                                QSBrangeAngle.append(Q_SB_angle)
                                QSBrangeErec.append(Q_SB_erec)
    
                            else:
                                Q_SB_angle_histo += np.histogram( Q_SB_angle,\
                                        bins = binsQSBAngle )[0]
                                Q_SB_erec_histo  += np.histogram( Q_SB_erec,\
                                        bins = binsQSBErec )[0]
    
                        if( 0 == FIX_BIN_SIZES_SB):
                            minR = min(QSBrangeAngle)
                            maxR = max(QSBrangeAngle)
                            width = 0.5 * math.fabs( maxR - minR )
                            minR = minR - width
                            maxR = maxR + width
                            binsQSBAngle = np.linspace( minR, maxR, N_bins )

                            minR = min(QSBrangeErec)
                            maxR = max(QSBrangeErec)
                            width = 0.5 * math.fabs( maxR - minR )
                            minR = minR - width
                            maxR = maxR + width
                            binsQSBErec = np.linspace( minR, maxR, N_bins)
                            Q_SB_angle_histo = np.histogram( QSBrangeAngle,\
                                bins = binsQSBAngle )[0]
                            Q_SB_erec_histo  = np.histogram( QSBrangeErec,\
                                bins = binsQSBErec )[0]
                            FIX_BIN_SIZES_SB = 1
    
                        del prob_DM_B_time
                        del prob_DM_S_time
                        del prob_DM_B_angle
                        del prob_DM_S_angle
                        del prob_DM_B_erec
                        del prob_DM_S_erec
                        del prob_nu_B_time
                        del prob_nu_S_time
                        del prob_nu_B_angle
                        del prob_nu_S_angle
                        del prob_nu_B_erec
                        del prob_nu_S_erec
                        del Q_SB_angle
                        del Q_SB_erec
        
                        if print_info==1:
                            print 'DONE!'
                            print 'Q-statistics...'
        
                ###calculate the values of the test statistics
                ###
                cl_angle = statistic.get_cl( Q_SB_angle_histo, Q_B_angle_histo,
                                              binsQSBAngle, binsQBAngle, 
                                              N_bins, steps, accuracy)
                cl_erec= statistic.get_cl( Q_SB_erec_histo, Q_B_erec_histo,
                                           binsQSBErec, binsQBErec,
                                           N_bins, steps, accuracy )
                ###write to file
                ###
                f=open(filename_dm,'a')
                f.write(str(M_det)+ ' '+str(N_nu_exp)+' '+\
                        str(m_DM_array[mm])+' '+str(sigma[ss])+\
                        ' '+str(N_DM_exp)+' '+str(cl_angle)+\
                        ' '+str(cl_erec)+'\n')
                f.close()
            
            if cl_angle<0.00001 and mem1==0:#and cl_erec<0.00001 and mem1==0:
                waszero=True
                mem1=1
        
            if cl_angle>0.00135 and mem2==0:#cl_erec>0.1 and mem2==0:
                wasbig=True
                mem2=1
    
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
print ''
print ''
print ''
print '####################################################################'
print 'END'
print '####################################################################'
print ''
