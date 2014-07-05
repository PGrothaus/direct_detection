import math
from math import sqrt,pi
import numpy as np

####################
#conversion factors#
####################
GeVkg=1.78e-27#GeV into kg
keVkg=1.78e-33#keV into kg
keVJ=1.602e-16#keV into J
GeVcm=0.1975e-13#GeV^-1 into cm
kgg=1.e3#kg into g
Jcms=1.e4#converting into cm and s
yrs=365.25*24.*3600.#s
days=24.*3600#s
keVfm=1./(0.1975e6)#1keV in fm^-1
mkeV=1./(2.e-10)#1m in keV-1
tkeV=1.e36/1.78#1t in keV
rpd=2*pi/360.

###########
#constants#
###########
sW2=0.23150
N0=6.022e23#mol^-1
GF=1.16637e-5#GeV^-2
gv=2.*sW2-0.5+1.
ga=0.5+1.
me=511.e-6#GeV
c0=2.99792458e10#cm/s
AU=149597870700.#m
mP=0.938272046
mN=0.939565379
amu=0.931494061#GeV
###############
#DM properties#
###############
rho0=0.3#GeV/cm^3
v0=220.e5#cm/s
v_esc=544.e5#cm/s
vE=232.e5#cm/s
sigma_v=v0/sqrt(2.)#cm/s
sigma0=1.e-40

#isoevents=2.3# at 90% c.l. (from 1307.5458, p7)
isoevents=0.06#to avoid memory error

###################
#detector material#
###################
###

###XENON
###
#A=131.293#g mol^-1
#Z=54.
#N=77.388#this includes the different isotopes of Xenon
	###choosing N=78 and the normalisation of the solar
	###neutrino fluxes reproduces 574 neutrino events
	###for a 2t-year Xenon detector

###ARGON
###
#A=39.948
#Z=18.
#N=22.

###CF4
###	(must do for now, think about it again. DM scatters 10 times more on the F cause of A^2 and CF_4)
A=0.2*88.0043
Z=0.2*(4.*9.+1.*6.)
N=0.2*(4.*10.+1.*6.)

###Helium
###
#A=4.
#Z=2.
#N=2.

#mT=Z*mP+N*mN#in GeV
mT=A*amu
Qw=N-(1.-4.*sW2)*Z
cf=1.23*A**(1./3)-0.6#fm
sf=0.9#fm
af=0.52#fm
rn=sqrt(cf**2+7./3*pi**2*af**2-5.*sf**2)#fm


###options
###
E_thr=5.
basenamepdf='pdfs/2dpdf_CF4_'+str(E_thr)+'keV'
eff_name='eff_cf4_5keV.dat'
N_erec=31
N_theta=16
upper_threshold=100.#keV

nu_mod=True		#whether the annual modulation of neutrinos should be included
dm_mod=True		#whether annual modulation of dark matter should be included
rec_eff=True		#whether detector response in recoil energy should be included
			#(need to reproduce neutrino lookups if this is switched!)
channel_time=True	#whether the time should be included in the pdf's
channel_Erec=True	#whether recoil energy should be included in the pdf's
channel_angle=True	#whether cos theta should be included and make the pdf 2 dimensional
gain_direction=True	#to find out how much directionality helps.
resolution_angle=True
resolution_energy=True
nu_uncertain=True	#whether to include neutrino flux uncertainties

E_rec_bin=np.linspace(E_thr,upper_threshold,N_erec)
theta_bin=np.linspace(-1.,1.,N_theta)

angle_res=0.52#30 degree in radians
energy_res=0.1#this has to be confirmed by someone

###implementation of errorfunction
###
def erf(x):
    # save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)

