import numpy as np
#from constants import erf
from math import erf

aa=np.arange(0.,100.,0.5)

eff=lambda x: 0.5*(1+erf((x-5)/15))

f=open('eff_cf4_5keV.dat','w')
f.close()
f=open('eff_cf4_5keV.dat','a')
for ii in range (len(aa)):
	f.write(str(aa[ii])+' '+str(eff(aa[ii]))+'\n')
f.close()

