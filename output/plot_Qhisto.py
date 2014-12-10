import numpy as np
import matplotlib.pyplot as plt

name='FINAL/Q_histo/pdf_angle_200.txt'
data=np.loadtxt(name)
Q_B=data[:,0]
Q_SB=data[:,1]

plt.figure(1)
#plt.subplot(211)
#
#plt.hist(Q_SB,50,normed=True,alpha=1.,color='red')
#plt.hist(Q_B,50,normed=True,alpha=0.6,color='blue')
#plt.xlim(-2000,500)
#plt.ylabel(r'$ p_{\theta_{\rm sun}} (\rm Q)$',fontsize=20)
#plt.tick_params(axis='both',labelsize=14)
#
name='FINAL/Q_histo/pdf_erec_200.txt'
data=np.loadtxt(name)
Q_B=data[:,0]
Q_SB=data[:,1]

#plt.subplot(212)

plt.hist(Q_SB,50,normed=True,alpha=1.,color='red')
plt.hist(Q_B,50,normed=True,alpha=0.6,color='blue')
plt.xlim(-300,200)
plt.xlabel('Q',fontsize=18)
plt.ylabel(r'$ p_{E_{\rm rec}} (\rm Q)$',fontsize=20)
plt.tick_params(axis='both',labelsize=14)

plt.show(all)
