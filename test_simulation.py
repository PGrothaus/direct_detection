import nose
import modules.statistic as statistic
import numpy as np
import scipy
from scipy import interpolate
from scipy.interpolate import interp1d, RectBivariateSpline

def test_cl():
    N_bins = 75
    steps = 5000
    accuracy = 0.005

    Q_grid_B = np.linspace( -10, 10., N_bins) 
    Q_grid_SB = np.linspace( -10., 10., N_bins) 

    Qval_B = np.random.normal( -2, 1.82, 10000 )
    Qval_SB = np.random.normal( 2, 1.82, 10000 )

    hist_SB = np.histogram(Qval_SB, bins = Q_grid_SB)[0]
    hist_B =  np.histogram(Qval_B, bins = Q_grid_B)[0]

    overlap = statistic.get_cl(hist_SB, hist_B,\
                               Q_grid_SB, Q_grid_B,\
                               N_bins, steps, accuracy)

    #want per cent level agreement
    overlap = 100 * overlap
    overlap = int( overlap )

    print overlap
    print '12 and 14 are OK'
    assert overlap == 13 

def test_valQ():
###don't change anything here!!!!!
###
    N_Q = 1
    N_tt = 6
    t_edges = np.linspace(0. ,1., N_tt)

    #number of events shouldn't matter
    #because we give each event the same coordinates
    n_nu_arr = np.array( [2,4,6] )
    n_nu_arr = np.tile(n_nu_arr, N_Q)
    NN_nu = np.sum(n_nu_arr)

    t_nu = .4 * np.ones(NN_nu)
    E_rec_nu = 1. * np.ones(NN_nu)
    cos_nu = 1. * np.ones(NN_nu)

    xVal = np.linspace( 0., 100., 5)
    yVal = np.linspace( -1., 1., 5)
    #f_array contains S_theta for each N_tt
    #assume to return E_rec
    f_array = []
    for ii in range (N_tt):
        pdfval = 1./(2.+ ii) * np.ones( (5,5) )
        f_ipl=RectBivariateSpline(xVal,yVal,\
                                 pdfval,kx=1,ky=1)
        f_array.append(f_ipl)

    #Pb_ipl is B_theta
    #assume to be flat for test
    pdfval = 0.2 * np.ones( (5,5) )
    Pb_ipl = RectBivariateSpline(xVal,yVal,\
                                 pdfval,kx=1,ky=1)

    #without angular information
    f_array_noangle = [] 
    for ii in range (N_tt):
        pdfval = 0.4 * np.ones( (5) )
        f_ipl=interp1d(xVal, pdfval, kind = 'linear' )
        f_array_noangle.append(f_ipl)
    Pb_noangle_ipl = lambda x : .1

    #create linear time probabilities
    xVal = np.linspace(0., 1., 5)
    yVal = np.linspace(1., 2., 5)
    pdf_DM_t = scipy.interpolate.interp1d(xVal, yVal, kind = 'linear')
    xVal = np.linspace(0., 1., 5)
    yVal = np.linspace(0., 1., 5)
    pdf_nu_t = scipy.interpolate.interp1d(xVal, yVal, kind = 'linear') 

    prob_nu_B_time, prob_nu_S_time,\
    prob_nu_B_erec, prob_nu_S_erec,\
    prob_nu_B_angle, prob_nu_S_angle =\
        statistic.evaluate_Q( t_edges, \
            t_nu, E_rec_nu, cos_nu,\
            f_array, Pb_ipl,\
            f_array_noangle, Pb_noangle_ipl,\
            pdf_DM_t, pdf_nu_t,\
            NN_nu, n_nu_arr, N_Q, N_tt)   
     
    assert prob_nu_B_time[0][0] == 0.4
    assert prob_nu_S_time[0][0] == 1.4
    assert prob_nu_B_erec[0][0] == 0.1
    assert prob_nu_S_erec[0][0] == 0.4
    assert prob_nu_B_angle[0][0] == 0.2
    assert prob_nu_S_angle[0][0] == 0.25
    
