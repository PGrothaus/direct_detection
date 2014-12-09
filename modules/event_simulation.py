import constants
from constants import *
from neutrino_functions import *
from dm_functions import *

def simulate_events(nu_type, N0, N_sim, source_length, rnd_B_t ):
    array_length=max( 10, min( int( 1. * N0 * N_sim ), source_length ) )
    if channel_time==True:
        r_array  = np.random.uniform(0.,1.,array_length)
        t_src_nu =  rnd_B_t( r_array ) 
    if channel_Erec==True:
        E_rec_src_nu,cos_src_nu = recoil_nu(E_thr, nu_type ,
                                    N_evt=array_length, mode=1 )
    return t_src_nu, E_rec_src_nu, cos_src_nu
    

def get_event_array(NN_nu, array_length, t_src, E_rec_src, cos_src ):
    t_array_out, E_rec_array_out, cos_array_out = [], [], []
    while ((len(E_rec_array_out))<NN_nu):
        jj=np.random.randint(0,array_length-1,\
                               array_length)
        t_array_tmp=t_src[jj]
        E_rec_array_tmp=E_rec_src[jj]
        cos_array_tmp=cos_src[jj]
        t_array_out=np.concatenate(\
                  (t_array_out,t_array_tmp))
        E_rec_array_out=np.concatenate(\
                  (E_rec_array_out,E_rec_array_tmp))
        cos_array_out=np.concatenate(\
                  (cos_array_out,cos_array_tmp))
    t_array_out=t_array_out[:NN_nu]
    E_rec_array_out=E_rec_array_out[:NN_nu]
    cos_array_out=cos_array_out[:NN_nu]
    return t_array_out, E_rec_array_out, cos_array_out


def simulate_dm_events(m_DM_array, source_length,\
                       t0, t1, N_tt, rnd_DM_t):
    if channel_time==True:
        r_array=np.random.uniform(0.,1.,source_length)
        t_src_DM = rnd_DM_t( r_array )
    if channel_Erec==True:
        t_src_DM,E_rec_src_DM,cos_src_DM = \
            recoil_dm(E_thr,m_DM_array,t0,t1,N_tt,\
            N_evt=source_length,mode=1)

    return t_src_DM, E_rec_src_DM, cos_src_DM

        
