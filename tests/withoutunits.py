__author__ = 'Robert Meyer'


import numpy as np
import scipy.sparse as spsp
import matplotlib.pyplot as mpl

from visualcortex.model.adexpmixed import AdexpNetwork


def run_adlif():

    np.random.seed(42)
    nE = 400
    nI = 200
    synfactor = 500.0
    nSEE = 20 * nE
    nSIE = 20 * nE
    nSEI = 30*nI
    nSII = 10*nI
    nAE = 100
    nAI = 50
    nSIA = 100*nI
    nSEA = 100*nE

    S_EE = np.zeros((nE,nE))
    S_EI = np.zeros((nI,nE))
    S_IE = np.zeros((nE,nI))
    S_II = np.zeros((nI,nI))

    randE1 = np.random.random_integers(0, nE-1, nSEE)
    randE2 = np.random.random_integers(0, nE-1, nSEE)
    S_EE[randE2, randE1] = float(0.1*1e-9)
    randE2 = np.random.random_integers(0, nE-1, nSIE)
    randI1 =  np.random.random_integers(0, nI-1, nSIE)
    S_IE[randE2, randI1] = float(0.5*1e-9)
    randE1 = np.random.random_integers(0, nE-1, nSEI)
    randI2 =  np.random.random_integers(0, nI-1, nSEI)
    S_EI[randI2, randE1] = float(20*1e-9)
    randI1 = np.random.random_integers(0, nI-1, nSII)
    randI2 =  np.random.random_integers(0, nI-1, nSII)
    S_II[randI2, randI1] = float(10*1e-9)
    S_II = spsp.csr_matrix(S_II)


    freq=10.0
    stepsizeAE = 1.0/(freq*nAE)
    stepsizeAI = 1.9/(freq*nAI)

    duration = float(500*1e-3)
    linspikesAE = np.arange(0, duration, stepsizeAE)
    linspikesAI = np.arange(0, duration, stepsizeAI)
    neuronsAE=np.random.randint(0,nAE,len(linspikesAE))
    neuronsAI=np.random.randint(0,nAI,len(linspikesAI))
    spikes_AE = np.array(zip(neuronsAE, linspikesAE ))
    spikes_AI = np.array(zip(neuronsAI, linspikesAI ))
    I_EA = np.ones(nE)*0.0
    I_IA = np.ones(nI)*0.0
    S_EA = np.zeros((nAE,nE))
    S_IA = np.zeros((nAI,nI))
    randE1 = np.random.random_integers(0, nE-1, nSEA)
    randE2 = np.random.random_integers(0, nAE-1, nSEA)
    S_EA[randE2, randE1] = float(0.15*1e-9)
    randE2 = np.random.random_integers(0, nAI-1, nSIA)
    randI1 =  np.random.random_integers(0, nI-1, nSIA)
    S_IA[randE2, randI1] = float(0.3*1e-9)


    parameters = {}
    parameters['dt'] = float(0.1*1e-3)
    parameters['sample_dt'] = float(0.8*1e-3)
    parameters['current_time'] = float(100*1e-3)
    parameters['exp_steps'] = 1000000
    parameters['precomputed_exp'] = None
    parameters['min_value_exp'] = None
    parameters['max_value_exp'] = None

    parameters['duration'] = duration
    parameters['max_delay'] = None

    parameters[('exc_neurons', 'sample')] = np.array([0,1,2,3,4,5])
    parameters[('inh_neurons', 'sample')] = np.array([0,1,2,3,4,5])

    parameters[('exc_neurons', 'sample_spikes')] = None
    parameters[('inh_neurons', 'sample_spikes')] = np.array(range(int(nI/2.0)))

    parameters[('exc_neurons','V')] = np.ones(nE)*-65*1e-3
    parameters[('inh_neurons','V')] = np.ones(nI)*-65*1e-3
    parameters[('exc_neurons','w')] = np.zeros(nE)
    parameters[('inh_neurons','w')] = np.zeros(nI)

    parameters[('exc_neurons','refractory')] = None
    parameters[('inh_neurons','refractory')] = None
    parameters[('exc_neurons','tau_refr')] = float(2*1e-3)
    parameters[('inh_neurons','tau_refr')] = float(2*1e-3)
    # parameters[('exc_neurons','D')] = float(0.0*1e-3)
    # parameters[('inh_neurons','D')] = 0.0
    parameters[('exc_neurons','a')] = 0.0
    parameters[('inh_neurons','a')] = float(0.0*1e-9)
    parameters[('exc_neurons','b')] = float(50.0*1e-12)
    parameters[('inh_neurons','b')] = float(5.0*1e-12)
    # parameters[('exc_neurons', 'I_Aff_bias')] = float(-0.1*1e-12)
    parameters[('inh_neurons', 'I_Aff_bias')] = 0.0
    parameters[('exc_neurons','C')] = float(200*1e-12)
    parameters[('inh_neurons','C')] = float(200*1e-12)
    parameters[('exc_neurons','gL')] = float(10*1e-9)
    parameters[('inh_neurons','gL')] = float(20*1e-9)
    parameters[('exc_neurons','EL')] = float(-65*1e-3)
    parameters[('inh_neurons','EL')] = -65*1e-3
    parameters[('exc_neurons','VT')] = float(-50*1e-3)
    parameters[('inh_neurons','VT')] = float(-50*1e-3)
    parameters[('exc_neurons','DeltaT')] = float(2*1e-3)
    parameters[('inh_neurons','DeltaT')] = float(2*1e-3)
    parameters[('exc_neurons','Vth')] = float(-40*1e-3)
    parameters[('inh_neurons','Vth')] = float(-40*1e-3)
    parameters[('exc_neurons','Vr')] = float(-65*1e-3)
    parameters[('inh_neurons','Vr')] = float(-65*1e-3)
    parameters[('exc_neurons','tau_w')] = float(100*1e-3)
    parameters[('inh_neurons','tau_w')] = float(100*1e-3)

    parameters[('exc_neurons','maxrate')] = None
    parameters[('inh_neurons','maxrate')] = None

    parameters[('exc2exc_synapses', 'g_AMPA')] = np.zeros(nE)
    parameters[('inh2exc_synapses', 'g')] = np.zeros(nE)
    parameters[('exc2inh_synapses', 'g_AMPA')] = np.zeros(nI)
    parameters[('inh2inh_synapses', 'g')] = np.zeros(nI)
    parameters[('aff2exc_synapses', 'g_AMPA')] = np.zeros(nE)
    parameters[('aff2inh_synapses', 'g_AMPA')] = np.zeros(nI)

    parameters[('exc2exc_synapses', 'I')] = np.zeros(nE)
    parameters[('inh2exc_synapses', 'I')] = np.zeros(nE)
    parameters[('exc2inh_synapses', 'I')] = np.zeros(nI)
    parameters[('inh2inh_synapses', 'I')] = np.zeros(nI)
    parameters[('aff2exc_synapses', 'I')] = np.zeros(nE)
    parameters[('aff2inh_synapses', 'I')] = np.zeros(nI)

    parameters[('exc2exc_synapses', 'tau_rise_NMDA')] = 5.0*1e-3
    parameters[('exc2inh_synapses', 'tau_rise_NMDA')] = 5.0*1e-3
    parameters[('aff2exc_synapses', 'tau_rise_NMDA')] = float(5.0*1e-3)
    parameters[('aff2inh_synapses', 'tau_rise_NMDA')] = float(5.0*1e-3)

    parameters[('exc2exc_synapses', 'tau_decay_AMPA')] = 3*1e-3
    parameters[('exc2exc_synapses', 'tau_decay_NMDA')] = 80*1e-3
    parameters[('inh2exc_synapses', 'tau_decay')] = float(5*1e-3)
    parameters[('exc2inh_synapses', 'tau_decay_AMPA')] = 3*1e-3
    parameters[('exc2inh_synapses', 'tau_decay_NMDA')] = 80*1e-3
    parameters[('inh2inh_synapses', 'tau_decay')] = float(5*1e-3)
    parameters[('aff2exc_synapses', 'tau_decay_AMPA')] = 3*1e-3
    parameters[('aff2exc_synapses', 'tau_decay_NMDA')] = 80*1e-3
    parameters[('aff2inh_synapses', 'tau_decay_AMPA')] = 3*1e-3
    parameters[('aff2inh_synapses', 'tau_decay_NMDA')] = 80*1e-3

    parameters[('exc2exc_synapses', 'E_syn')] = 0.0
    parameters[('inh2exc_synapses', 'E_syn')] = -80*1e-3
    parameters[('exc2inh_synapses', 'E_syn')] = 0.0
    parameters[('inh2inh_synapses', 'E_syn')] = float(-80*1e-3)
    parameters[('aff2exc_synapses', 'E_syn')] = 0.0
    parameters[('aff2inh_synapses', 'E_syn')] = 0.0

    parameters[('exc2exc_synapses', 'input')] = None
    parameters[('inh2exc_synapses', 'input')] = None
    parameters[('exc2inh_synapses', 'input')] = None
    parameters[('inh2inh_synapses', 'input')] = None

    parameters[('exc2exc_synapses', 'S')] = S_EE
    parameters[('inh2exc_synapses', 'S')] = S_EI
    parameters[('exc2inh_synapses', 'S')] = S_IE
    parameters[('inh2inh_synapses', 'S')] = S_II
    parameters[('aff2exc_synapses', 'S')] = S_EA
    parameters[('aff2inh_synapses', 'S')] = S_IA


    parameters[('aff2exc_synapses', 'spikes')] = spikes_AE
    parameters[('aff2inh_synapses', 'spikes')] = spikes_AI



    net = AdexpNetwork(parameters)

    dictionary= net.run()
    short_dict = net.short_results

    times = dictionary['sample_times']
    mV_E = dictionary[('exc_neurons','V')]
    mw_E = dictionary[('exc_neurons','w')]
    mV_I = dictionary[('inh_neurons','V')]
    mw_I = dictionary[('inh_neurons','w')]
    mI_EE = short_dict['I_EE']
    mI_EI = short_dict['I_EI']
    mI_IE = short_dict['I_IE']
    mI_II = short_dict['I_II']
    mI_IA = short_dict['I_IA']
    mI_EA = short_dict['I_EA']
    mg_EE = short_dict['g_EE']
    mg_EI = short_dict['g_EI']
    mg_IE = short_dict['g_IE']
    mg_II = short_dict['g_II']
    mg_IA = short_dict['g_IA']
    mg_EA = short_dict['g_EA']
    new_spiketimes_E = dictionary[('exc_neurons','spikes')]
    new_spiketimes_I = dictionary[('inh_neurons','spikes')]
    mpl.figure()
    mpl.subplot(1,2,1)

    print short_dict['sample_times'][-1]
    print short_dict['current_time']
    for x in range(3):
        mpl.plot(times, mV_E[x,:])
        mpl.title('V_E')

    mpl.subplot(1,2,2)
    for x in range(3):
        mpl.plot(times, mw_E[x,:])
        mpl.title('w_E')

    mpl.figure()
    mpl.subplot(1,2,1)
    for x in range(3):
        mpl.plot(times, mV_I[x,:])
        mpl.title('V_I')
    mpl.subplot(1,2,2)
    for x in range(3):
        mpl.plot(times, mw_I[x,:])
        mpl.title('w_I')

    mpl.figure()
    mpl.subplot(1,4,1)
    for x in range(3):

        mpl.plot(times, mI_EE[x,:] + mI_EI[x,:])
        mpl.title('I_tot_E')
    mpl.subplot(1,4,2)
    for x in range(3):

        mpl.plot(times, mI_EE[x,:])
        mpl.title('I_EE')
    mpl.subplot(1,4,3)
    for x in range(3):

        mpl.plot(times, mI_EI[x,:])
        mpl.title('I_EI')

    mpl.subplot(1,4,4)
    for x in range(3):

        mpl.plot(times, mI_EA[x,:])
        mpl.title('I_EA')

    mpl.figure()
    mpl.subplot(1,4,1)
    for x in range(3):
        mpl.plot(times, mI_IE[x,:] + mI_II[x,:])
        mpl.title('I_tot_I')
    mpl.subplot(1,4,2)
    for x in range(3):

        mpl.plot(times, mI_IE[x,:])
        mpl.title('I_IE')
    mpl.subplot(1,4,3)
    for x in range(3):

        mpl.plot(times, mI_II[x,:])
        mpl.title('I_II')

    mpl.subplot(1,4,4)
    for x in range(3):

        mpl.plot(times, mI_IA[x,:])
        mpl.title('I_IA')

    #### g
    mpl.figure()
    mpl.subplot(1,3,1)
    for x in range(3):

        mpl.plot(times, mg_EE[x,:])
        mpl.title('g_EE')
    mpl.subplot(1,3,2)
    for x in range(3):

        mpl.plot(times, mg_EI[x,:])
        mpl.title('g_EI')

    mpl.subplot(1,3,3)
    for x in range(3):

        mpl.plot(times, mg_EA[x,:])
        mpl.title('g_EA')

    mpl.figure()
    mpl.subplot(1,4,1)
    for x in range(3):

        mpl.plot(times, mg_IE[x,:])
        mpl.title('g_IE')
    mpl.subplot(1,3,2)
    for x in range(3):

        mpl.plot(times, mg_II[x,:])
        mpl.title('g_II')

    mpl.subplot(1,3,3)
    for x in range(3):

        mpl.plot(times, mg_IA[x,:])
        mpl.title('g_IA')


    mpl.figure()
    mpl.scatter(new_spiketimes_E[:,1], new_spiketimes_E[:,0])
    mpl.title('Spikes E')
    mpl.figure()
    mpl.scatter(new_spiketimes_I[:,1], new_spiketimes_I[:,0])
    mpl.title('Spikes I')
    mpl.figure()
    mpl.scatter(spikes_AE[:,1], spikes_AE[:,0])
    mpl.title('Spikes A')
    mpl.show()

    # print net.short_results

if __name__ == '__main__':
    run_adlif()