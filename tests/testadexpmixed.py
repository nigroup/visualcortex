__author__ = 'Robert Meyer'

import unittest

import numpy as np
import scipy.sparse as spsp
import matplotlib.pyplot as mpl

from visualcortex.model.adexpmixed import AdexpNetwork
from brian.stdunits import *

class ADLIFTest(unittest.TestCase):

    def test_parameter_settings(self):
        x = AdexpNetwork()
        parameters = {'S_XX': 'E_syn'}
        x.add(parameters)
        x.add(tau_rise_NMDA_XE=0.0004, tau_decay_XI=0.004)
        x.add(EL_E=444.0)
        x.add({('all_neurons', 'EL'):33})

        with self.assertRaises(KeyError):
            x.add(ffrf=42)

        with self.assertRaises(KeyError):
            x.add(fff_EE=42)


    def test_adlif(self):

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
        S_EE[randE2, randE1] = float(0.1*nS)
        randE2 = np.random.random_integers(0, nE-1, nSIE)
        randI1 =  np.random.random_integers(0, nI-1, nSIE)
        S_IE[randE2, randI1] = float(0.5*nS)
        randE1 = np.random.random_integers(0, nE-1, nSEI)
        randI2 =  np.random.random_integers(0, nI-1, nSEI)
        S_EI[randI2, randE1] = float(20*nS)
        randI1 = np.random.random_integers(0, nI-1, nSII)
        randI2 =  np.random.random_integers(0, nI-1, nSII)
        S_II[randI2, randI1] = float(10*nS)
        S_II = spsp.csr_matrix(S_II)


        freq=10.0
        stepsizeAE = 1.0/(freq*nAE)
        stepsizeAI = 1.9/(freq*nAI)

        duration = float(500*ms)
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
        S_EA[randE2, randE1] = float(0.15*nS)
        randE2 = np.random.random_integers(0, nAI-1, nSIA)
        randI1 =  np.random.random_integers(0, nI-1, nSIA)
        S_IA[randE2, randI1] = float(0.3*nS)

        self.parameters = {}
        self.parameters['dt'] = float(0.1*ms)
        self.parameters['sample_dt'] = float(0.8*ms)
        self.parameters['current_time'] = float(100*ms)
        self.parameters['exp_steps'] = 1000000
        self.parameters['precomputed_exp'] = None
        self.parameters['min_value_exp'] = None
        self.parameters['max_value_exp'] = None

        self.parameters['duration'] = duration
        self.parameters['max_delay'] = None

        self.parameters[('exc_neurons', 'sample')] = np.array([0,1,2,3,4,5])
        self.parameters[('inh_neurons', 'sample')] = np.array([0,1,2,3,4,5])

        self.parameters[('exc_neurons', 'sample_spikes')] = None
        self.parameters[('inh_neurons', 'sample_spikes')] = np.array(range(int(nI/2.0)))

        self.parameters[('exc_neurons','V')] = np.ones(nE)*-65*mV
        self.parameters[('inh_neurons','V')] = np.ones(nI)*-65*mV
        self.parameters[('exc_neurons','w')] = np.zeros(nE)
        self.parameters[('inh_neurons','w')] = np.zeros(nI)

        self.parameters[('exc_neurons','refractory')] = None
        self.parameters[('inh_neurons','refractory')] = None
        self.parameters[('exc_neurons','tau_refr')] = float(2*ms)
        self.parameters[('inh_neurons','tau_refr')] = float(2*ms)
        # self.parameters[('exc_neurons','D')] = float(0.0*mV)
        # self.parameters[('inh_neurons','D')] = 0.0
        self.parameters[('exc_neurons','a')] = 0.0
        self.parameters[('inh_neurons','a')] = float(0.0*nS)
        self.parameters[('exc_neurons','b')] = float(50.0*pA)
        self.parameters[('inh_neurons','b')] = float(5.0*pA)
        # self.parameters[('exc_neurons', 'I_Aff_bias')] = float(-0.1*pA)
        self.parameters[('inh_neurons', 'I_Aff_bias')] = 0.0
        self.parameters[('exc_neurons','C')] = float(200*pF)
        self.parameters[('inh_neurons','C')] = float(200*pF)
        self.parameters[('exc_neurons','gL')] = float(10*nS)
        self.parameters[('inh_neurons','gL')] = float(20*nS)
        self.parameters[('exc_neurons','EL')] = float(-65*mV)
        self.parameters[('inh_neurons','EL')] = -65*mV
        self.parameters[('exc_neurons','VT')] = float(-50*mV)
        self.parameters[('inh_neurons','VT')] = float(-50*mV)
        self.parameters[('exc_neurons','DeltaT')] = float(2*mV)
        self.parameters[('inh_neurons','DeltaT')] = float(2*mV)
        self.parameters[('exc_neurons','Vth')] = float(-40*mV)
        self.parameters[('inh_neurons','Vth')] = float(-40*mV)
        self.parameters[('exc_neurons','Vr')] = float(-65*mV)
        self.parameters[('inh_neurons','Vr')] = float(-65*mV)
        self.parameters[('exc_neurons','tau_w')] = float(100*ms)
        self.parameters[('inh_neurons','tau_w')] = float(100*ms)

        self.parameters[('exc_neurons','maxrate')] = None
        self.parameters[('inh_neurons','maxrate')] = None

        self.parameters[('exc2exc_synapses', 'g_AMPA')] = np.zeros(nE)
        self.parameters[('inh2exc_synapses', 'g')] = np.zeros(nE)
        self.parameters[('exc2inh_synapses', 'g_AMPA')] = np.zeros(nI)
        self.parameters[('inh2inh_synapses', 'g')] = np.zeros(nI)
        self.parameters[('aff2exc_synapses', 'g_AMPA')] = np.zeros(nE)
        self.parameters[('aff2inh_synapses', 'g_AMPA')] = np.zeros(nI)

        self.parameters[('exc2exc_synapses', 'I')] = np.zeros(nE)
        self.parameters[('inh2exc_synapses', 'I')] = np.zeros(nE)
        self.parameters[('exc2inh_synapses', 'I')] = np.zeros(nI)
        self.parameters[('inh2inh_synapses', 'I')] = np.zeros(nI)
        self.parameters[('aff2exc_synapses', 'I')] = np.zeros(nE)
        self.parameters[('aff2inh_synapses', 'I')] = np.zeros(nI)

        self.parameters[('exc2exc_synapses', 'tau_rise_NMDA')] = 5.0*ms
        self.parameters[('exc2inh_synapses', 'tau_rise_NMDA')] = 5.0*ms
        self.parameters[('aff2exc_synapses', 'tau_rise_NMDA')] = float(5.0*ms)
        self.parameters[('aff2inh_synapses', 'tau_rise_NMDA')] = float(5.0*ms)

        self.parameters[('exc2exc_synapses', 'tau_decay_AMPA')] = 3*ms
        self.parameters[('exc2exc_synapses', 'tau_decay_NMDA')] = 80*ms
        self.parameters[('inh2exc_synapses', 'tau_decay')] = float(5*ms)
        self.parameters[('exc2inh_synapses', 'tau_decay_AMPA')] = 3*ms
        self.parameters[('exc2inh_synapses', 'tau_decay_NMDA')] = 80*ms
        self.parameters[('inh2inh_synapses', 'tau_decay')] = float(5*ms)
        self.parameters[('aff2exc_synapses', 'tau_decay_AMPA')] = 3*ms
        self.parameters[('aff2exc_synapses', 'tau_decay_NMDA')] = 80*ms
        self.parameters[('aff2inh_synapses', 'tau_decay_AMPA')] = 3*ms
        self.parameters[('aff2inh_synapses', 'tau_decay_NMDA')] = 80*ms

        self.parameters[('exc2exc_synapses', 'E_syn')] = 0.0
        self.parameters[('inh2exc_synapses', 'E_syn')] = -80*mV
        self.parameters[('exc2inh_synapses', 'E_syn')] = 0.0
        self.parameters[('inh2inh_synapses', 'E_syn')] = float(-80*mV)
        self.parameters[('aff2exc_synapses', 'E_syn')] = 0.0
        self.parameters[('aff2inh_synapses', 'E_syn')] = 0.0

        self.parameters[('exc2exc_synapses', 'input')] = None
        self.parameters[('inh2exc_synapses', 'input')] = None
        self.parameters[('exc2inh_synapses', 'input')] = None
        self.parameters[('inh2inh_synapses', 'input')] = None

        self.parameters[('exc2exc_synapses', 'S')] = S_EE
        self.parameters[('inh2exc_synapses', 'S')] = S_EI
        self.parameters[('exc2inh_synapses', 'S')] = S_IE
        self.parameters[('inh2inh_synapses', 'S')] = S_II
        self.parameters[('aff2exc_synapses', 'S')] = S_EA
        self.parameters[('aff2inh_synapses', 'S')] = S_IA


        self.parameters[('aff2exc_synapses', 'spikes')] = spikes_AE
        self.parameters[('aff2inh_synapses', 'spikes')] = spikes_AI



        net = AdexpNetwork(self.parameters)

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
    unittest.main()
