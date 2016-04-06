__author__ = 'robert'

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
        nE = 2
        nI = 2

        S_EE = np.zeros((nE,nE))
        S_EI = np.zeros((nI,nE))
        S_IE = np.zeros((nE,nI))
        S_II = np.zeros((nI,nI))

        # randE1 = np.random.random_integers(0, nE-1, nSEE)
        # randE2 = np.random.random_integers(0, nE-1, nSEE)
        # S_EE[randE2, randE1] = float(0*nS)
        # randE2 = np.random.random_integers(0, nE-1, nSIE)
        # randI1 =  np.random.random_integers(0, nI-1, nSIE)
        # S_IE[randE2, randI1] = float(0*nS)
        # randE1 = np.random.random_integers(0, nE-1, nSEI)
        # randI2 =  np.random.random_integers(0, nI-1, nSEI)
        # S_EI[randI2, randE1] = float(0*nS)
        # randI1 = np.random.random_integers(0, nI-1, nSII)
        # randI2 =  np.random.random_integers(0, nI-1, nSII)
        # S_II[randI2, randI1] = float(0*nS)
        # S_II = spsp.csr_matrix(S_II)


        freq=10.0

        duration = float(1.1*ms)
        spikes_AE = np.array([[1.0, 1.00105], [0.0, 1.0011]])
        spikes_AI = np.array([[1.0, 1.00106], [0.0, 1.00094]])
        S_EA = np.ones((2,2))*1.0
        S_IA = np.ones((2,2))*1.0

        self.parameters = {}
        self.parameters['dt'] = float(0.05*ms)
        self.parameters['sample_dt'] = float(0.25*ms)
        self.parameters['current_time'] = float(1000*ms)
        self.parameters['exp_steps'] = 1000000
        self.parameters['precomputed_exp'] = None
        self.parameters['min_value_exp'] = None
        self.parameters['max_value_exp'] = None

        self.parameters['duration'] = duration
        self.parameters['max_delay'] = None

        self.parameters[('exc_neurons', 'sample')] = np.array([0])
        self.parameters[('inh_neurons', 'sample')] = np.array([0])

        self.parameters[('exc_neurons', 'sample_spikes')] = None
        self.parameters[('inh_neurons', 'sample_spikes')] = None

        self.parameters[('exc_neurons','V')] = np.ones(nE)*-65*mV
        self.parameters[('inh_neurons','V')] = np.ones(nI)*-65*mV
        self.parameters[('exc_neurons','w')] = np.zeros(nE)
        self.parameters[('inh_neurons','w')] = np.zeros(nI)

        self.parameters[('exc_neurons','refractory')] = None
        self.parameters[('inh_neurons','refractory')] = None
        self.parameters[('exc_neurons','tau_refr')] = float(2*ms)
        self.parameters[('inh_neurons','tau_refr')] = float(2*ms)
        # self.parameters[('exc_neurons','D')] = float(0.0*mV)
        #self.parameters[('inh_neurons','D')] = 0.0
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

        net.run()

        input_EA = net.short_parameters['input_EA']
        input_IA = net.short_parameters['input_IA']

        self.assertTrue(input_EA[0,0]==1.0)
        self.assertTrue(input_EA[1,0]==1.0)

        self.assertTrue(input_EA[0,1]==0.0)
        self.assertTrue(input_EA[1,1]==0.0)

        self.assertTrue(input_IA[0,1]==0.0)
        self.assertTrue(input_IA[1,0]==1.0)

        old_sample_time = net.results['sample_times'][-1]
        net.run(1.0*ms)
        new_sample_time = net.results['sample_times'][0]
        sample_dt = net.parameters['sample_dt']
        self.assertTrue(new_sample_time-old_sample_time-sample_dt<0.00000001)
        # print net.short_results


if __name__ == '__main__':
    unittest.main()