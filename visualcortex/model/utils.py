__author__ = 'Robert Meyer'


import numpy as np
from scipy import sparse as spsp, sparse


def check_matrix_and_delay(matrix=None, delay_matrix=None, source_size=None, target_size=None,
                           name='connection matrix'):
    if matrix is None:
        if delay_matrix is not None:
            raise ValueError('You cannot have a delay matrix without having a weight matrix')

        print('No %s found, making empty matrix' % name)
        matrix = spsp.csr_matrix((source_size, target_size))
    else:
        if source_size is not None:
            assert source_size == matrix.shape[0]
        else:
            source_size = matrix.shape[0]
        if target_size is not None:
            assert target_size == matrix.shape[1]
        else:
            target_size = matrix.shape[1]

    if delay_matrix is None:
        print('No delays of %s found, making empty matrix' % name)
        delay_matrix = spsp.csr_matrix((source_size, target_size))

    if spsp.issparse(delay_matrix):
        data = delay_matrix.data
    else:
        data = delay_matrix
    if len(data)> 0:
        max_delay = np.max(data)
    else:
        max_delay = 0.0

    return matrix, delay_matrix, max_delay


#@jit
def convert_matrix_jit(matrix, delay_matrix, dt):

    if not spsp.issparse(matrix):
        matrix = spsp.csr_matrix(matrix)
    if not spsp.issparse(delay_matrix):
        delay_matrix = spsp.csr_matrix(delay_matrix)

    delay_matrix_double_copy = delay_matrix.tocsr(copy=True)
    delay_matrix_int = spsp.csr_matrix(delay_matrix_double_copy/dt, dtype=int)

    non_zeros = []
    non_zeros_delay=[]
    non_zeros_value=[]
    rows = range(matrix.shape[0])
    nzlength = 1
    for row in rows:
        nonzero = np.nonzero(matrix[row,:])[1]
        if len(nonzero) > 0:
            nonzero_delay = np.array(delay_matrix_int[row, nonzero].todense().flat)
            nonzero_value = np.array(matrix[row, nonzero].todense().flat)
        else:
            nonzero_delay = None
            nonzero_value = None
        non_zeros.append(nonzero)
        non_zeros_delay.append(nonzero_delay)
        non_zeros_value.append(nonzero_value)
        if len(nonzero) > nzlength:
            nzlength=len(nonzero)


    non_zero_mat = np.ones((len(rows), nzlength), dtype=np.int)*-1
    non_zero_mat_values = np.ones((len(rows), nzlength))*-1.0
    non_zero_mat_delays = np.ones((len(rows), nzlength), dtype=np.int)*-1
    for idx, non_zero in enumerate(non_zeros):
        non_zero_delay = non_zeros_delay[idx]
        non_zero_value = non_zeros_value[idx]
        if len(non_zero)>0:
            non_zero_delay[non_zero_delay==0]=1 # Minimum delay is dt
            non_zero_mat[idx, 0:len(non_zero)] = non_zero
            non_zero_mat_delays[idx, 0:len(non_zero)] = non_zero_delay
            non_zero_mat_values[idx, 0:len(non_zero)] = non_zero_value

    return non_zero_mat, non_zero_mat_values, non_zero_mat_delays, nzlength


def create_inputs(n, inputs, max_delay_ints, name):
    new_inputs = np.zeros((n, max_delay_ints), order='F')
    if inputs is not None:
        print 'Found previous inputs %s' % name
        old_size = inputs.shape[1]
        new_inputs[:,:old_size]=inputs[:,:old_size]
    return new_inputs


def roll_inputs(inputs, delay_offset, max_delay_ints):
    rolling = max_delay_ints-delay_offset
    inputs = np.roll(inputs, rolling, axis=1)
    return inputs


def reduce_afferent_spiketimes(spiketimes, current_time, duration):

    if len(spiketimes)>0:
        logical = np.logical_and(spiketimes[:,1]>current_time,
                                 spiketimes[:,1]<=current_time+duration)
        return spiketimes[logical,:]
    else:
        return np.array([[0, -999999999999999.]]) # Strange bug, memory layout 'F'
    # does not work with current set back, if I do not return at least a single spike here