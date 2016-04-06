This repository contains the network of the PhD Thesis 
"Correlations and Coding in Visual Cortex" by Robert Meyer.


The network ``AdexpNetwork`` can be found in ``visualcortex.model.adexpmixed``.
The parameters are explained in the ``AdexpNetwork``'s documentation.
A usage example is given in ``tests/testadexpmixed.py`` as a 
unittest test case.
The example (but not the network itself) requires [BRIAN](http://briansimulator.org/).
BRIAN is only needed for its unit package (e.g. units like mV or pA).
Moreover, the example can also be found without the BRIAN requirement, no units, and as 
a simple function in ``tests/withoutunits.py``.

