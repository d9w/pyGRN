from pygrn import grns
import json
import numpy as np


def test_export():
    grn = grns.ClassicGRN()
    grn.random(10, 10, 10)
    str_grn = str(grn)
    assert str_grn == json.dumps({'num_input': grn.num_input,
                                  'num_output': grn.num_output,
                                  'num_regulatory': grn.num_regulatory,
                                  'ids': grn.identifiers.tolist(),
                                  'enh': grn.enhancers.tolist(),
                                  'inh': grn.inhibitors.tolist(),
                                  'beta': grn.beta,
                                  'delta': grn.delta})


def test_import():
    grn = grns.ClassicGRN()
    grn.random(10, 10, 10)
    str_grn = str(grn)
    grn2 = grns.ClassicGRN()
    grn2.from_str(str_grn)
    assert grn.num_input == grn2.num_input
    assert grn.num_output == grn2.num_output
    assert grn.num_regulatory == grn2.num_regulatory
    assert np.all(grn.identifiers == grn2.identifiers)
    assert np.all(grn.enhancers == grn2.enhancers)
    assert np.all(grn.inhibitors == grn2.inhibitors)
    assert grn.beta == grn2.beta
    assert grn.delta == grn2.delta
