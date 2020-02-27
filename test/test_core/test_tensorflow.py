from gempy.core.interp_methods.tf_2D import constant

def test_tf(val = 2):
    assert constant(val) == 2

