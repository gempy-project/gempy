import gempy as gp


def test_edit_1(one_fault_model_no_interp):
    model = one_fault_model_no_interp
    rpn = gp.edit(model, 'add_surfaces', surface_list=['foo'])

    assert rpn.df.iloc[-1, 0] == 'foo'
    assert model._surfaces.df.iloc[-1, 0]
