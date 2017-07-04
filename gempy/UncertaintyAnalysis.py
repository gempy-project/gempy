import theano


def create_model_op(interp_data):
    input_data_T = interp_data.interpolator.tg.input_parameters_list()
    input_data_P = interp_data.get_input_data()
    return theano.OpFromGraph(input_data_T, [interp_data.interpolator.tg.whole_block_model(interp_data.n_faults)],
                              on_unused_input='ignore')


# Code to select only some of the raws or a column stochastic:
#     ref = pm.Deterministic('reference', T.set_subtensor(
#         ref[T.nonzero(T.cast(select.as_matrix(), "int8"))[0], 2],
#         ref[T.nonzero(T.cast(select.as_matrix(), "int8"))[0], 2] + reservoir))
#     rest = pm.Deterministic('rest', T.set_subtensor(
#         rest[T.nonzero(T.cast(select.as_matrix(), "int8"))[0], 2],
#         rest[T.nonzero(T.cast(select.as_matrix(), "int8"))[0], 2] + reservoir))  #
