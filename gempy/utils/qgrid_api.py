
import warnings
try:
    import qgrid
except ImportError:
    warnings.warn('qgrid package is not installed. No interactive dataframes available.')
#
#
# # =============================================================
# # This are functions that modify the data objects independently
#
# def create_faults_qgrid(faults_object):
#     qgrid_widget = qgrid.show_grid(faults_object.df,
#                                    show_toolbar=False,
#                                    column_options={'editable': True},
#                                    column_definitions={'isFinite': {'editable': False}})
#
#     def handle_set_is_fault(event, widget, debug=False):
#         if debug is True:
#             print(event)
#             print(widget)
#
#         if event['column'] == 'isFault':
#             idx = event['index']
#             #      cat_idx = qgrid_widget.df.loc[idx, 'series_names']
#
#             faults_object.set_is_fault([idx])
#             qgrid_widget._update_df()
#
#     qgrid_widget.on('cell_edited', handle_set_is_fault)
#
#     return qgrid_widget
#
#
# def create_rescaling_data_qgrid(rescaling_object):
#     qgrid_widget = qgrid.show_grid(rescaling_object.df,
#                                    show_toolbar=False,
#                                    grid_options={'sortable': False, 'highlightSelectedCell': True})
#
#     def handle_row_edit(event, widget, debug=False):
#         if debug is True:
#             print(event)
#             print(widget)
#             print(qgrid_widget._df)
#
#         if event['column'] == 'centers':
#             try:
#                 value = event['new']
#                 assert value.shape[0] is 3
#
#                 #rescaling_object.df.loc['values', event['column']] = value
#                 rescaling_object.modify_rescaling_parameters(event['column'], value)
#                 qgrid_widget._update_df()
#
#             except AssertionError:
#                 print('centers length must be 3: XYZ')
#                 qgrid_widget._update_df()
#
#     qgrid_widget.on('cell_edited', handle_row_edit)
#     return qgrid_widget
#
#
# def create_options_qgrid(options_object):
#     qgrid_widget = qgrid.show_grid(options_object.df,
#                                    show_toolbar=False,
#                                    grid_options={'sortable': False, 'highlightSelectedCell': True})
#
#     def handle_row_edit(event, widget, debug=False):
#         if debug is True:
#             print(event)
#             print(widget)
#             print(qgrid_widget._df)
#
#         if event['column'] == 'verbosity':
#             import numpy as np
#             value = np.fromstring(event['new'][1:-1], sep=',')
#         else:
#             value = event['new']
#
#         options_object.df.loc['values', event['column']] = value
#         qgrid_widget._update_df()
#
#     qgrid_widget.on('cell_edited', handle_row_edit)
#     return qgrid_widget
#
#
# def create_kriging_parameters_qgrid(kriging_parameters_object):
#     qgrid_widget = qgrid.show_grid(kriging_parameters_object.df,
#                                    show_toolbar=False,
#                                    grid_options={'sortable': False, 'highlightSelectedCell': True},
#                                    )
#
#     def handle_row_edit(event, widget, debug=False):
#         import numpy as np
#
#         if debug is True:
#             print(event)
#             print(widget)
#             print(qgrid_widget._df)
#
#         if event['column'] == 'drift equations':
#             value = np.fromstring(event['new'][1:-1], sep=',')
#             try:
#                 assert value.shape[0] is kriging_parameters_object.structure.df.loc[
#                     'values', 'len series surface_points'].shape[0]
#
#                 kriging_parameters_object.df.loc['values', event['column']] = value
#                 qgrid_widget._update_df()
#
#             except AssertionError:
#                 print('u_grade length must be the same as the number of series')
#                 qgrid_widget._update_df()
#
#         else:
#
#             kriging_parameters_object.df.loc['values', event['column']] = event['new']
#             qgrid_widget._update_df()
#
#     qgrid_widget.on('cell_edited', handle_row_edit)
#     return qgrid_widget
#
#
# def create_faults_relations_qgrid(faults_object):
#     # We need to add the qgrid special columns to categories
#     faults_object.faults_relations_df.columns = faults_object.faults_relations_df.columns.add_categories(
#         ['index', 'qgrid_unfiltered_index'])
#
#     qgrid_widget = qgrid.show_grid(faults_object.faults_relations_df,
#                                    show_toolbar=False)
#
#     def handle_set_fault_relation(event, widget, debug=False):
#         if debug is True:
#             print(event)
#             print(widget)
#
#         faults_object.faults_relations_df = qgrid_widget.get_changed_df()
#         qgrid_widget._update_df()
#
#     qgrid_widget.on('cell_edited', handle_set_fault_relation)
#     return qgrid_widget
#
#
# def create_surface_points_qgrid(surface_points_object):
#     if surface_points_object.df.shape[0] == 0:
#         # TODO DEBUG: I am not sure that formations always has at least one entry. Check it
#         surface_points_object.add_surface_points(0, 0, 0, surface_points_object.formations.df['formation'].iloc[0])
#
#     qgrid_widget = qgrid.show_grid(
#         surface_points_object.df,
#         show_toolbar=True,
#         grid_options={'sortable': False, 'highlightSelectedCell': True},
#         column_options={'editable': False},
#         column_definitions={'X': {'editable': True},
#                             'Y': {'editable': True},
#                             'Z': {'editable': True},
#                             'surface': {'editable': True}})
#
#     def handle_row_surface_points_add(event, widget, debug=False):
#         if debug is True:
#             print(event)
#             print(widget)
#             print(qgrid_widget._df)
#         idx = event['index']
#         xyzs = qgrid_widget._df.loc[idx, ['X', 'Y', 'Z', 'surface']]
#
#         self._geo.add_surface_points(*xyzs)
#         qgrid_widget._update_df()
#
#     def handle_row_surface_points_delete(event, widget, debug=False):
#         if debug is True:
#             print(event)
#             print(widget)
#         idx = event['indices']
#
#         surface_points_object.del_surface_points(idx)
#         qgrid_widget._update_df()
#
#     def handle_cell_surface_points_edit(event, widget, debug=False):
#         if debug is True:
#             print(event)
#             print(widget)
#
#         column = event['column']
#         idx = event['index']
#         value = event['new']
#
#         surface_points_object.modify_surface_points(idx, **{column: value})
#
#         qgrid_widget._update_df()
#
#     qgrid_widget.on('row_removed', handle_row_surface_points_delete)
#     qgrid_widget.on('row_added', handle_row_surface_points_add)
#     qgrid_widget.on('cell_edited', handle_cell_surface_points_edit)
#     return qgrid_widget
#
#
# def create_orientations_qgrid(orientations_object):
#     if orientations_object.df.shape[0] == 0:
#         # TODO DEBUG: I am not sure that formations always has at least one entry. Check it
#         orientations_object.add_orientations(0, 0, 0,
#                                              orientations_object.formations.df['formation'].iloc[0],
#                                              0, 0, 1,
#                                              )
#
#     qgrid_widget = qgrid.show_grid(
#         orientations_object.df,
#         show_toolbar=True,
#         grid_options={'sortable': False, 'highlightSelectedCell': True},
#         column_options={'editable': False},
#         column_definitions={'X': {'editable': True},
#                             'Y': {'editable': True},
#                             'Z': {'editable': True},
#                             'G_x': {'editable': True},
#                             'G_y': {'editable': True},
#                             'G_z': {'editable': True},
#                             'azimuth': {'editable': True},
#                             'dip': {'editable': True},
#                             'pole': {'editable': True},
#                             'surface': {'editable': True}})
#
#     def handle_row_orientations_add(event, widget, debug=False):
#         if debug is True:
#             print(event)
#             print(widget)
#             print(qgrid_widget._df)
#         idx = event['index']
#         xyzs = qgrid_widget._df.loc[idx, ['X', 'Y', 'Z','surface', 'G_x', 'G_y', 'G_z']]
#
#         orientations_object.add_orientation(*xyzs)
#         qgrid_widget._update_df()
#
#     def handle_row_orientations_delete(event, widget, debug=False):
#         if debug is True:
#             print(event)
#             print(widget)
#         idx = event['indices']
#
#         orientations_object.del_orientation(idx)
#         qgrid_widget._update_df()
#
#     def handle_cell_orientations_edit(event, widget, debug=False):
#         if debug is True:
#             print(event)
#             print(widget)
#
#         column = event['column']
#         idx = event['index']
#         value = event['new']
#
#         orientations_object.modify_orientation(idx, **{column: value})
#         qgrid_widget._update_df()
#
#     qgrid_widget.on('row_removed', handle_row_orientations_delete)
#     qgrid_widget.on('row_added', handle_row_orientations_add)
#     qgrid_widget.on('cell_edited', handle_cell_orientations_edit)
#     return qgrid_widget
#
#
# def create_formations_qgrid(formation_object):
#     if formation_object.df.shape[0] == 0:
#         # TODO DEBUG: I am not sure that formations always has at least one entry. Check it
#         formation_object.set_surfaces_names(['surface1'])
#
#     qgrid_widget = qgrid.show_grid(formation_object.df, show_toolbar=True,
#                                    column_options={'editable': True},
#                                    column_definitions={'id': {'editable': False},
#                                                        'basement': {'editable': False}})
#
#     def handle_row_formation_add(event, widget, debug=False):
#         if debug is True:
#             print(event)
#             print(widget)
#         idx = event['index']
#         formation_object.add_surface(['surface' + str(idx)])
#         qgrid_widget._update_df()
#
#     def handle_row_formation_delete(event, widget, debug=False):
#         if debug is True:
#             print(event)
#             print(widget)
#         idx = event['indices']
#         formation_object.delete_surface(idx)
#         qgrid_widget._update_df()
#
#     def handle_cell_formation_edit(event, widget, debug=False):
#         if debug is True:
#             print(event)
#             print(widget)
#         if event['column'] == 'formation':
#             formation_object.rename_formations(event['old'], event['new'])
#         if event['column'] == 'series':
#             idx = event['index']
#             new_series = event['new']
#             formation_object.map_series({new_series:formation_object.df.loc[idx, ['formation']]})
#         if event['column'] == 'isBasement':
#             idx = event['index']
#             formation_object.set_basement(formation_object.df.loc[idx, ['formation']])
#         qgrid_widget._update_df()
#
#     qgrid_widget.on('row_removed', handle_row_formation_delete)
#     qgrid_widget.on('row_added', handle_row_formation_add)
#     qgrid_widget.on('cell_edited', handle_cell_formation_edit)
#
#     return qgrid_widget
#
#
# def create_series_qgrid(series_object):
#     # I need to do a serious hack because qgrid does not accept categorical index yet
#     qgrid_widget = qgrid.show_grid(
#         series_object.df.reset_index().rename(columns={'index': 'series_names'}).astype({'series_names': str}),
#         show_toolbar=True,
#         column_options={'editable': True},
#         column_definitions={'order_series': {'editable': False},
#                             })
#
#     def handle_row_series_add(event, widget, debug=False):
#         if debug is True:
#             print(event)
#             print(widget)
#             print(series_object.df.reset_index())
#
#         idx = event['index']
#         series_object.add_series(['series' + str(idx)])
#         qgrid_widget.df = series_object.df.reset_index().rename(columns={'index': 'series_names'}).astype(
#             {'series_names': str})
#         qgrid_widget._update_df()
#
#     def handle_row_series_delete(event, widget, debug=False):
#         if debug is True:
#             print(event)
#             print(widget)
#         idx = event['indices']
#         cat_idx = qgrid_widget.df.loc[idx, 'series_names']
#
#         series_object.delete_series(cat_idx)
#
#         qgrid_widget.df = series_object.df.reset_index().rename(columns={'index': 'series_names'}).astype(
#             {'series_names': str})
#         qgrid_widget._update_df()
#
#     def handle_cell_series_edit(event, widget, debug=False):
#         idx = event['index']
#         cat_idx = qgrid_widget.df.loc[idx, 'series_names']
#         if debug is True:
#             print(event)
#             print(widget)
#             print(cat_idx)
#             print(series_object.df.index)
#         if event['column'] == 'series_names':
#             series_object.rename_series({event['old']: event['new']})
#         if event['column'] == 'BottomRelation':
#             series_object.df.loc[cat_idx, 'BottomRelation'] = event['new']
#
#         qgrid_widget.df = series_object.df.reset_index().rename(columns={'index': 'series_names'}).astype(
#             {'series_names': str})
#         qgrid_widget._update_df()
#
#     qgrid_widget.on('row_removed', handle_row_series_delete)
#     qgrid_widget.on('row_added', handle_row_series_add)
#     qgrid_widget.on('cell_edited', handle_cell_series_edit)
#
#     return qgrid_widget
