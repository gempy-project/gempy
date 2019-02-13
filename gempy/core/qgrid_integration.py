from gempy.core.model import Model
from gempy.plot.plot import vtkPlot
import qgrid


class QgridModelIntegration(object):
    """
    Class that handles the changes done interactively in qgrid and updates a Model object.

    NOTE: We could do a smaller version of this class with only updates de data_object by extractiong the static
     methods to functions.

    """
    def __init__(self, geo_model: Model, vtk_object: vtkPlot = None):
        # TODO add on all to update from data_object and plots?

        self._geo_model = geo_model
        self.qgrid_fo = None
        self.qgrid_se = None
        self.qgrid_fa = None
        self.qgrid_fr = None
        self.qgrid_in = None
        self.qgrid_or = None
        self.qgrid_ad = None

    def set_interactive_df(self, data_type: str):
        if data_type is 'formations':
            self.qgrid_fo = create_formations_qgrid(self._geo_model.formations)
            return self.qgrid_fo
        elif data_type is 'series':
            self.qgrid_se = create_series_qgrid(self._geo_model.series)
            return self.qgrid_se
        elif data_type is 'faults':
            self.qgrid_fa = create_faults_qgrid(self._geo_model.faults)
            return self.qgrid_fa
        elif data_type is 'faults_relations':
            self.qgrid_fr = create_faults_relations_qgrid(self._geo_model.faults)
            return self.qgrid_fr
        elif data_type is 'interfaces':
            self.qgrid_in = create_interfaces_qgrid(self._geo_model)
            return self.qgrid_in
        elif data_type is 'orientation':
            self.qgrid_or = create_orientations_qgrid(self._geo_model)
            return self.qgrid_or
        elif data_type is 'options':
            self.qgrid_op = create_series_qgrid(self._geo_model.additional_data.options)
            return self.qgrid_op
        elif data_type is 'kriging':
            self.qgrid_kr = create_series_qgrid(self._geo_model.additional_data.kriging_data)
            return self.qgrid_kr
        elif data_type is 'rescale':
            self.qgrid_re = create_series_qgrid(self._geo_model.additional_data.rescaling_data)
            return self.qgrid_re

        else:
            raise AttributeError('data_type must be either formations, series, faults, faults_relations,'
                                 ' interfaces, orientations,'
                                  'options, kriging or rescale. UPDATE message')
        # return self.qgrid_widget

    def show(self, data_type: str):

        if data_type is 'formations':
            self.qgrid_fo._update_df()
            return self.qgrid_fo
        elif data_type is 'series':
            self.qgrid_se._update_df()
            return self.qgrid_se
        elif data_type is 'faults':
            self.qgrid_fa._update_df()
            return self.qgrid_fa
        elif data_type is 'faults_relations':
            self.qgrid_fr._update_df()
            return self.qgrid_fr
        elif data_type is 'interfaces':
            self.qgrid_in._update_df()
            return self.qgrid_in
        elif data_type is 'orientation':
            self.qgrid_or._update_df()
            return self.qgrid_or
        elif data_type is 'options':
            self.qgrid_op._update_df()
            return self.qgrid_op
        elif data_type is 'kriging':
            self.qgrid_kr._update_df()
            return self.qgrid_kr
        elif data_type is 'rescale':
            self.qgrid_re._update_df()
            return self.qgrid_re

        else:
            raise AttributeError('data_type must be either formations, series, faults, interfaces, orientations,'
                                 'options, kriging or rescale. UPDATE message')


def create_faults_qgrid(faults_object):
    qgrid_widget = qgrid.show_grid(faults_object.df,
                                   show_toolbar=False,
                                   column_options={'editable': True},
                                   column_definitions={'isFinite': {'editable': False}})

    def handle_set_is_fault(event, widget, debug=False):
        if debug is True:
            print(event)
            print(widget)

        if event['column'] == 'isFault':
            idx = event['index']
            #      cat_idx = qgrid_widget.df.loc[idx, 'series_names']

            faults_object.set_is_fault([idx])
            qgrid_widget._update_df()

    qgrid_widget.on('cell_edited', handle_set_is_fault)

    return qgrid_widget


def create_rescaling_data_qgrid(rescaling_object):
    qgrid_widget = qgrid.show_grid(rescaling_object.df,
                                   show_toolbar=False,
                                   grid_options={'sortable': False, 'highlightSelectedCell': True})

    def handle_row_edit(event, widget, debug=False):
        if debug is True:
            print(event)
            print(widget)
            print(qgrid_widget._df)

        if event['column'] == 'centers':
            try:
                value = event['new']
                assert value.shape[0] is 3

                rescaling_object.df.loc['values', event['column']] = value
                qgrid_widget._update_df()

            except AssertionError:
                print('centers length must be 3: XYZ')
                qgrid_widget._update_df()

    qgrid_widget.on('cell_edited', handle_row_edit)
    return qgrid_widget


def create_options_qgrid(options_object):
    qgrid_widget = qgrid.show_grid(options_object.df,
                                   show_toolbar=False,
                                   grid_options={'sortable': False, 'highlightSelectedCell': True})

    def handle_row_edit(event, widget, debug=False):
        if debug is True:
            print(event)
            print(widget)
            print(qgrid_widget._df)

        if event['column'] == 'verbosity':
            import numpy as np
            value = np.fromstring(event['new'][1:-1], sep=',')
        else:
            value = event['new']

        options_object.df.loc['values', event['column']] = value
        qgrid_widget._update_df()

    qgrid_widget.on('cell_edited', handle_row_edit)
    return qgrid_widget


def create_kriging_parameters_qgrid(kriging_parameters_object):
    qgrid_widget = qgrid.show_grid(kriging_parameters_object.df,
                                   show_toolbar=False,
                                   grid_options={'sortable': False, 'highlightSelectedCell': True},
                                   )

    def handle_row_edit(event, widget, debug=False):
        import numpy as np

        if debug is True:
            print(event)
            print(widget)
            print(qgrid_widget._df)

        if event['column'] == 'drift equations':
            value = np.fromstring(event['new'][1:-1], sep=',')
            try:
                assert value.shape[0] is kriging_parameters_object.structure.df.loc[
                    'values', 'len series interfaces'].shape[0]

                kriging_parameters_object.df.loc['values', event['column']] = value
                qgrid_widget._update_df()

            except AssertionError:
                print('u_grade length must be the same as the number of series')
                qgrid_widget._update_df()

        else:

            kriging_parameters_object.df.loc['values', event['column']] = event['new']
            qgrid_widget._update_df()

    qgrid_widget.on('cell_edited', handle_row_edit)
    return qgrid_widget


def create_faults_relations_qgrid(faults_object):
    # We need to add the qgrid special columns to categories
    faults_object.faults_relations_df.columns = faults_object.faults_relations_df.columns.add_categories(
        ['index', 'qgrid_unfiltered_index'])

    qgrid_widget = qgrid.show_grid(faults_object.faults_relations_df,
                                   show_toolbar=False)

    def handle_set_fault_relation(event, widget, debug=False):
        if debug is True:
            print(event)
            print(widget)

        faults_object.faults_relations_df = qgrid_widget.get_changed_df()
        qgrid_widget._update_df()

    qgrid_widget.on('cell_edited', handle_set_fault_relation)
    return qgrid_widget


def create_interfaces_qgrid(interfaces_object):
    if interfaces_object.df.shape[0] == 0:
        # TODO DEBUG: I am not sure that formations always has at least one entry. Check it
        interfaces_object.add_interface(0, 0, 0, interfaces_object.formations.df['formation'].iloc[0])

    qgrid_widget = qgrid.show_grid(
        interfaces_object.df,
        show_toolbar=True,
        grid_options={'sortable': False, 'highlightSelectedCell': True},
        column_options={'editable': False},
        column_definitions={'X': {'editable': True},
                            'Y': {'editable': True},
                            'Z': {'editable': True},
                            'surface': {'editable': True}})

    def handle_row_interfaces_add(event, widget, debug=False):
        if debug is True:
            print(event)
            print(widget)
            print(qgrid_widget._df)
        idx = event['index']
        xyzs = qgrid_widget._df.loc[idx, ['X', 'Y', 'Z', 'surface']]

        interfaces_object.add_interface(*xyzs)
        qgrid_widget._update_df()

    def handle_row_interfaces_delete(event, widget, debug=False):
        if debug is True:
            print(event)
            print(widget)
        idx = event['indices']

        interfaces_object.del_interface(idx)
        qgrid_widget._update_df()

    def handle_cell_interfaces_edit(event, widget, debug=False):
        if debug is True:
            print(event)
            print(widget)

        column = event['column']
        idx = event['index']
        value = event['new']

        interfaces_object.modify_interface(idx, **{column: value})

        qgrid_widget._update_df()

    qgrid_widget.on('row_removed', handle_row_interfaces_delete)
    qgrid_widget.on('row_added', handle_row_interfaces_add)
    qgrid_widget.on('cell_edited', handle_cell_interfaces_edit)
    return qgrid_widget


def create_orientations_qgrid(orientations_object):
    if orientations_object.df.shape[0] == 0:
        # TODO DEBUG: I am not sure that formations always has at least one entry. Check it
        orientations_object.add_orientations(0, 0, 0,
                                             orientations_object.formations.df['formation'].iloc[0],
                                             0, 0, 1,
                                             )

    qgrid_widget = qgrid.show_grid(
        orientations_object.df,
        show_toolbar=True,
        grid_options={'sortable': False, 'highlightSelectedCell': True},
        column_options={'editable': False},
        column_definitions={'X': {'editable': True},
                            'Y': {'editable': True},
                            'Z': {'editable': True},
                            'G_x': {'editable': True},
                            'G_y': {'editable': True},
                            'G_z': {'editable': True},
                            'azimuth': {'editable': True},
                            'dip': {'editable': True},
                            'pole': {'editable': True},
                            'surface': {'editable': True}})

    def handle_row_orientations_add(event, widget, debug=False):
        if debug is True:
            print(event)
            print(widget)
            print(qgrid_widget._df)
        idx = event['index']
        xyzs = qgrid_widget._df.loc[idx, ['X', 'Y', 'Z','surface', 'G_x', 'G_y', 'G_z']]

        orientations_object.add_orientation(*xyzs)
        qgrid_widget._update_df()

    def handle_row_orientations_delete(event, widget, debug=False):
        if debug is True:
            print(event)
            print(widget)
        idx = event['indices']

        orientations_object.del_orientation(idx)
        qgrid_widget._update_df()

    def handle_cell_orientations_edit(event, widget, debug=False):
        if debug is True:
            print(event)
            print(widget)

        column = event['column']
        idx = event['index']
        value = event['new']

        orientations_object.modify_orientation(idx, **{column: value})
        qgrid_widget._update_df()

    qgrid_widget.on('row_removed', handle_row_orientations_delete)
    qgrid_widget.on('row_added', handle_row_orientations_add)
    qgrid_widget.on('cell_edited', handle_cell_orientations_edit)
    return qgrid_widget


def create_formations_qgrid(formation_object):
    if formation_object.df.shape[0] == 0:
        # TODO DEBUG: I am not sure that formations always has at least one entry. Check it
        formation_object.set_formation_names_pro(['surface1'])

    qgrid_widget = qgrid.show_grid(formation_object.df, show_toolbar=True,
                                   column_options={'editable': True},
                                   column_definitions={'id': {'editable': False},
                                                       'basement': {'editable': False}})

    def handle_row_formation_add(event, widget, debug=False):
        if debug is True:
            print(event)
            print(widget)
        idx = event['index']
        formation_object.add_formation(['surface' + str(idx)])
        qgrid_widget._update_df()

    def handle_row_formation_delete(event, widget, debug=False):
        if debug is True:
            print(event)
            print(widget)
        idx = event['indices']
        formation_object.delete_formation(idx)
        qgrid_widget._update_df()

    def handle_cell_formation_edit(event, widget, debug=False):
        if debug is True:
            print(event)
            print(widget)
        if event['column'] == 'formation':
            formation_object.rename_formations(event['old'], event['new'])
        if event['column'] == 'series':
            idx = event['index']
            new_series = event['new']
            formation_object.map_series({new_series:formation_object.df.loc[idx, ['formation']]})
        if event['column'] == 'isBasement':
            idx = event['index']
            formation_object.set_basement(formation_object.df.loc[idx, ['formation']])
        qgrid_widget._update_df()

    qgrid_widget.on('row_removed', handle_row_formation_delete)
    qgrid_widget.on('row_added', handle_row_formation_add)
    qgrid_widget.on('cell_edited', handle_cell_formation_edit)

    return qgrid_widget


def create_series_qgrid(series_object):
    # I need to do a serious hack because qgrid does not accept categorical index yet
    qgrid_widget = qgrid.show_grid(
        series_object.df.reset_index().rename(columns={'index': 'series_names'}).astype({'series_names': str}),
        show_toolbar=True,
        column_options={'editable': True},
        column_definitions={'order_series': {'editable': False},
                            })

    def handle_row_series_add(event, widget, debug=False):
        if debug is True:
            print(event)
            print(widget)
            print(series_object.df.reset_index())

        idx = event['index']
        series_object.add_series(['series' + str(idx)])
        qgrid_widget.df = series_object.df.reset_index().rename(columns={'index': 'series_names'}).astype(
            {'series_names': str})
        qgrid_widget._update_df()

    def handle_row_series_delete(event, widget, debug=False):
        if debug is True:
            print(event)
            print(widget)
        idx = event['indices']
        cat_idx = qgrid_widget.df.loc[idx, 'series_names']

        series_object.delete_series(cat_idx)

        qgrid_widget.df = series_object.df.reset_index().rename(columns={'index': 'series_names'}).astype(
            {'series_names': str})
        qgrid_widget._update_df()

    def handle_cell_series_edit(event, widget, debug=False):
        idx = event['index']
        cat_idx = qgrid_widget.df.loc[idx, 'series_names']
        if debug is True:
            print(event)
            print(widget)
            print(cat_idx)
            print(series_object.df.index)
        if event['column'] == 'series_names':
            series_object.rename_series({event['old']: event['new']})
        if event['column'] == 'BottomRelation':
            series_object.df.loc[cat_idx, 'BottomRelation'] = event['new']

        qgrid_widget.df = series_object.df.reset_index().rename(columns={'index': 'series_names'}).astype(
            {'series_names': str})
        qgrid_widget._update_df()

    qgrid_widget.on('row_removed', handle_row_series_delete)
    qgrid_widget.on('row_added', handle_row_series_add)
    qgrid_widget.on('cell_edited', handle_cell_series_edit)

    return qgrid_widget
