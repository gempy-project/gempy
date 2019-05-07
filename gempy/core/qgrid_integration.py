from gempy.core.model import Model
import qgrid


class QgridModelIntegration(object):
    """
    Class that handles the changes done interactively in qgrid and updates a Model object.

    NOTE: We could do a smaller version of this class with only updates de data_object by extractiong the static
     methods to functions.

    """
    def __init__(self, geo_model: Model, plot_object=None):
        # TODO add on all to update from data_object and plots?

        self._geo_model = geo_model
        self._plot_object = plot_object
        if plot_object is not None:
            # Check if the window is already open
            self.set_vtk_object(plot_object)

            # if hasattr(plot_object, 'interactor'):
            #     self._plot_object.set_surface_points()
            #     self._plot_object.set_orientations()

        self.qgrid_fo = self.set_interactive_df('surfaces')
        self.qgrid_se = self.set_interactive_df('series')
        self.qgrid_fa = self.set_interactive_df('faults')
        self.qgrid_fr = self.set_interactive_df('faults_relations')
        self.qgrid_in = self.set_interactive_df('surface_points')
        self.qgrid_or = self.set_interactive_df('orientations')
        self.qgrid_op = self.set_interactive_df('options')
        self.qgrid_kr = self.set_interactive_df('kriging')
        self.qgrid_re = self.set_interactive_df('rescale')

    def update_plot(self):
        if self._plot_object is not None:
            self._plot_object.update_model()
        else:
            pass

    def set_interactive_df(self, data_type: str):
        if data_type == 'surfaces':
            self.qgrid_fo = self.create_surfaces_qgrid()
            return self.qgrid_fo
        elif data_type == 'series':
            self.qgrid_se = self.create_series_qgrid()
            return self.qgrid_se
        elif data_type == 'faults':
            self.qgrid_fa = self.create_faults_qgrid()
            return self.qgrid_fa
        elif data_type == 'faults_relations':
            self.qgrid_fr = self.create_faults_relations_qgrid()
            return self.qgrid_fr
        elif data_type == 'surface_points':
            self.qgrid_in = self.create_surface_points_qgrid()
            return self.qgrid_in
        elif data_type == 'orientations':
            self.qgrid_or = self.create_orientations_qgrid()
            return self.qgrid_or
        elif data_type == 'options':
            self.qgrid_op = self.create_options_qgrid()
            return self.qgrid_op
        elif data_type == 'kriging':
            self.qgrid_kr = self.create_kriging_parameters_qgrid()
            return self.qgrid_kr
        elif data_type == 'rescale':
            self.qgrid_re = self.create_rescaling_data_qgrid()
            return self.qgrid_re

        else:
            raise AttributeError('data_type must be either surfaces, series, faults, faults_relations,'
                                 ' surface_points, orientations,'
                                  'options, kriging or rescale. UPDATE message')
        # return self.qgrid_widget

    def set_vtk_object(self, vtk_object):
        self._plot_object = vtk_object
        if not hasattr(vtk_object, 'interactor'):
            self._plot_object.set_surface_points()
            self._plot_object.set_orientations()

    def update_qgrd_objects(self):
        qgrid_objects = [self.qgrid_fo, self.qgrid_se, self.qgrid_fa, #self.qgrid_fr,
                         self.qgrid_in,
                         self.qgrid_or, self.qgrid_op, self.qgrid_kr, self.qgrid_re]

        for e, qgrid_object in enumerate(qgrid_objects):
           # print('qgrid_object' + str(e) )
            if e == 1:
                qgrid_object.df = self._geo_model.series.df.reset_index().rename(columns={'index': 'series_names'}).astype(
                    {'series_names': str})

            self.update_qgrid_object(qgrid_object)

    @staticmethod
    def update_qgrid_object(qgrid_object):
        qgrid_object._ignore_df_changed = True
        # make a copy of the user's dataframe
        qgrid_object._df = qgrid_object.df.copy()

        # insert a column which we'll use later to map edits from
        # a filtered version of this df back to the unfiltered version
        qgrid_object._df.insert(0, qgrid_object._index_col_name, range(0, len(qgrid_object._df)))

        # keep an unfiltered version to serve as the starting point
        # for filters, and the state we return to when filters are removed
        qgrid_object._unfiltered_df = qgrid_object._df.copy()

        qgrid_object._update_table(update_columns=True, fire_data_change_event=True)
        qgrid_object._ignore_df_changed = False

    def get(self, data_type: str):

        if data_type is 'surfaces':
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
        elif data_type is 'surface_points':
            self.qgrid_in._update_df()
            return self.qgrid_in
        elif data_type is 'orientations':
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
            raise AttributeError('data_type must be either surfaces, series, faults, surface_points, orientations,'
                                 'options, kriging or rescale. UPDATE message')

    def create_faults_qgrid(self):
        faults_object = self._geo_model.faults

        qgrid_widget = qgrid.show_grid(faults_object.df,
                                       show_toolbar=False,
                                       grid_options={'sortable': False, 'highlightSelectedCell': True},
                                       column_options={'editable': True},
                                       column_definitions={'isFinite': {'editable': True}})

        def handle_set_is_fault(event, widget, debug=False):
            if debug is True:
                print(event)
                print(widget)

            if event['column'] == 'isFault':
                idx = event['index']
                #      cat_idx = qgrid_widget.df.loc[idx, 'series_names']

                self._geo_model.set_is_fault([idx], toggle=True)
                self.update_plot()

            if event['column'] == 'isFinite':
                idx = event['index']
                #      cat_idx = qgrid_widget.df.loc[idx, 'series_names']

                self._geo_model.set_is_finite_fault([idx], toggle=True)
                self.update_plot()

            self.update_qgrd_objects()
            self.qgrid_fr._rebuild_widget()

        qgrid_widget.on('cell_edited', handle_set_is_fault)

        return qgrid_widget

    def create_rescaling_data_qgrid(self):
        rescaling_object = self._geo_model.rescaling

        qgrid_widget = qgrid.show_grid(rescaling_object.df,
                                       show_toolbar=False,
                                       grid_options={'sortable': False, 'highlightSelectedCell': True})

        def handle_row_edit(event, widget, debug=False):
            if debug is True:
                print(event)
                print(widget)
                print(qgrid_widget._df)

            value = event['new']
            self._geo_model.modify_rescaling_parameters(event['column'], value)
            self.update_qgrd_objects()
            self.update_plot()

        qgrid_widget.on('cell_edited', handle_row_edit)
        return qgrid_widget

    def create_options_qgrid(self):
        options_object = self._geo_model.additional_data.options

        qgrid_widget = qgrid.show_grid(options_object.df,
                                       show_toolbar=False,
                                       grid_options={'sortable': False, 'highlightSelectedCell': True})

        def handle_row_edit(event, widget, debug=False):
            if debug is True:
                print(event)
                print(widget)
                print(self)
                print(qgrid_widget._df)

            if event['column'] == 'verbosity':
                import numpy as np
                value = np.fromstring(event['new'][1:-1], sep=',')
            else:
                value = event['new']
            self._geo_model.modify_options(event['column'], value)
            self.update_qgrd_objects()
            self.update_plot()

        qgrid_widget.on('cell_edited', handle_row_edit)
        return qgrid_widget

    def create_kriging_parameters_qgrid(self):
        kriging_parameters_object = self._geo_model.additional_data.kriging_data

        qgrid_widget = qgrid.show_grid(kriging_parameters_object.df,
                                       show_toolbar=False,
                                       grid_options={'sortable': False, 'highlightSelectedCell': True},
                                       )

        def handle_row_edit(event, widget, debug=False):

            if debug is True:
                print(event)
                print(widget)
                print(qgrid_widget._df)

            self._geo_model.modify_kriging_parameters(event['column'], event['new'])
            self.update_qgrd_objects()
            self.update_plot()

        qgrid_widget.on('cell_edited', handle_row_edit)
        return qgrid_widget

    def create_faults_relations_qgrid(self):
        faults_object = self._geo_model.faults

        # We need to add the qgrid special columns to categories if does not exist
        try:
            faults_object.faults_relations_df.columns = faults_object.faults_relations_df.columns.add_categories(
                ['index', 'qgrid_unfiltered_index'])
        except ValueError:
            pass

        qgrid_widget = qgrid.show_grid(faults_object.faults_relations_df,
                                       grid_options={'sortable': False, 'highlightSelectedCell': True},
                                       show_toolbar=False)

        def handle_set_fault_relation(event, widget, debug=False):
            if debug is True:
                print(event)
                print(widget)

            # This data frame is quite independent to anything else:
            self._geo_model.set_fault_relation(qgrid_widget.get_changed_df().values)

            # faults_object.faults_relations_df.update(qgrid_widget.get_changed_df())

            self.update_qgrd_objects()
            self.update_plot()
        qgrid_widget.on('cell_edited', handle_set_fault_relation)
        return qgrid_widget

    def create_surface_points_qgrid(self):
        surface_points_object = self._geo_model.surface_points
        self._geo_model.set_default_surfaces()
        self._geo_model.set_default_surface_point(plot_object=self._plot_object)

        qgrid_widget = qgrid.show_grid(
            surface_points_object.df,
            show_toolbar=True,
            grid_options={'sortable': False, 'highlightSelectedCell': True},
            column_options={'editable': False},
            column_definitions={'X': {'editable': True},
                                'Y': {'editable': True},
                                'Z': {'editable': True},
                                'surface': {'editable': True}})

        def handle_row_surface_points_add(event, widget, debug=False):
            if debug is True:
                print(event)
                print(widget)
                print(qgrid_widget._df)
            idx = event['index']

            xyzs = qgrid_widget._df.loc[idx, ['X', 'Y', 'Z', 'surface']]
            self._geo_model.add_surface_points(*xyzs, idx=int(idx), plot_object=self._plot_object)
            self.update_qgrd_objects()

        def handle_row_surface_points_delete(event, widget, debug=False):
            if debug is True:
                print(event)
                print(widget)
            idx = event['indices']

            self._geo_model.delete_surface_points(idx, plot_object=self._plot_object)
            self.update_qgrd_objects()

        def handle_cell_surface_points_edit(event, widget, debug=False):
            if debug is True:
                print(event)
                print(widget)

            column = event['column']
            idx = event['index']
            value = event['new']

            try:
                self._geo_model.modify_surface_points(idx, **{column: value, 'plot_object': self._plot_object})
            except AssertionError:
                pass
            self.update_qgrd_objects()

        qgrid_widget.on('row_removed', handle_row_surface_points_delete)
        qgrid_widget.on('row_added', handle_row_surface_points_add)
        qgrid_widget.on('cell_edited', handle_cell_surface_points_edit)
        return qgrid_widget

    def create_orientations_qgrid(self):
        orientations_object = self._geo_model.orientations
        self._geo_model.set_default_orientation(plot_object=self._plot_object)

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
            xyzs = qgrid_widget._df.loc[idx, ['X', 'Y', 'Z', 'surface']]
            gxyz = qgrid_widget._df.loc[idx, ['G_x', 'G_y', 'G_z']]
            self._geo_model.add_orientations(*xyzs, pole_vector=gxyz.values, idx=int(idx), plot_object=self._plot_object)
            self.update_qgrd_objects()

        def handle_row_orientations_delete(event, widget, debug=False):
            if debug is True:
                print(event)
                print(widget)
            idx = event['indices']

            self._geo_model.delete_orientations(idx, plot_object=self._plot_object)
            self.update_qgrd_objects()

        def handle_cell_orientations_edit(event, widget, debug=False):
            if debug is True:
                print(event)
                print(widget)

            column = event['column']
            idx = event['index']
            value = event['new']

            self._geo_model.modify_orientations(idx, **{column: value, 'plot_object': self._plot_object})
            self.update_qgrd_objects()

        qgrid_widget.on('row_removed', handle_row_orientations_delete)
        qgrid_widget.on('row_added', handle_row_orientations_add)
        qgrid_widget.on('cell_edited', handle_cell_orientations_edit)
        return qgrid_widget

    def create_surfaces_qgrid(self):
        surface_object = self._geo_model.surfaces

        surface_object.set_default_surface_name()
        qgrid_widget = qgrid.show_grid(surface_object.df, show_toolbar=True,
                                       grid_options={'sortable': False, 'highlightSelectedCell': True},
                                       column_options={'editable': True},
                                       column_definitions={'id': {'editable': False},
                                                           'isBasement': {'editable': False}})

        def handle_row_surface_add(event, widget, debug=False):
            if debug is True:
                print(event)
                print(widget)
            idx = event['index']
            self._geo_model.add_surfaces(['surface' + str(idx+1)])
            #surface_object.add_surface(['surface' + str(idx)])
            self.update_qgrd_objects()
            self.qgrid_in._rebuild_widget()
            self.qgrid_or._rebuild_widget()

        def handle_row_surface_delete(event, widget, debug=False):
            if debug is True:
                print(event)
                print(widget)
            idx = event['indices']
            self._geo_model.delete_surfaces(idx)
            #surface_object.delete_surface(idx)
            self.update_qgrd_objects()
            self.qgrid_in._rebuild_widget()
            self.qgrid_or._rebuild_widget()

        def handle_cell_surface_edit(event, widget, debug=False):
            if debug is True:
                print(event)
                print(widget)
            if event['column'] == 'surface':
                self._geo_model.rename_surfaces({event['old']: event['new']})
            if event['column'] == 'series':
                idx = event['index']
                new_series = event['new']
                self._geo_model.map_series_to_surfaces({new_series: surface_object.df.loc[idx, ['surface']]},
                                                       set_series=False, sort_geometric_data=True)
            # if event['column'] == 'isBasement':
            #     idx = event['index']
            #     surface_object.set_basement(surface_object.df.loc[idx, ['surface']])

            if event['column'] == 'order_surfaces':
                idx = event['index']
                try:
                    self._geo_model.modify_order_surfaces(int(event['new']), idx)
                except AssertionError:
                    pass

            self.update_qgrd_objects()

            if event['column'] == 'surface':
                self.qgrid_in._rebuild_widget()
                self.qgrid_or._rebuild_widget()

        qgrid_widget.on('row_removed', handle_row_surface_delete)
        qgrid_widget.on('row_added', handle_row_surface_add)
        qgrid_widget.on('cell_edited', handle_cell_surface_edit)

        return qgrid_widget

    def create_series_qgrid(self):
        series_object = self._geo_model.series

        # I need to do a serious hack because qgrid does not accept categorical index yet
        qgrid_widget = qgrid.show_grid(
            series_object.df.reset_index().rename(columns={'index': 'series_names'}).astype({'series_names': str}),
            show_toolbar=True,
            grid_options={'sortable': False, 'highlightSelectedCell': True},
            column_options={'editable': True},
            column_definitions={'order_series': {'editable': True},
                                })

        def handle_row_series_add(event, widget, debug=False):
            if debug is True:
                print(event)
                print(widget)
                print(series_object.df.reset_index())

            idx = event['index']
            self._geo_model.add_series(['series' + str(idx)])

            # This is because qgrid does not accept editing indeces. We enable the modification of the series name
            # by reindexing the df and change another column
            #self._geo_model.faults.faults_relations_df.columns = self._geo_model.faults.faults_relations_df.columns.add_categories(
            #    ['index', 'qgrid_unfiltered_index'])

            qgrid_widget.df = series_object.df.reset_index().rename(columns={'index': 'series_names'}).astype(
                {'series_names': str})
            self.update_qgrd_objects()
            self.qgrid_fo._rebuild_widget()
            self.qgrid_fr._rebuild_widget()

        def handle_row_series_delete(event, widget, debug=False):
            if debug is True:
                print(event)
                print(widget)
            idx = event['indices']
            cat_idx = qgrid_widget.df.loc[idx, 'series_names']

            self._geo_model.delete_series(cat_idx)

            qgrid_widget.df = series_object.df.reset_index().rename(columns={'index': 'series_names'}).astype(
                {'series_names': str})
            self.update_qgrd_objects()
            self.qgrid_fo._rebuild_widget()
            self.qgrid_fr._rebuild_widget()

        def handle_cell_series_edit(event, widget, debug=False):
            idx = event['index']
            cat_idx = qgrid_widget.df.loc[idx, 'series_names']
            if debug is True:
                print(event)
                print(widget)
                print(cat_idx)
                print(series_object.df.index)
            if event['column'] == 'series_names':
                self._geo_model.rename_series({event['old']: event['new']})
            if event['column'] == 'BottomRelation':
                #series_object.df.loc[cat_idx, 'BottomRelation'] = event['new']
                self._geo_model.set_bottom_relation(cat_idx, event['new'])
            if event['column'] == 'order_series':
                idx = event['index']
                try:
                    self._geo_model.modify_order_series(int(event['new']), idx)
                except AssertionError:
                    pass

            self.update_plot()

         #   self._geo_model.update_from_series(rename_series={event['old']: event['new']})
            # Hack for the faults relations
            print(self._geo_model.faults.faults_relations_df.columns)
           # self._geo_model.faults.faults_relations_df.columns = self._geo_model.faults.faults_relations_df.columns.add_categories(
          #      ['index', 'qgrid_unfiltered_index'])

            qgrid_widget.df = series_object.df.reset_index().rename(columns={'index': 'series_names'}).astype(
                {'series_names': str})

            self.update_qgrd_objects()
            self.qgrid_fo._rebuild_widget()
            self.qgrid_fr._rebuild_widget()

        qgrid_widget.on('row_removed', handle_row_series_delete)
        qgrid_widget.on('row_added', handle_row_series_add)
        qgrid_widget.on('cell_edited', handle_cell_series_edit)

        return qgrid_widget

