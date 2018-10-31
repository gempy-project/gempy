
import warnings
try:
    import qgrid
except ImportError:
    warnings.warn('qgrid package is not installed. No interactive dataframes available.')


def interactive_df_open(geo_data, itype):
    """
    Open the qgrid interactive DataFrame (http://qgrid.readthedocs.io/en/latest/).
    To seve the changes see: :func:`~gempy.gempy_front.interactive_df_change_df`


    Args:
        geo_data (:class:`gempy.data_management.InputData`)
        itype(str {'all', 'interfaces', 'orientaions', 'formations', 'series', 'df', 'fautls_relations'}): input
            data type to be retrieved.

    Returns:
        :class:`DataFrame`: Interactive DF
    """
    return geo_data.interactive_df_open(itype)


def interactive_df_change_df(geo_data, only_selected=False):
    """
    Confirm and return the changes made to a dataframe using qgrid interactively. To update the
    :class:`gempy.data_management.InputData` with the modify categories_df use the correspondant set function.

    Args:
        geo_data (:class:`gempy.data_management.InputData`): the same :class:`gempy.data_management.InputData`
            object used to call :func:`~gempy.gempy_front.interactive_df_open`
        only_selected (bool) if True only returns the selected rows

    Returns:
        :class:`DataFrame`
    """
    return geo_data.interactive_df_get_changed_df(only_selected=only_selected)