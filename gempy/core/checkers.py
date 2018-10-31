# TODO
# - Check the basement layer is not in in Interfaces and Orientations


def check_fault_relations(self):
    # Method to check that only older df offset newer ones?
    #
    # try:
    #     # Check if there is already a categories_df
    #     self.df
    #
    #     try:
    #         if any(self.df.columns != series.columns):
    #             series_fault = self.count_faults()
    #             self.df = pn.DataFrame(index=series.columns, columns=['isFault'])
    #             self.df['isFault'] = self.df.index.isin(series_fault)
    #     except ValueError:
    #         series_fault = self.count_faults()
    #         self.df = pn.DataFrame(index=series.columns, columns=['isFault'])
    #         self.df['isFault'] = self.df.index.isin(series_fault)
    #
    #     if series_fault:
    #         self.df['isFault'] = self.df.index.isin(series_fault)
    #
    # except AttributeError:
    #
    #     if not series_fault:
    #         series_fault = self.count_faults()
    #         self.df = pn.DataFrame(index=series.columns, columns=['isFault'])
    #         self.df['isFault'] = self.df.index.isin(series_fault)

    # self.interfaces.loc[:, 'isFault'] = self.interfaces['series'].isin(self.df.index[self.df['isFault']])
    # self.orientations.loc[:, 'isFault'] = self.orientations['series'].isin(
    #     self.df.index[self.df['isFault']])
    pass