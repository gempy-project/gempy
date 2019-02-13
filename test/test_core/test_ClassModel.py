import gempy.core.data as gd
import gempy.core.gempy_api as gp
import pandas as pn
import numpy as np
import os
import pytest
input_path = os.path.dirname(__file__)+'/../../notebooks'

class TestModel:

    def test_test(self):

        model = gp.Model()


        model.read_data(path_i=input_path + "/input_data/tut_chapter1/simple_fault_model_points.csv",
                        path_o=input_path + "/input_data/tut_chapter1/simple_fault_model_orientations.csv")

        gp.map_series_to_formations(model, {"Fault_Series": 'Main_Fault',
                                                "Strat_Series": ('Sandstone_2', 'Siltstone',
                                                                 'Shale', 'Sandstone_1', 'basement')})
        # In[20]:

        model.map_series_to_formations({"Fault_Series": ('Main_Fault', 'Silstone'),
                                        "Strat_Series": ('Sandstone_2'
                                                         'Shale',)})

        # In[21]:

        model.formations

        # In[16]:

        model.series

        # In[14]:

        model.interfaces

        # In[ ]:

        # In[ ]:

        # In[ ]:

        model.formations.df['series']

        # In[ ]:

        series_list = list(mapping_object.keys())
        model.series.set_series_index(series_list)

        # In[22]:

        mapping_object = {"Fault_Series": 'Main_Fault',
                          "Strat_Series": ('Siltstone',
                                           'Shale', 'Sandstone_1')}
        if type(mapping_object) is dict:

            s = []
            f = []
            for k, v in mapping_object.items():
                for form in np.atleast_1d(v):
                    s.append(k)
                    f.append(form)

            # TODO does series_mapping have to be in self?
            new_series_mapping = pn.DataFrame([pn.Categorical(s, model.series.df.index)],
                                              f, columns=['series'])

        # In[23]:

        new_series_mapping

        # In[ ]:

        new_series_mapping.index

        # In[24]:

        b = model.formations.df['formation'].isin(new_series_mapping['series'].index)
        b

        # In[ ]:

        new_series_mapping

        # In[ ]:

        idx = model.formations.df.index[b]
        idx

        # In[ ]:

        model.formations.df.loc[idx, 'series'] = model.formations.df.loc[idx, 'formation'].map(
            new_series_mapping['series'])

        # In[ ]:

        model.formations

        # In[ ]:

        model.formations.df['series'][b]

        # In[ ]:

        model.formations.df['series'][b] = model.formations.df['formation'][b].map(new_series_mapping['series'])

        # In[ ]:

        model.formations.df['formation'].map(new_series_mapping['series'])

        # In[ ]:

        # model.formations._series_mapping['series'].cat.add_categories(new_series_mapping['series'].cat.categories, inplace =True)

        # In[ ]:

        a = new_series_mapping['series'].cat.categories
        a.isin(new_series_mapping['series'].cat.categories)

        # In[ ]:

        model.formations._series_mapping['series']

        # In[ ]:

        new_series_mapping['series']

        # In[ ]:

        a

        # In[ ]:

        new_series_mapping.append(model.formations._series_mapping, verify_integrity=False)['series']

        # In[ ]:

        model.formations._series_mapping

        # In[ ]:

        a = model.formations.df['series'].cat
        a.add_categories(['foo', 'bla'])

        # In[ ]:

        import pandas as pd

        df = pd.DataFrame([], columns=['a', 'b'])
        df['a'] = pd.Categorical([], [0, 1])

        new_df = pd.DataFrame.from_dict({'a': [0, 1, 1, 1, 0, 0], 'b': [1, 1, 8, 4, 0, 0]})
        new_df['a'] = pd.Categorical(new_df['a'], [0, 1])

        df.append(new_df, ignore_index=False)['a']

        # In[ ]:

        df

        # In[ ]:

        # In[ ]:

        # In[ ]:

        # In[ ]:

        # In[ ]:

        model.interfaces.df.head()

        # In[ ]:

        model.orientations.df.head()

        # Next we need to categorize each surface into the right series. This will update all the Dataframes depending on `Formations` and `Series` to the right categories:

        # In[ ]:

        model.map_series_to_formations({"Fault_Series": 'Main_Fault',
                                        "Strat_Series": ('Sandstone_2', 'Siltstone',
                                                         'Shale', 'Sandstone_1')})

        # In[ ]:

        get_ipython().run_line_magic('debug', '')

        # In[ ]:

        model.formations.df['series']

        # In[ ]:

        model.formations

        # In[ ]:

        model.interfaces.df.head()

        # In[ ]:

        model.series

        # In the case of having faults we need to assign wich series are faults:

        # In[ ]:

        model.faults

        # In[ ]:

        model.set_is_fault(['Fault_Series'])

        # In[ ]:

        model.interfaces.df.head()

        # Again as we can see, as long we use the model methods all the dependent objects change inplace accordingly. If for any reason you do not want this behaviour you can always use the individual methods of the objects (e.g. `model.faults.set_is_fault`)

        # In[ ]:

        model.additional_data

        # ## Setting grid
        #
        # So far we have worked on data that depends exclusively of input (i.e. sequeantial pile, interfaces, orientations, etc). With things like grid the idea is the same:

        # In[ ]:

        model.grid.values

        # In[ ]:

        model.set_regular_grid([0, 10, 0, 10, 0, 10], [50, 50, 50])

        # In[ ]:

        model.additional_data

        # -------------------

        # ## Getting data

        # Alternatively we can access the dataframe by:

        # In[ ]:

        gp.get_data(geo_model, 'formations')

        # The class `gempy.core.model.Model` works as the parent container of our project. Therefore the main step of any project is to create an instance of this class. In the official documentation we use normally geo_model (geo_data in the past) as name of this instance.
        #
        # When we instiantiate a `Model` object we full data structure_data is created. By using `gp.init_data` and `set_series` we set the default values -- given the attributes -- to all of fields. Data is stored in pandas dataframes. With `gp.get_data` and the name of the data object it is possible to have access to the dataframes:
        #
        # `str`['all', 'interfaces', 'orientations', 'formations', 'series', 'faults', 'faults_relations',
        #         additional data]
        #
        # These dataframes are stored in specific objects. These objects contain the specific methods to manipulate them. You access these objects with the spectific getter or as a attribute of `Model`

        # In[ ]:



