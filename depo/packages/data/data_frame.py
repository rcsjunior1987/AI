import pandas as pd
from pandas.core.base import PandasObject
from ..messages.error import Error as errorMessage

from .frame import Frame

class Data_Frame(Frame):

    def _get_file_handled(self):
        return super()._get_file_handled()

#----------------------------------------------------------

    def _set_file_handled(self, value):
        return super()._set_file_handled(value)

#----------------------------------------------------------

    def _get_exception(self):
        return super()._get_exception()

#----------------------------------------------------------

    def _set_exception(self, value):
        return super()._set_exception(value)

#----------------------------------------------------------

    def __read_csv(self):
        return pd.read_csv(self._get_file_handled())

#----------------------------------------------------------

    def _read_csv(self):
        self._set_exception(None)

        try:
            if (self._get_file_handled() != None):
                self._set_file_handled(self.__read_csv())
                return pd.DataFrame(self._get_file_handled())
        except Exception as e:
            self._set_exception(e.__class__)
            errorMessage(self._get_exception(), self._get_file_handled().name).print()

        return pd.read_csv(self._get_file_handled())

#----------------------------------------------------------

    def merge(self, data_set_on_right, on=None, how='inner'):
        return pd.merge(self
                      , data_set_on_right
                      , on=on
                      , how = how)

    PandasObject.merge = merge

#----------------------------------------------------------

    def print_nan_columns(self):
        # Total missing values per Column
        mis_val = self.isnull().sum()

        # Total 0 values per Column    
        zero_val = (self == 0.00).astype(int).sum(axis=0)

        zero_val_percent = 100 * zero_val / len(self)
              
        # Percentage of missing values per Column
        mis_percent = 100 * self.isnull().sum() / len(self)

        mis_zero_val = (zero_val + mis_val)

        mis_zero_percent = 100 * mis_zero_val / len(self)
      
        dtypes_table = self.dtypes

        # Make a table with the results
        mis_table = pd.concat([mis_val
                             , mis_percent
                             , zero_val
                             , zero_val_percent
                             , mis_zero_val
                             , mis_zero_percent
                             , dtypes_table]
                             , axis=1)

        # Rename the columns
        mis_columns = mis_table.rename(
        columns = {0 : 'Missing Values'
                 , 1 : '% Missing Values'
                 , 2 : 'Zero  Values'
                 , 3 : '% Missing Values'
                 , 4 : 'Zero Missing Values'
                 , 5 : '% Zero Missing Values'
                 , 6 : 'Data Type'
                 })

        # Sort the table by percentage of missing descending
        mis_columns = mis_columns.sort_values(
        'Missing Values', ascending=False).round(2)

        # Print some summary information
        print ("The selected dataframe has " + str(self.shape[1]) + " columns.\n"      
            "   There are " + str(mis_columns.shape[0]) +
              " columns that have missing values.")

        #----------------------------

        # Total missing values
        mis_val_total = pd.Series(mis_val.sum()) 

        zero_val_total = pd.Series(zero_val.sum())

        # Total of missing values
        mis_percent_total = pd.Series(mis_percent.sum())

        zero_val_percent_total = pd.Series(zero_val_percent.sum()) 

        mis_zero_val_total = pd.Series(mis_zero_val.sum())

        mis_zero_percent_total = pd.Series(mis_zero_percent.sum())

        dtypes_table = pd.Series("-")
        
        mis_table_total = pd.concat([mis_val_total
                                   , mis_percent_total
                                   , zero_val_total
                                   , zero_val_percent_total
                                   , mis_zero_val_total
                                   , mis_zero_percent_total
                                   , dtypes_table]
                                   , axis=1)

        mis_table_total = mis_table_total.set_axis(['Total'])

        # Rename the columns
        mis_columns_total = mis_table_total.rename(
        columns = {0 : 'Missing Values'
                 , 1 : '% Missing Values'
                 , 2 : 'Zero  Values'
                 , 3 : '% Missing Values'
                 , 4 : 'Zero Missing Values'
                 , 5 : '% Zero Missing Values'
                 , 6 : 'Data Type'
                 })
        
        mis_columns = pd.concat([mis_columns, mis_columns_total], axis=0)

        #----------------------------
        
        # Return the dataframe with missing information
        return mis_columns

    PandasObject.print_nan_columns = print_nan_columns

#----------------------------------------------------------

    def check_balancing(self):
        pass