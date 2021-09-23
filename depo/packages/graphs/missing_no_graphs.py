from .graph import Graph
import missingno as msno

from pandas.core.base import PandasObject

class Missing_no_graphs(Graph):

#----------------------------------------------------------

    def _print_bar_chart(self):
        return msno.bar(self, color="dodgerblue", sort="ascending", figsize=(10,5), fontsize=12)

    PandasObject.print_nan_bar_chart = _print_bar_chart

#----------------------------------------------------------

    def _print_matrix(self):
       return msno.matrix(self, sparkline=False, figsize=(10,5), fontsize=12, color=(0.27, 0.52, 1.0))

    PandasObject.print_nan_matrix = _print_matrix

#----------------------------------------------------------

    def _print_heat_map(self):
        return msno.heatmap(self, cmap="RdYlGn", figsize=(10,5), fontsize=12)

    PandasObject.print_nan_heat_map = _print_heat_map

#----------------------------------------------------------

    def _print_dendrogram(self):
        return msno.dendrogram(self, figsize=(10,5), method="centroid", fontsize=10)

    PandasObject.print_nan_dendrogram = _print_dendrogram
