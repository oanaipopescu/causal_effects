import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple

from .causal_graph import CausalGraph
from tigramite.data_processing import DataFrame as TigramiteDataFrame
from tigramite.causal_effects import CausalEffects
from dowhy import CausalModel


class DataLoader:
    def __init__(self, data: pd.DataFrame, graph: CausalGraph):
        """
        Initialize with dataset and causal graph.

        Parameters:
        - data: pandas DataFrame
        - graph: CausalGraph instance
        """
                
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise ValueError("data must be a pandas DataFrame")

        if len(data.columns) == 0:
            raise ValueError("The dataset is empty. Please provide a non-empty DataFrame.")
        elif len(data.columns) != len(graph.variable_names):
            raise ValueError("The number of columns in the dataset must match the number of variables in the graph.")

        self.graph = graph
        self.variable_names = self.graph.variable_names

    def plot_scatter_matrix(self):
        """Plot scatterplots for each pair of variables."""
        pd.plotting.scatter_matrix(self.data[self.variable_names], figsize=(12, 12))
        plt.suptitle("Scatter Matrix")
        plt.tight_layout()
        plt.show()

    def to_tigramite_dataframe(self) -> TigramiteDataFrame:
        """Convert to Tigramite DataFrame."""
        data_array = self.data.to_numpy()
        return TigramiteDataFrame(data=data_array, var_names=self.variable_names)


    def generate_tigramite_causal_effects(self, A: List[Tuple[int, int]], 
                                          Y: List[Tuple[int, int]], S: Union[None, List[Tuple[int, int]]] = None,
                                          hidden_vars: Union[None, List[Tuple[int, int]]] = None) -> CausalEffects:
        """Create Tigramite CausalEffects object from the graph and data."""
        return CausalEffects(graph=self.graph.adjacency_matrix, graph_type=self.graph.graph_type,
                             X=A, Y=Y, S=S, hidden_variables=hidden_vars)

    def to_dowhy_model(self, A: List[int], Y: List[int]) -> CausalModel:
        """
        Create a DoWhy CausalModel.

        Parameters:
        - A: name of treatment variable
        - Y: name of outcome variable

        Returns:
        - CausalModel
        """
        nx_graph = self.graph.to_networkx()
        return CausalModel(
            data=self.data,
            treatment=A,
            outcome=Y,
            graph=nx_graph
        )
