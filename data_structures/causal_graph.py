import networkx as nx
from typing import List, Union, Tuple
import numpy as np

class CausalGraph:
    """
    Causal graph representation built from a Tigramite-style string adjacency matrix and variable names.
    Provides a method to convert to a networkx.DiGraph.
    """
    def __init__(self, graph_type: str, adjacency_matrix: np.ndarray, variable_names: List[str]) -> None:
        """
        Initialize the CausalGraph.

        Parameters:
        - type: String indicating the type of causal graph - 'DAG' or 'CPDAG' for no hidden conf., 'ADMG' or 'PAG' for hidden conf.
        - adjacency_matrix: 2D numpy array with dtype='<U3' representing edge types ('', '-->', 'o->', etc.)
        - variable_names: List of variable names corresponding to matrix indices
        """
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        if adjacency_matrix.shape[0] != len(variable_names):
            raise ValueError("Number of variable names must match adjacency matrix size")
        if graph_type not in {"dag", "cpdag", "admg", "pag"}:
            raise ValueError("graph_type must be one of: 'DAG', 'CPDAG', 'ADMG', 'PAG'")

        self.graph_type = graph_type
        self.adjacency_matrix = adjacency_matrix
        self.variable_names = variable_names
        self.is_timeseries = True if adjacency_matrix.ndim == 3 else False

    def get_parents(self, variable: Union[str, Tuple[int, int], int]) -> Union[List[str], List[Tuple[int, int]]]:
        """
        Get the variable names of parent variables for a given variable.

        Parameters:
        - variable: name of the variable

        Returns:
        - List of names of parent variables
        """
        parents = []
        if isinstance(variable, tuple):
            if len(variable) != 2 or variable[1] > 0:
                raise ValueError("Tuple variable must be of length 2 and second element must be 0")
            for i in range(self.adjacency_matrix.shape[0]):
                for k in range(self.adjacency_matrix.shape[2]):
                    if self.adjacency_matrix[i, variable[0], k] == '-->' or self.adjacency_matrix[i, variable[0], k] == 'o->':
                        parents.append((i, k))
        elif isinstance(variable, int):
            for i in range(self.adjacency_matrix.shape[0]):
                if self.adjacency_matrix[i, variable] == '-->' or self.adjacency_matrix[i, variable] == 'o->':
                    parents.append((i, 0))
        elif isinstance(variable, str):
            if variable not in self.variable_names:
                raise ValueError(f"Variable '{variable}' not found in the graph.")
            index = self.variable_names.index(variable)
            for i in range(self.adjacency_matrix.shape[0]):
                if self.adjacency_matrix[i, index] == '-->' or self.adjacency_matrix[i, index] == 'o->':
                    parents.append(self.variable_names[i])

        return parents

    def get_variable_indices(self, variable: List[str]) -> List[int]:
        """
        Get the indices of the given variable names in the adjacency matrix.

        Parameters:
        - variable: List of variable names

        Returns:
        - List of indices corresponding to the variable names
        """
        indices = []
        for var in variable:
            if var not in self.variable_names:
                raise ValueError(f"Variable '{var}' not found in the graph.")
            indices.append((self.variable_names.index(var), 0))
        return indices
    
    def to_networkx(self) -> nx.DiGraph:
        """
        Convert the internal adjacency matrix to a networkx.DiGraph.
        Edge interpretation depends on the graph_type.
        """
        G = nx.DiGraph()
        G.add_nodes_from(self.variable_names)
        for i, src in enumerate(self.variable_names):
            for j, tgt in enumerate(self.variable_names):
                edge_type = self.adjacency_matrix[i, j]

                if self.graph_type in {"dag", "admg"}:
                    if '-->' in edge_type:
                        G.add_edge(src, tgt)
                    # Bidirected edges for ADMG can be stored separately if needed
                elif self.graph_type == "cpdag":
                    if '-->' in edge_type:
                        G.add_edge(src, tgt)
                    elif '<--' in edge_type:
                        G.add_edge(tgt, src)
                    elif '---' in edge_type:
                        G.add_edge(src, tgt)
                        G.add_edge(tgt, src)
                elif self.graph_type == "pag":
                    if '-->' in edge_type or 'o->' in edge_type or 'o-o' in edge_type:
                        G.add_edge(src, tgt)
        return G

    def save_numpy(self, path: str) -> None:
        """
        Save the adjacency matrix, variable names, and graph type to a .npz file.
        """
        np.savez_compressed(
            path,
            adjacency_matrix=self.adjacency_matrix,
            variable_names=self.variable_names,
            graph_type=self.graph_type
        )
    def __repr__(self):
        return f"CausalGraph(variables={self.variable_names})"
