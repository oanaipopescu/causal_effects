import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Adjust the path to import from the parent directory

import numpy as np
import networkx as nx
import pytest
from data_structures.causal_graph import CausalGraph

def test_to_networkx_dag():
    adjacency = np.array([
        ['', '-->', ''],
        ['', '', '-->'],
        ['', '', '']
    ], dtype='<U3')
    var_names = ['A', 'B', 'C']
    cg = CausalGraph('DAG', adjacency, var_names)
    nx_graph = cg.to_networkx()
    assert isinstance(nx_graph, nx.DiGraph)
    assert set(nx_graph.nodes) == set(var_names)
    assert set(nx_graph.edges) == {('A', 'B'), ('B', 'C')}

def test_to_networkx_cpdag():
    adjacency = np.array([
        ['', '-->', '---'],
        ['<--', '', ''],
        ['---', '', '']
    ], dtype='<U3')
    var_names = ['X', 'Y', 'Z']
    cg = CausalGraph('CPDAG', adjacency, var_names)
    nx_graph = cg.to_networkx()
    assert set(nx_graph.edges) == {('X', 'Y'), ('X', 'Z'), ('Z', 'X')}

def test_to_networkx_admg():
    adjacency = np.array([
        ['', '-->', '<->'],
        ['', '', ''],
        ['<->', '', '']
    ], dtype='<U3')
    var_names = ['U', 'V', 'W']
    cg = CausalGraph('ADMG', adjacency, var_names)
    nx_graph = cg.to_networkx()
    assert ('U', 'V') in nx_graph.edges

def test_to_networkx_pag():
    adjacency = np.array([
        ['', 'o->', 'o-o'],
        ['', '', '-->'],
        ['', '', '']
    ], dtype='<U3')
    var_names = ['P', 'Q', 'R']
    cg = CausalGraph('PAG', adjacency, var_names)
    nx_graph = cg.to_networkx()
    assert ('P', 'Q') in nx_graph.edges
    assert ('P', 'R') in nx_graph.edges
    assert ('Q', 'R') in nx_graph.edges

def test_get_parents_str():
    adjacency = np.array([
        ['', '-->', ''],
        ['', '', '-->'],
        ['', '', '']
    ], dtype='<U3')
    var_names = ['A', 'B', 'C']
    cg = CausalGraph('DAG', adjacency, var_names)
    assert cg.get_parents('B') == ['A']
    assert cg.get_parents('C') == ['B']
    assert cg.get_parents('A') == []

def test_get_parents_int():
    adjacency = np.array([
        ['', '-->', ''],
        ['', '', '-->'],
        ['', '', '']
    ], dtype='<U3')
    var_names = ['A', 'B', 'C']
    cg = CausalGraph('DAG', adjacency, var_names)
    assert cg.get_parents(1) == [0]
    assert cg.get_parents(2) == [1]
    assert cg.get_parents(0) == []

def test_get_parents_tuple():
    # 3D adjacency matrix for time series or similar use-case
    adjacency = np.zeros((2, 2, 2), dtype='<U3')
    adjacency[0, 1, 0] = '-->'
    cg = CausalGraph('DAG', adjacency, ['A', 'B'])
    assert cg.get_parents((1, 0)) == [(0, 0)]

def test_get_parents_invalid_str():
    adjacency = np.array([
        ['', '-->'],
        ['', '']
    ], dtype='<U3')
    var_names = ['A', 'B']
    cg = CausalGraph('DAG', adjacency, var_names)
    with pytest.raises(ValueError):
        cg.get_parents('C')

def test_get_parents_invalid_tuple():
    adjacency = np.zeros((2, 2, 2), dtype='<U3')
    cg = CausalGraph('DAG', adjacency, ['A', 'B'])
    with pytest.raises(ValueError):
        cg.get_parents((1, 1))

if __name__ == "__main__":
    # Run only this file's tests
    raise SystemExit(pytest.main([__file__]))