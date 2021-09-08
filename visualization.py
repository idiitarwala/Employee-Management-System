"""
File containing functions to visualize graph using networkx and matplotlib.
"""
from typing import List, Union, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import networkx as nx

plt.ioff()


def visualize(data: List[Tuple[str, str]], root: Union[str, int]) -> None:
    """
    This function takes in a parameter: data which is a graph dataset where each tuple represents
    an edge in the graph. The root parameter is the node at the top of the graph with a depth of
    0 (all other nodes will fall under this root node).

    This function takes the data, converts it to a networkx graph and visualizes it using the
    matplotlib library.

    NOTE: We are enforcing a tree-like hierarchical structure for our graph visualizations.
    """
    fig = plt.figure()
    graph = nx.Graph()
    if isinstance(data, str):
        graph.add_node(data)
    else:
        graph.add_edges_from(data)
    pos = enforce_tree_structure(graph, root)
    nx.draw(graph, pos=pos, with_labels=True, node_shape="s", node_color="none",
            bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))

    plt.savefig('graph.png')
    plt.close(fig)
    print('Image has been created')


def enforce_tree_structure(graph: nx.Graph, root: Optional[int] = None,
                           width: float = 1.0, vertical_space: float = 0.2,
                           vertical_location: float = 0, horizontal_location: float = 0.5) -> \
        Dict[Union[str, int], Tuple[float, float]]:
    """
    This function returns a dictionary with the keys representing the nodes of the graph and
    their respective values being a tuple containing the vertical and horizontal location of the
    nodes in the visualization.

    This function calls a helper method which recursively sets the position of each node such
    that it represents a hierarchical structure of a tree.
    """
    visited = []
    return _enforce_tree_structure(graph, root, visited, width, vertical_space, vertical_location,
                                   horizontal_location)


def _enforce_tree_structure(graph: nx.Graph, root: Union[str, int], visited: List[Union[str, int]],
                            width: float = 1.0, vertical_space: float = 0.2,
                            vertical_location: float = 0.0, horizontal_location: float = 0.5,
                            pos: Any = None, parent: Any = None) -> \
        Dict[Union[str, int], Tuple[float, float]]:
    """
    This function is a helper function of enforce_tree_structure which uses DFS to set the positions
    of each node where the graph is spaced out to look like a tree.

    NOTE: this algorithm was inspired from a post on StackOverFlow (cited in report)

    """
    if graph.neighbors(root) is None:
        return pos
    else:
        if root not in visited:
            visited.append(root)
            if pos is None:
                pos = {root: (horizontal_location, vertical_location)}
            else:
                pos[root] = (horizontal_location, vertical_location)
            neighbors = list(graph.neighbors(root))
            if parent is not None:
                neighbors.remove(parent)
            if len(neighbors) != 0:
                dx = width / len(neighbors)
                nextx = horizontal_location - width / 2 - dx / 2
                for neighbor in neighbors:
                    nextx += dx
                    pos = _enforce_tree_structure(graph, neighbor, width=dx,
                                                  vertical_space=vertical_space,
                                                  vertical_location=vertical_location - vertical_space,
                                                  horizontal_location=nextx, pos=pos,
                                                  parent=root, visited=visited)
            return pos


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'max-line-length': 1000,
        # E1136, R1710 for Optional[] typing, R0913 for too many arguments
        'disable': ['E1136', 'R1710', 'R0913', 'E9998'],
        'extra-imports': ['matplotlib.pyplot', 'networkx'],
        'max-nested-blocks': 4
    })
