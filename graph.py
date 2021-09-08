"""
File containing relevant graph classes and various utility methods
regarding them.
"""
from typing import List, Dict, Set, Optional, Union
from collections import deque
import visualization


class Vertex:
    """
    Vertex class to represent employee data.

    Instance Attributes:
        - name: String containing first and last name of employee
        - uid: Integer which represents unique id of employee
        - performance: Integer representing how well the employee is performing
        - performance_progression: Array of past performance values
        - depth: If used with Tree class, this is integer representing depth of node

    Representation Invariants:
        - not self.name.isdigit()
        - 0 <= self.performance <= 100
    """
    name: str
    uid: int
    performance: int
    performance_progression: List[int]
    depth: Optional[int]

    def __init__(self, name: str, uid: int, performance: int = 50) -> None:
        self.name = name
        self.uid = uid
        self.performance = performance
        self.performance_progression = []
        self.depth = None


class Graph:
    """
    Graph template class. Contains no restrictions (ie. tree vs directed acyclic graph).
    Supports basic graph computations.

    Instance Attributes:
        - vertices: Dictionary mapping uid to corresponding Vertex object
        - edges: Dictionary mapping uid to set of corresponding connection uid
    """
    # Private Instance Attributes:
    #     - _counter: Integer that can be incremented to generate uid
    _counter: int
    vertices: Dict[int, Vertex]
    edges: Dict[int, Set[int]]

    def __init__(self) -> None:
        self._counter = 0
        self.vertices = {}
        self.edges = {}

    def __len__(self) -> int:
        """
        Return length of graph. Length is to be defined as number of vertices.

        >>> g = Graph()
        >>> len(g)
        0
        """
        return len(self.vertices)

    def has(self, uid: int) -> bool:
        """
        Return True if uid in self.vertices. False otherwise.

        >>> g = Graph()
        >>> g.has(1)
        False
        """
        return uid in self.vertices

    def get(self, uid: int) -> Optional[Vertex]:
        """
        Get corresponding Vertex object given uid.
        """
        if not self.has(uid):
            print('The uid ' + str(uid) + ' is not contained in graph.')
            return
        return self.vertices[uid]

    def add_vertex(self, vertex: Vertex) -> None:
        """
        Given a Vertex object, add it to the graph.

        >>> g = Graph()
        >>> g.add_vertex(Vertex('Bob', 1))
        >>> g.has(1)
        True
        """
        if self.has(vertex.uid):
            print('A vertex with uid ' + str(vertex.uid) + ' already exists in graph.')
            return
        self.vertices[vertex.uid] = vertex
        self.edges[vertex.uid] = set()

    def add_edge(self, start: int, end: int) -> None:
        """
        Given start and end uid, add directed edge to the graph: start -> end.
        """
        if not self.has(start):
            print('The uid ' + str(start) + ' is not contained in graph.')
            return
        if not self.has(end):
            print('The uid ' + str(end) + ' is not contained in graph.')
            return
        if end in self.edges[start]:
            print('This edge already exists.')
            return
        self.edges[start].add(end)

    def update_vertex_performance(self, uid: int, new_score: int) -> None:
        """
        Update the performance score of vertex.

        Preconditions:
            - 0 <= new_score <= 100
        """
        if not self.has(uid):
            print('The uid ' + str(uid) + ' is not contained in graph.')
        vertex = self.vertices[uid]
        vertex.performance_progression.append(vertex.performance)
        vertex.performance = new_score

    @property
    def next_uid(self) -> int:
        """
        Advance _counter to generate next uid.

        >>> graph = Graph()
        >>> graph.next_uid
        1
        >>> graph.next_uid
        2
        """
        self._counter += 1
        self._counter = max(self._counter, len(self) + 1)
        return self._counter

    def get_connected_component(self, start: int) -> Set[int]:
        """
        Return set of uid of connected component given starting uid.
        If uid is not valid, return empty set.

        CAUTION: only use when graph is undirected.

        >>> g = Graph()
        >>> g.get_connected_component(1) == set()
        True

        >>> g = Graph()
        >>> g.add_vertex(Vertex('Bob', 1))
        >>> g.add_vertex(Vertex('Joe', 2))
        >>> g.add_vertex(Vertex('Howard', 3))
        >>> g.add_edge(1, 2)
        >>> g.add_edge(2, 1)
        >>> g.get_connected_component(1) == {1, 2}
        True
        """
        if not self.has(start):
            return set()
        visited = set()
        self._dfs(start, visited)
        return visited

    def get_connected_components(self) -> List[Set[int]]:
        """
        Return a list of sets containing uid of vertices.
        Each element of list should represent one connected component.
        """
        big_visited = set()
        components = []
        # Perform DFS algorithm
        for uid in self.vertices:
            if uid in big_visited:
                continue
            component = self.get_connected_component(uid)
            components.append(component)
            big_visited = big_visited.union(component)
        return components

    def _dfs(self, start: int, visited: Set[int]) -> None:
        """
        Perform DFS on graph starting with start uid.
        """
        if start in visited:
            return
        visited.add(start)
        for uid in self.edges[start]:
            self._dfs(uid, visited)

    @property
    def is_connected(self) -> bool:
        """
        Return boolean indicating if graph is connected.
        """
        cc = self.get_connected_components()
        if len(cc) == 0 or len(cc) == 1:
            return True
        return False

    def shortest_path(self, start: int, end: int) -> Optional[List[int]]:
        """
        Given start uid and end uid, return list of uid indicating shortest path between them.

        >>> g = Graph()
        >>> g.add_vertex(Vertex('Bob', 1))
        >>> g.add_vertex(Vertex('Joe', 2))
        >>> g.add_vertex(Vertex('Howard', 3))
        >>> g.add_edge(1, 2)
        >>> g.add_edge(2, 1)
        >>> g.add_edge(2, 3)
        >>> g.add_edge(3, 2)
        >>> g.shortest_path(1, 3) == [1, 2, 3]
        True
        """
        if not self.has(start):
            print('The uid ' + str(start) + ' is not contained in graph.')
            return
        if not self.has(end):
            print('The uid ' + str(end) + ' is not contained in graph.')
            return

        queue = deque()
        visited = set()
        parent = {}

        visited.add(start)
        queue.append(start)
        parent[start] = None
        while len(queue) != 0:
            front = queue.popleft()
            for vertex in self.edges[front]:
                if vertex in visited:
                    continue
                visited.add(vertex)
                queue.append(vertex)
                parent[vertex] = front

        path = []
        curr = end
        while curr in parent:
            path.append(curr)
            curr = parent[curr]

        return path[::-1]


class Tree(Graph):
    """
    Tree class which extends Graph.
    """
    # Private Instance Attributes:
    #     - _root: Integer denoting uid of root Vertex
    _root: Optional[int]

    def __init__(self) -> None:
        super().__init__()
        self._root = None

    def add_root(self, vertex: Vertex) -> None:
        """
        Add root vertex to tree.

        >>> t = Tree()
        >>> t.add_root(Vertex('Bob', 1))
        >>> t.has(1)
        True
        """
        if self._root is not None:
            print('Root vertex already present in graph.')
            return
        self.add_vertex(vertex)
        self._root = vertex.uid

    def add_edge(self, start: int, end: int) -> None:
        """
        Given start and end uid, add undirected edge to the graph: start <-> end.
        """
        if not self.has(start):
            print('The uid ' + str(start) + ' is not contained in graph.')
            return
        if not self.has(end):
            print('The uid ' + str(end) + ' is not contained in graph.')
            return
        if end in self.edges[start]:
            print('This edge already exists.')
            return
        self.edges[start].add(end)
        self.edges[end].add(start)

        # Check if cyclic
        if self.is_cyclic:
            print('Adding edge between uid ' + str(start) + ' and ' + str(end)
                  + ' makes graph cyclic.')
            self.edges[start].remove(end)
            self.edges[end].remove(start)

    @property
    def is_cyclic(self) -> bool:
        """
        Return True if graph contains a cycle.
        False otherwise.
        """
        cnt = 0
        for uid in self.vertices:
            cnt += len(self.edges[uid])
        cnt /= 2
        if cnt <= len(self) - 1:
            return False
        return True

    def dfs(self, start: int, visited: Set[int],
            parent: Dict[int, Optional[int]], depth: int = 0) -> None:
        """
        Perform DFS on graph starting with start uid.
        Update vertex depths.
        """
        visited.add(start)
        self.vertices[start].depth = depth
        for uid in self.edges[start]:
            if uid in visited:
                continue
            parent[uid] = start
            self.dfs(uid, visited, parent, depth + 1)

    def lca(self, first: int, second: int) -> Optional[int]:
        """
        Find lowest comment ancestor of first and second.
        """
        if not self.has(first):
            print('The uid ' + str(first) + ' is not contained in graph.')
            return
        if not self.has(second):
            print('The uid ' + str(second) + ' is not contained in graph.')
            return
        # Update depths
        parent = {self._root: None}
        self.dfs(self._root, set(), parent)
        i = first
        j = second
        while self.vertices[i].depth != self.vertices[j].depth:
            if self.vertices[i].depth < self.vertices[j].depth:
                j = parent[j]
            else:
                i = parent[i]
        while i != j:
            i = parent[i]
            j = parent[j]
        print('Response to RESOLVE ' + str(first) + ' ' + str(second))
        print('\tCommon superior to uid ' + str(first) + ' and '
              + str(second) + ' is uid ' + str(i))
        path = self.shortest_path(first, second)
        print('\tPath of chain of command is ' + str(path))
        return i

    def delete_vertex(self, uid: int) -> None:
        """
        Given vertex uid, delete the vertex and promote a child if necessary.
        Promotion strategy: if has child, promote child with highest performance.
        If multiple children tie for highest performance, pick first one iterated over.
        """
        if uid == self._root:
            print('Cannot delete root vertex.')
            return
        if not self.has(uid):
            print('The uid ' + str(uid) + ' is not contained in graph.')
            return
        if len(self.edges[uid]) > 1:
            # First, determine the parent uid
            parent = {self._root: None}
            self.dfs(self._root, set(), parent)
            p_uid = parent[uid]

            # Now, determine uid of best performing child (pick any if tie)
            curr = -1
            it = None
            c_uid = set()
            for child in self.edges[uid]:
                if child == p_uid:
                    continue
                if self.vertices[child].performance > curr:
                    curr = self.vertices[child].performance
                    it = child
                c_uid.add(child)
            c_uid.remove(it)

            # Delete the vertex
            del self.vertices[uid]

            # Delete edges connected to it
            for conn in self.edges[uid]:
                self.edges[conn].remove(uid)
            del self.edges[uid]

            # Promote
            self.edges[p_uid].add(it)
            self.edges[it].add(p_uid)
            for child in c_uid:
                self.edges[it].add(child)
                self.edges[child].add(it)

        else:
            # This means vertex is a leaf
            del self.vertices[uid]
            for conn in self.edges[uid]:
                self.edges[conn].remove(uid)
            del self.edges[uid]

    def demote_vertex(self, uid: int) -> None:
        """
        Given vertex uid, demote the vertex. If vertex is a leaf, delete it.
        Otherwise, swap the vertex with its highest performing child.
        Upon a tie for highest performing child, choose first one iterated on.
        """
        if uid == self._root:
            print('Cannot demote root vertex.')
            return
        if not self.has(uid):
            print('The uid ' + str(uid) + ' is not contained in graph.')
            return
        if len(self.edges[uid]) > 1:
            # Idea is to "swap" with best performing child
            # First, determine the parent uid
            parent = {self._root: None}
            self.dfs(self._root, set(), parent)
            p_uid = parent[uid]

            # Determine uid of best performing child (pick any if tie)
            curr = -1
            it = None
            for child in self.edges[uid]:
                if child == p_uid:
                    continue
                if self.vertices[child].performance > curr:
                    curr = self.vertices[child].performance
                    it = child
            # Swap positions
            a = self.edges[uid].copy()
            b = self.edges[it].copy()
            for conn in a:
                self.edges[conn].remove(uid)
                self.edges[conn].add(it)
            for conn in b:
                self.edges[conn].remove(it)
                self.edges[conn].add(uid)
            a = self.edges[uid].copy()
            b = self.edges[it].copy()
            self.edges[uid] = b
            self.edges[it] = a
        else:
            self.delete_vertex(uid)

    def convert_to_networkx(self) -> Union[List[tuple[str, str]], str]:
        """
        Converts our tree implementation to a networkx compatible tree.
        """
        lst = []
        # when root is the only node
        root_id = self._root
        root_name = self.vertices[root_id].name

        if self.edges[root_id] == set():
            return root_name + ":" + str(root_id)
        else:
            for uid in self.edges:
                node = self.vertices[uid]
                name = node.name
                set_of_edges = self.edges[uid]
                for edge in set_of_edges:
                    edge_node = self.vertices[edge]
                    edge_name = edge_node.name
                    lst.append((name + ':' + str(uid), edge_name + ':' + str(edge)))
                    # Handle duplicates
            s = set()
            for a, b in lst:
                if (b, a) in s:
                    continue
                s.add((a, b))

            return list(s)

    def visualize_graph(self) -> None:
        """
        Visualizes current state of the employee tree.
        """
        if self._root is None:
            print("Tree does not have root")
            return
        else:
            root_id = self._root
            root_name = self.vertices[root_id].name
            root_name_id = root_name + ":" + str(root_id)
            data = self.convert_to_networkx()
            visualization.visualize(data, root_name_id)


if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod(verbose=True)
    python_ta.check_all(config={
        'max-line-length': 1000,
        # E1136, R1710 for Optional[] typing, E9998 for IO
        'disable': ['E1136', 'R1710', 'E9998'],
        'extra-imports': ['collections', 'visualization'],
        'max-nested-blocks': 4
    })
