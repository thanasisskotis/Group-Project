from typing import List, Tuple, Set

class Edge:
    def __init__(self, u: int, v: int, weight: int):
        self.u = u
        self.v = v
        self.weight = weight
    
    def __eq__(self, other):
        return (self.u == other.u and self.v == other.v and self.weight == other.weight) or \
               (self.u == other.v and self.v == other.u and self.weight == other.weight)
    
    def __repr__(self):
        return f"({self.u}, {self.v}, {self.weight})"

def update_mst_linear(V: int, graph_edges: List[Edge], mst_edges: List[Edge], 
                      u: int, v: int, new_weight: int) -> List[Edge]:
    # Find the edge in the graph and update its weight
    old_weight = None
    for e in graph_edges:
        if (e.u == u and e.v == v) or (e.u == v and e.v == u):
            old_weight = e.weight
            e.weight = new_weight
            break
    
    if old_weight is None:
        raise ValueError(f"Edge ({u}, {v}) not found in the graph")
    
    # Check if the edge is in the MST
    edge_in_mst = None
    for i, e in enumerate(mst_edges):
        if (e.u == u and e.v == v) or (e.u == v and e.v == u):
            edge_in_mst = e
            mst_edges[i].weight = new_weight
            break
    
    # Case 1: Weight increase for an edge in MST
    if edge_in_mst and new_weight > old_weight:
        return handle_weight_increase(V, graph_edges, mst_edges, edge_in_mst)
    
    # Case 2: Weight decrease for an edge not in MST
    elif not edge_in_mst and new_weight < old_weight:
        return handle_weight_decrease(V, graph_edges, mst_edges, u, v, new_weight)
    
    # No change needed in other cases
    return mst_edges

def handle_weight_increase(V: int, graph_edges: List[Edge], mst_edges: List[Edge], 
                           edge_to_remove: Edge) -> List[Edge]:
    # Step 1: Build MST adjacency list (without edge_to_remove)
    adj_list = [[] for _ in range(V)]
    for e in mst_edges:
        if e != edge_to_remove:
            adj_list[e.u].append(e.v)
            adj_list[e.v].append(e.u)
    
    # Step 2: Use BFS/DFS to find one component starting from edge_to_remove.u
    visited = [False] * V
    component = set()
    
    def dfs(node):
        visited[node] = True
        component.add(node)
        for neighbor in adj_list[node]:
            if not visited[neighbor]:
                dfs(neighbor)
    
    dfs(edge_to_remove.u)

    # If v is still reachable, the MST remains connected — return early
    if edge_to_remove.v in component:
        return mst_edges

    # Step 3: Find minimum weight edge connecting the two components
    min_edge = None
    for e in graph_edges:
        u_in = e.u in component
        v_in = e.v in component
        if u_in != v_in:  # One in component, one out
            if min_edge is None or e.weight < min_edge.weight:
                min_edge = e

    # Step 4: Replace the removed edge with the new one
    new_mst = [e for e in mst_edges if e != edge_to_remove]
    if min_edge:
        new_mst.append(min_edge)

    return new_mst


def handle_weight_decrease(V: int, graph_edges: List[Edge], mst_edges: List[Edge], 
                           u: int, v: int, new_weight: int) -> List[Edge]:
    # Build adjacency list from current MST
    adj_list = [[] for _ in range(V)]
    for e in mst_edges:
        adj_list[e.u].append((e.v, e))
        adj_list[e.v].append((e.u, e))
    
    # Find the path from u to v in the MST
    parent = [-1] * V
    edge_to_parent = [None] * V
    visited = [False] * V
    
    def dfs(node, target):
        if node == target:
            return True
        visited[node] = True
        for neighbor, edge in adj_list[node]:
            if not visited[neighbor]:
                parent[neighbor] = node
                edge_to_parent[neighbor] = edge
                if dfs(neighbor, target):
                    return True
        return False
    
    if not dfs(u, v):
        return mst_edges
    
    # Find the edge with the maximum weight in the path from u to v
    curr = v
    max_edge = None
    while curr != u:
        edge = edge_to_parent[curr]
        if max_edge is None or edge.weight > max_edge.weight:
            max_edge = edge
        curr = parent[curr]
    
    # If the new edge has smaller weight, replace the max weight edge
    if max_edge and max_edge.weight > new_weight:
        result = [e for e in mst_edges if e != max_edge]
        result.append(Edge(u, v, new_weight))
        return result
    
    return mst_edges


def build_mst_kruskal(V: int, edges: List[Edge]) -> List[Edge]:
    parent = list(range(V))
    rank = [0] * V
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            if rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            elif rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
    
    mst = []
    edges_sorted = sorted(edges, key=lambda e: e.weight)
    for e in edges_sorted:
        if find(e.u) != find(e.v):
            mst.append(e)
            union(e.u, e.v)
    
    return mst

def test_mst_update_case_2():
    print("=== MST Update Algorithm Test: Case 2 ===")
    V = 7
    edges = [
        Edge(0, 1, 2), Edge(0, 2, 3), Edge(1, 2, 1),
        Edge(1, 3, 4), Edge(2, 4, 5), Edge(3, 4, 6),
        Edge(3, 5, 7), Edge(4, 5, 8), Edge(5, 6, 2), Edge(4, 6, 9)
    ]

    print("Original Graph:")
    for e in edges:
        print(f"{e.u} - {e.v} (Weight: {e.weight})")

    # Build initial MST
    mst = build_mst_kruskal(V, edges.copy())
    print("\nInitial MST:")
    for e in mst:
        print(f"{e.u} - {e.v} (Weight: {e.weight})")

    # Test Case 1: Increase weight of edge (1-2) from 1 to 10 (was part of MST)
    print("\nTest Case: Increase weight of MST edge (1-2) from 1 to 10")
    mst = update_mst_linear(V, edges, mst, 1, 2, 10)
    print("Updated MST:")
    for e in mst:
        print(f"{e.u} - {e.v} (Weight: {e.weight})")

    # Test Case 2: Decrease non-MST edge (4-6) from 9 to 1
    print("\nTest Case: Decrease non-MST edge (4-6) from 9 to 1")
    mst = update_mst_linear(V, edges, mst, 4, 6, 1)
    print("Updated MST:")
    for e in mst:
        print(f"{e.u} - {e.v} (Weight: {e.weight})")


if __name__ == "__main__":
    test_mst_update_case_2()
