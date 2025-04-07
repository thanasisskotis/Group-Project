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
    """
    Update the MST when the weight of edge (u,v) changes.
    
    Args:
        V: Number of vertices
        graph_edges: All edges in the graph
        mst_edges: Edges in the current MST
        u, v: The vertices of the edge whose weight is changing
        new_weight: The new weight of edge (u,v)
        
    Returns:
        Updated MST edges
    """
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
    """
    Handle the case when the weight of an edge in the MST increases.
    
    Args:
        V: Number of vertices
        graph_edges: All edges in the graph
        mst_edges: Edges in the current MST
        edge_to_remove: The edge whose weight increased
        
    Returns:
        Updated MST edges
    """
    # Create adjacency list from MST (without the edge to remove)
    adj_list = [[] for _ in range(V)]
    for e in mst_edges:
        if e != edge_to_remove:
            adj_list[e.u].append(e.v)
            adj_list[e.v].append(e.u)
    
    # Find connected components using BFS
    visited = [False] * V
    component = [0] * V
    
    def bfs(start, comp_id):
        queue = [start]
        visited[start] = True
        component[start] = comp_id
        
        while queue:
            node = queue.pop(0)
            for neighbor in adj_list[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    component[neighbor] = comp_id
                    queue.append(neighbor)
    
    # Run BFS to mark components
    comp_id = 1
    for i in range(V):
        if not visited[i]:
            bfs(i, comp_id)
            comp_id += 1
    
    # If removing the edge doesn't disconnect the graph, no change needed
    if component[edge_to_remove.u] == component[edge_to_remove.v]:
        return mst_edges
    
    # Find the minimum weight edge that connects the two components
    min_edge = None
    for e in graph_edges:
        if e != edge_to_remove and component[e.u] != component[e.v]:
            if component[e.u] == component[edge_to_remove.u] and component[e.v] == component[edge_to_remove.v] or \
               component[e.u] == component[edge_to_remove.v] and component[e.v] == component[edge_to_remove.u]:
                if min_edge is None or e.weight < min_edge.weight:
                    min_edge = e
    
    # Remove the edge with increased weight and add the replacement edge
    result = [e for e in mst_edges if e != edge_to_remove]
    if min_edge:
        result.append(min_edge)
    
    return result

def handle_weight_decrease(V: int, graph_edges: List[Edge], mst_edges: List[Edge], 
                           u: int, v: int, new_weight: int) -> List[Edge]:
    """
    Handle the case when the weight of an edge not in the MST decreases.
    
    Args:
        V: Number of vertices
        graph_edges: All edges in the graph
        mst_edges: Edges in the current MST
        u, v: The vertices of the edge whose weight decreased
        new_weight: The new weight of edge (u,v)
        
    Returns:
        Updated MST edges
    """
    # Create adjacency list from MST
    adj_list = [[] for _ in range(V)]
    edge_indices = {}
    
    for i, e in enumerate(mst_edges):
        adj_list[e.u].append(e.v)
        adj_list[e.v].append(e.u)
        edge_indices[(e.u, e.v)] = i
        edge_indices[(e.v, e.u)] = i
    
    # Find the path from u to v in the MST
    path = []
    visited = [False] * V
    parent = [-1] * V
    
    def find_path(start, end):
        queue = [start]
        visited[start] = True
        
        while queue:
            node = queue.pop(0)
            if node == end:
                # Reconstruct the path
                curr = end
                while curr != start:
                    prev = parent[curr]
                    if prev < curr:
                        path.append(edge_indices[(prev, curr)])
                    else:
                        path.append(edge_indices[(curr, prev)])
                    curr = prev
                return True
            
            for neighbor in adj_list[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = node
                    queue.append(neighbor)
        
        return False
    
    # If there's no path from u to v, MST can't be improved
    if not find_path(u, v):
        return mst_edges
    
    # Find the heaviest edge on the path
    max_weight_edge_idx = -1
    max_weight = -1
    
    for idx in path:
        if mst_edges[idx].weight > max_weight:
            max_weight = mst_edges[idx].weight
            max_weight_edge_idx = idx
    
    # If the new edge has lower weight than the heaviest edge on the path, replace it
    if max_weight > new_weight:
        result = [e for i, e in enumerate(mst_edges) if i != max_weight_edge_idx]
        result.append(Edge(u, v, new_weight))
        return result
    
    return mst_edges

def build_mst_kruskal(V: int, edges: List[Edge]) -> List[Edge]:
    """
    Build an MST using Kruskal's algorithm.
    Used for initial MST creation and testing.
    """
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

def test_mst_update():
    print("=== MST Update Algorithm Test ===")
    V = 5
    edges = [
        Edge(0, 1, 2), Edge(0, 3, 6), Edge(1, 2, 3),
        Edge(1, 3, 8), Edge(1, 4, 5), Edge(2, 4, 7), Edge(3, 4, 9)
    ]
    
    print("Original Graph:")
    for e in edges:
        print(f"{e.u} - {e.v} (Weight: {e.weight})")
    
    # Build initial MST
    mst = build_mst_kruskal(V, edges.copy())
    print("\nInitial MST:")
    for e in mst:
        print(f"{e.u} - {e.v} (Weight: {e.weight})")
    
    # Test Case 1: Weight increase in MST
    print("\nTest Case 1: Weight increase in MST (0-1: 2 -> 10)")
    updated_mst = update_mst_linear(V, edges.copy(), mst.copy(), 0, 1, 10)
    print("Updated MST:")
    for e in updated_mst:
        print(f"{e.u} - {e.v} (Weight: {e.weight})")
    
    # Reset for test case 2
    edges = [
        Edge(0, 1, 2), Edge(0, 3, 6), Edge(1, 2, 3),
        Edge(1, 3, 8 ), Edge(1, 4, 5), Edge(2, 4, 7), Edge(3, 4, 9)
    ]
    mst = build_mst_kruskal(V, edges.copy())
    
    # Test Case 2: Weight decrease outside MST
    print("\nTest Case 2: Weight decrease outside MST (3-4: 9 -> 1)")
    updated_mst = update_mst_linear(V, edges.copy(), mst.copy(), 3, 4, 1)
    print("Updated MST:")
    for e in updated_mst:
        print(f"{e.u} - {e.v} (Weight: {e.weight})")

if __name__ == "__main__":
    test_mst_update()
