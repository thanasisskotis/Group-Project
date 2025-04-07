import heapq

INF = int(1e9)

def find_shortest_directed_cycle(n, adj):
    min_cycle = INF

    for u in range(n):
        dist = [INF] * n
        dist[u] = 0
        pq = [(0, u)]

        while pq:
            d, v = heapq.heappop(pq)
            if d > dist[v]:
                continue
            for to, w in adj[v]:
                if dist[to] > d + w:
                    dist[to] = d + w
                    heapq.heappush(pq, (dist[to], to))

        for v in range(n):
            for to, w in adj[v]:
                if to == u and dist[v] != INF:
                    min_cycle = min(min_cycle, dist[v] + w)

    return -1 if min_cycle == INF else min_cycle

if __name__ == "__main__":
    # Example graph: 4 nodes, 5 edges
    n = 4
    adj = [[] for _ in range(n)]

    # Add edges: (from, to, weight)
    adj[0].append((1, 2))
    adj[1].append((2, 3))
    adj[2].append((0, 4))  # Cycle: 0 → 1 → 2 → 0 with weight 9
    adj[1].append((3, 2))
    adj[3].append((1, 1))  # Cycle: 1 → 3 → 1 with weight 3

    result = find_shortest_directed_cycle(n, adj)
    if result == -1:
        print("No directed cycle found.")
    else:
        print("Length of shortest directed cycle:", result)
