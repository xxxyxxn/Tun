import time
import util
import networkx as nx
from itertools import islice
import numpy as np
import torch
from model_gnn import GNNRoutePlanner
from collections import defaultdict
import util

def convert_to_gnn_input(prob_info):
    N = prob_info['N']
    x = torch.zeros((N, 4))
    for i in range(N):
        x[i][0] = 1 if i == 0 else 0  # gate 여부
        x[i][1] = i / N               # 정규화된 index
        x[i][2] = (i % 3) / 3         # dummy feature
        x[i][3] = 1                   # bias term
    edge_index = torch.tensor(prob_info['E'], dtype=torch.long).t().contiguous()
    return x, edge_index


def algorithm(prob_info, timelimit=60):
    import time
    import torch
    import networkx as nx
    from itertools import islice
    from collections import defaultdict

    start_time = time.time()

    N = prob_info['N']
    E = prob_info['E']
    K = prob_info['K']
    P = prob_info['P']
    F = prob_info['F']
    LB = prob_info['LB']

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(E)

    node_allocations = [-1] * N
    demand_loaded_at = [-1] * len(K)
    solution = {p: [] for p in range(P)}

    max_k_paths = 5
    k_shortest_paths = [[] for _ in range(N)]
    for i in range(1, N):
        paths = list(islice(nx.shortest_simple_paths(G, 0, i), max_k_paths))
        k_shortest_paths[i] = paths

    x, edge_index = convert_to_gnn_input(prob_info)
    model = GNNRoutePlanner(x.shape[1], 64, 1, activation="leaky_relu")
    model.load_state_dict(torch.load("checkpoint/model.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x, edge_index).squeeze()).numpy()

    def is_path_clear(path, node_allocations, exclude=None):
        for i in path:
            if i == 0:
                continue
            if node_allocations[i] != -1 and i != exclude:
                return False
        return True

    def find_reachable_nodes(G, node_allocations):
        reachable, _ = util.bfs(G, node_allocations)
        return [n for n in reachable if node_allocations[n] == -1 and n != 0]

    def get_best_path_to_node(G, node_allocations, k_shortest_paths, target_node):
        best = None
        best_blockers = float('inf')
        for path in k_shortest_paths[target_node]:
            blockers = [i for i in path[:-1] if node_allocations[i] != -1]
            if len(blockers) < best_blockers:
                best_blockers = len(blockers)
                best = (blockers, path)
        return best

    def recursive_relocate(b, node_allocations, visited):
        if b in visited:
            return None, None
        visited.add(b)
        relocation_target_nodes = find_reachable_nodes(G, node_allocations)
        relocation_target_nodes = [t for t in relocation_target_nodes if t != b]
        for t in relocation_target_nodes:
            dist, prev = util.dijkstra(G, node_allocations, start=b)
            path_b_to_t = util.path_backtracking(prev, b, t)
            if len(path_b_to_t) < 2:
                continue
            if is_path_clear(path_b_to_t[1:], node_allocations):
                return path_b_to_t, t
            inner_blockers = [i for i in path_b_to_t[1:] if node_allocations[i] != -1]
            for ib in inner_blockers:
                relocate_result = recursive_relocate(ib, node_allocations, visited)
                if relocate_result != (None, None):
                    retry_result = recursive_relocate(b, node_allocations, visited)
                    if retry_result != (None, None):
                        return retry_result
        dist, prev = util.dijkstra(G, node_allocations, start=b)
        path_b_to_0 = util.path_backtracking(prev, b, 0)
        if len(path_b_to_0) >= 2 and is_path_clear(path_b_to_0[1:], node_allocations):
            return path_b_to_0, 0
        return None, None

    demand_priority = sorted(
        [(k, dest) for k, ((origin, dest), qty) in enumerate(K)],
        key=lambda x: -x[1]
    )
    candidate_nodes_by_score = sorted(
        [n for n in range(N) if n != 0],
        key=lambda n: scores[n],
        reverse=True
    )
    global_load_plan = defaultdict(list)
    for k, dest in demand_priority:
        origin, _ = K[k][0]
        node_candidates = candidate_nodes_by_score.copy()
        global_load_plan[origin].append((k, node_candidates))

    def run_unloading(p, node_allocations, demand_loaded_at):
        route_list = []
        rehandling_demands = []
        for k, ((origin, dest), quantity) in enumerate(K):
            if dest != p:
                continue
            unloading_nodes = [n for n in range(N) if node_allocations[n] == k]
            for n in unloading_nodes:
                if node_allocations[n] != k:
                    continue
                blockers, path = get_best_path_to_node(G, node_allocations, k_shortest_paths, n)
                if path is None or len(path) < 2:
                    continue
                for b in blockers:
                    relocate_result = recursive_relocate(b, node_allocations, set())
                    if relocate_result or relocate_result == (None, None):
                        continue
                    move_path, target = relocate_result
                    route_list.append((move_path, node_allocations[b]))
                    if target == 0:
                        rehandling_demands.append(node_allocations[b])
                        demand_loaded_at[node_allocations[b]] = -1
                    node_allocations[target] = node_allocations[b]
                    node_allocations[b] = -1
                route_list.append((path[::-1], k))
                node_allocations[n] = -1
                demand_loaded_at[k] = -1
        return route_list, rehandling_demands

    def run_loading(p, node_allocations, demand_loaded_at, scores, rehandling_demands, global_load_plan):
        route_list = []
        candidate_nodes = find_reachable_nodes(G, node_allocations)
        plan = global_load_plan.get(p, []) + [(k, candidate_nodes.copy()) for k in rehandling_demands if K[k][0][0] == p]
        for k, node_candidates in plan:
            assigned = False
            for n in node_candidates:
                dist, prev = util.dijkstra(G, node_allocations)
                path = util.path_backtracking(prev, 0, n)
                if len(path) < 2:
                    continue
                if is_path_clear(path[1:], node_allocations):
                    node_allocations[n] = k
                    demand_loaded_at[k] = p
                    route_list.append((path, k))
                    assigned = True
                    if n in candidate_nodes:
                        candidate_nodes.remove(n)
                    break
            if not assigned:
                removable = sorted(
                    [i for i in range(N) if node_allocations[i] != -1 and i != 0],
                    key=lambda x: K[node_allocations[x]][0][1]
                )
                for b in removable:
                    relocate_result = recursive_relocate(b, node_allocations, set())
                    if relocate_result == (None, None):
                        continue
                    move_path, target = relocate_result
                    route_list.append((move_path, node_allocations[b]))
                    if target == 0:
                        demand_loaded_at[node_allocations[b]] = -1
                    node_allocations[target] = node_allocations[b]
                    node_allocations[b] = -1
                    candidate_nodes.append(b)
                    break
        return route_list

    for p in range(P):
        print(f"\n[Port {p}] =======================")
        if p > 0:
            route_unload, rehandled = run_unloading(p, node_allocations, demand_loaded_at)
            print(f"[Port {p}] Unloaded {len(route_unload)} routes")
            solution[p].extend(route_unload)
        else:
            rehandled = []
        if p < P - 1:
            route_load = run_loading(p, node_allocations, demand_loaded_at, scores, rehandled, global_load_plan)
            print(f"[Port {p}] Loaded {len(route_load)} routes")
            solution[p].extend(route_load)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n[Done] Total time: {total_time:.2f} seconds")
    return solution








if __name__ == "__main__":
    # You can run this file to test your algorithm from terminal.

    import json
    import os
    import sys
    import jsbeautifier


    def numpy_to_python(obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)  
        if isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        raise TypeError(f"Type {type(obj)} not serializable")
    
    
    # Arguments list should be problem_name, problem_file, timelimit (in seconds)
    if len(sys.argv) == 4:
        prob_name = sys.argv[1]
        prob_file = sys.argv[2]
        timelimit = int(sys.argv[3])

        with open(prob_file, 'r') as f:
            prob_info = json.load(f)

        exception = None
        solution = None

        try:

            alg_start_time = time.time()

            # Run algorithm!
            solution = algorithm(prob_info, timelimit)

            alg_end_time = time.time()


            checked_solution = util.check_feasibility(prob_info, solution)

            checked_solution['time'] = alg_end_time - alg_start_time
            checked_solution['timelimit_exception'] = (alg_end_time - alg_start_time) > timelimit + 2 # allowing additional 2 second!
            checked_solution['exception'] = exception

            checked_solution['prob_name'] = prob_name
            checked_solution['prob_file'] = prob_file


            with open('results.json', 'w') as f:
                opts = jsbeautifier.default_options()
                opts.indent_size = 2
                f.write(jsbeautifier.beautify(json.dumps(checked_solution, default=numpy_to_python), opts))
                print(f'Results are saved as file results.json')
                
            sys.exit(0)

        except Exception as e:
            print(f"Exception: {repr(e)}")
            sys.exit(1)

    else:
        print("Usage: python myalgorithm.py <problem_name> <problem_file> <timelimit_in_seconds>")
        sys.exit(2)

