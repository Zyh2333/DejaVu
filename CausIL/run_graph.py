from CausIL.CausIL_MicroIRC import printDGEdges
import networkx as nx

def run_sum_personalization(graph, personalization):
    printDGEdges(graph)
    anomaly_score = nx.pagerank(
        graph, alpha=0.85, max_iter=100000, personalization=personalization)
    anomaly_score_map = {}
    for k in anomaly_score:
        anomaly_score_map.setdefault(k[:k.index('&')], 0)
        anomaly_score_map[k[:k.index('&')]] += anomaly_score[k]
    anomaly_score_map = sorted(anomaly_score_map.items(), key=lambda x: x[1], reverse=True)
    return anomaly_score_map

def run_sum(graph):
    personalization = {}
    for n in graph.nodes():
        inc_count = 0
        dec_count = 0
        inc_total = 0
        dec_total = 0
        for s, t, data in graph.in_edges(n, data=True):
            inc_total += data['weight']
            inc_count += 1
        for s, t, data in graph.out_edges(n, data=True):
            dec_total += data['weight']
            dec_count += 1
        if inc_count > 0 and dec_count > 0:
            personalization[n] = inc_total / inc_count - dec_total / dec_count
        elif inc_count > 0:
            personalization[n] = inc_total / inc_count
        elif dec_count > 0:
            personalization[n] = -dec_total / dec_count
        else:
            personalization[n] = 0
    return run_sum_personalization(graph, personalization)

def count(root_cause, failure_instance_scores):
    os_count = 0
    for i in range(len(failure_instance_scores)):
        if 'os' in failure_instance_scores[i][0] or 'db' in failure_instance_scores[i][0]:
            os_count += 1
            continue
        if root_cause in failure_instance_scores[i][0]:
            return i + 1 - os_count
    return 0


if __name__ == '__main__':
    graph = nx.read_gpickle('graph-graph_discovery_metric_sum_corr_enhance-docker_003-1684206091.112391.gpickle')
    anomaly_score_map = run_sum(graph)
    print(count('docker-003', anomaly_score_map))
    for a in anomaly_score_map:
        print(a)
