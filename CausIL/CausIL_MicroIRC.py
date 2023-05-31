import os

from libraries.FGES.runner import  fges_runner
from libraries.FGES.knowledge import Knowledge
from python.scores import *
from python.bnutils import *
import warnings
import time

warnings.filterwarnings("ignore")



def runner(g, data, disc, n_bins, score, bl_edges = None):
    """Wrapper Runner for FGES that sets the prohibited edges into knowledge class
    
    Returns: 
        g   : result graph appended to input g
        dag : the result graph from the algo
    """

    nodes = []
    args = nx.DiGraph()
    mappers = column_mapping(data)
    data.rename(columns=mappers[0], inplace=True)
    args.add_nodes_from(list(data.columns))
    for col in data.columns:
        args.nodes[col]['type'] = 'cont'
        args.nodes[col]['num_categories'] = 'NA'


    if bl_edges:
        knowledge = Knowledge()
        for bl_edge in bl_edges:
            if bl_edge[0] in mappers[0].keys() and bl_edge[1] in mappers[0].keys():
                knowledge.set_forbidden(mappers[0][bl_edge[0]], mappers[0][bl_edge[1]])
    else:
        knowledge = None

    result = fges_runner(data, args.nodes(data=True), n_bins = n_bins, disc = disc, score = score, knowledge = knowledge)
    dag = nx.DiGraph()
    dag.add_nodes_from(args.nodes(data=True))
    dag.add_edges_from(result['graph'].edges())

    data.rename(columns = mappers[1], inplace = True)
    nx.relabel_nodes(dag, mappers[1], copy=False)

    g.add_nodes_from(dag.nodes)
    g.add_edges_from(dag.edges)

    return g, dag



def preprocess_data(data):
    updated_data = pd.DataFrame()

    mu = {}
    num_inst = []

    # for cols having instance level data, concatenate and flatten them -> pools the data
    for col in data.columns:
        if 'agg' in col:
            continue
        node_data = data[col].to_numpy()
        # 根据负载指标长度统计实例个数
        if len(num_inst) == 0:
            for i in range(len(node_data)):
                num_inst.append(len(node_data[i]))
        # 将实例的负载时序指标扁平化
        flatten = np.concatenate(node_data.tolist())
        updated_data[col] = flatten.ravel()

    # for cols having aggregate level data, repeat the data for number of instances
    for col in data.columns:
        if 'inst' in col:
            continue
        node_data = np.array(list(data[col]))

        updated_data[col] = np.repeat(node_data, num_inst)

    return updated_data

def printDGNodes(DG):
    for node in DG.nodes(data=True):
        print(node)

def printDGEdges(DG):
    for edge in DG.edges(data=True):
        print(edge)

def run_graph_discovery_metric_sum(data, dag_cg, datapath, dataset, dk, score_func):
    g = nx.DiGraph()
    service_graph = []
    fges_time = []
    edge_map = {}

    # 根据call graph，取上游服务的工作负载指标，取下游服务的延时、异常指标，构建指标集合
    # For each service, construct a graph individually and then merge them
    for i, service in enumerate(dag_cg.nodes):

        print('===============')
        print("Service: {}".format(service))
        serv_data = pd.DataFrame()

        # Get instance level data corresponding to the service only
        filtered_cols = [col for col in data.columns if (str(service).replace('_', '-', 1) in col)]
        serv_data = serv_data.append(data[filtered_cols])

        # For the child services, get aggregate level data for latency and error
        child = [n[1] for n in dag_cg.out_edges(service)]
        print("Child Services:{}".format(child))
        child = [c.replace('_', '-', 1) for c in child]
        agg_cols = []
        for col in data.columns:
            for c in child:
                if c in col:
                    agg_cols.append(col)
        serv_data = pd.concat([serv_data, data[agg_cols]], axis=1)


        # For parent services, get aggregate level data for workload (aggregate worload = total workload)
        parent = [n[0] for n in dag_cg.in_edges(service)]
        parent = [p.replace('_', '-', 1) for p in parent]
        print("Parent Services:{}".format(parent))
        agg_cols = []
        for col in data.columns:
            for p in parent:
                if p in col:
                    agg_cols.append(col)
        serv_data = pd.concat([serv_data, data[agg_cols]], axis=1)

        # Pool the data
        # 将服务的n个实例的指标放在同一列，聚集性指标每个实例复制一份
        # new_data = preprocess_data(serv_data)
        new_data = serv_data

        # Renaming is done based on the names that were present for ground truth graph
        # for example, W_0 is renamed to 0W, U_0 is renamed to 0MU, C_0 is renamed to 0CU, etc.
        rename_col = {}
        # for col in new_data.columns:
        #     if 'U' in col:
        #         rename_col[col] = col.split('_')[1] + 'MU'
        #     elif 'C' in col:
        #         rename_col[col] = col.split('_')[1] + 'CU'
        #     else:
        #         rename_col[col] = col.split('_')[1] + col.split('_')[0]
        # new_data.rename(columns=rename_col, inplace = True)


        # Use domain knowledge or not
        if dk == 'N':
            bl_edges = None
        else:
            bl_edges = list(pd.read_csv(os.path.join(datapath, dataset, 'prohibit_edges.csv'))[['edge_source', 'edge_destination']].values)

        # Run FGES
        print('Starting FGES')

        if score_func == 'L':
            st_time = time.time()
            g, dag = runner(g, new_data, None, 1, linear_gaussian_score_iid, bl_edges)
            fges_time.append(time.time() - st_time)
        elif score_func == 'P2':
            st_time = time.time()
            g, dag = runner(g, new_data, None, 1, polynomial_2_gaussian_score_iid, bl_edges)
            fges_time.append(time.time() - st_time)
        elif score_func == 'P3':
            st_time = time.time()
            g, dag = runner(g, new_data, None, 1, polynomial_3_gaussian_score_iid, bl_edges)
            fges_time.append(time.time() - st_time)

        print('Finished FGES')
        for edge in dag.edges():
            edge_map.setdefault(str(edge[0]) + str(edge[1]), 0)
            edge_map[str(edge[0]) + str(edge[1])] += 1

        service_graph.append(dag)
        print('\n')

    return g, edge_map, service_graph, fges_time

def set_weight_by_FI_correlation_(graph: nx.DiGraph, data, alpha, edge_map):
    agg = np.mean
    for u, v in graph.edges():
        try:
            corr_map = int(edge_map[str(u) + str(v)]) * alpha
        except:
            corr_map = 0
        u_metrics = data[u].values
        v_metrics = data[v].values
        # corr = np.corrcoef(np.concatenate([u_metrics, v_metrics], axis=0), rowvar=True)[:len(u_metrics), -len(v_metrics):]
        corr = np.corrcoef(np.concatenate([u_metrics, v_metrics], axis=0), rowvar=True)
        np.nan_to_num(corr, copy=False)
        corr += corr_map
        graph[u][v]["weight"] = agg(min(1, np.abs(corr)))
    return graph

def correlation_enhance(g, edge_map, service_graph, fges_time, data):
    attri_graph = nx.DiGraph()
    alpha = 0.01
    # set_weight_by_FI_correlation_(g, data, alpha, edge_map)
    for edge in g.edges():
        attri_graph.add_edge(edge[0], edge[1], weight=min(1, abs(data[edge[0]].corr(data[edge[1]])) + int(edge_map[str(edge[0]) + str(edge[1])]) * alpha))
    printDGEdges(attri_graph)
    return attri_graph, service_graph, fges_time


def personalization_enhance(graph, edge_map, service_graph, fges_time, data):
    attri_graph = nx.DiGraph()
    alpha = 0.01
    for edge in graph.edges():
        attri_graph.add_edge(edge[0], edge[1], weight=abs(data[edge[0]].corr(data[edge[1]])))
    # set_weight_by_FI_correlation_(graph, data, alpha)
    personalization = {}
    for n in graph.nodes():
        inc_count = 0
        dec_count = 0
        inc_total = 0
        dec_total = 0
        for s, t, w in attri_graph.in_edges(n, data=True):
            inc_total += w['weight']
            inc_total += edge_map[s+t] * alpha
            inc_count += 1
        for s, t, w in attri_graph.out_edges(n, data=True):
            dec_total += w['weight']
            dec_total += edge_map[s+t] * alpha
            dec_count += 1
        if inc_count > 0 and dec_count > 0:
            personalization[n] = inc_total / inc_count - dec_total / dec_count
        elif inc_count > 0:
            personalization[n] = inc_total / inc_count
        elif dec_count > 0:
            personalization[n] = -dec_total / dec_count
        else:
            personalization[n] = 0
    return attri_graph, service_graph, fges_time, personalization

def run_graph_discovery_metric_sum_corr_enhance(data, dag_cg, datapath, dataset, dk, score_func):
    g, edge_map, service_graph, fges_time = run_graph_discovery_metric_sum(data, dag_cg, datapath, dataset, dk, score_func)
    return correlation_enhance(g, edge_map, service_graph, fges_time, data)

def run_graph_discovery_metric_sum_personalization_enhance(data, dag_cg, datapath, dataset, dk, score_func):
    g, edge_map, service_graph, fges_time = run_graph_discovery_metric_sum(data, dag_cg, datapath, dataset, dk, score_func)
    return personalization_enhance(g, edge_map, service_graph, fges_time, data)


# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser(description='Run CausIL')
#
#     # parser.add_argument('-D', '--dataset', required=True, help='Dataset type (synthetic/semi_synthetic)')
#     parser.add_argument('-D', '--dataset', default='synthetic', help='Dataset type (synthetic/semi_synthetic)')
#     # parser.add_argument('-S', '--num_services', type=int, required=True, help='Numer of Services in the dataset (10, 20, etc.)')
#     parser.add_argument('-S', '--num_services', type=int, default=20, help='Numer of Services in the dataset (10, 20, etc.)')
#     parser.add_argument('-G', '--graph_number', type=int, default=0, help='Graph Instance in the particular dataset [default: 0]')
#     parser.add_argument('--dk', default='Y', help='To use domain knowledge or not (Y/N) [default: Y]')
#     parser.add_argument('--score_func', default='P2', help='Which score function to use (L: linear, P2: polynomial of degree 2, P3: polynomial of degree 3) [default: P2]')
#
#     args = parser.parse_args()
#
#     if args.dataset != "synthetic" and args.dataset != "semi_synthetic":
#         print("Incorrect Dataset provided!!...")
#         print("======= EXIT ===========")
#         exit()
#
#
#     # Data set directory to use
#     datapath = f'Data/{args.num_services}_services'
#     dataset = f'{args.dataset}/Graph{args.graph_number}'
#
#     data, dag_gt, dag_cg = read_data(datapath, dataset, args.graph_number)
#
#     graph, service_graph, total_time = run_graph_discovery(data, dag_cg, datapath, dataset, args.dk, args.score_func)
#
#     print(f"Total Time of Computation: {np.sum(total_time)}")
#
#     compute_stats(graph, dag_gt)
