#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhuyuhan2333
"""

import pandas as pd
import numpy as np
import networkx as nx
import os

from CausIL.formalize import formalize
from CausIL.parse_yaml_graph_config import parse_graph
from CausIL.time_diy import Time
from CausIL.CausIL_MicroIRC import run_graph_discovery_metric_sum_corr_enhance
from CausIL.CausIL_MicroIRC import run_graph_discovery_metric_sum_personalization_enhance
from _datetime import datetime
from CausIL.run_graph import run_sum
from CausIL.run_graph import run_sum_personalization

import warnings
warnings.filterwarnings('ignore')


def dfTimelimit(df, begin_timestamp, end_timestamp):
    begin_index = 0
    end_index = 1
    for index, row in df.iterrows():
        if row['timestamp'] >= begin_timestamp:
            begin_index = index
            break
    for index, row in df.iterrows():
        if index > begin_index and row['timestamp'] >= end_timestamp:
            end_index = index
            break
    df = df.loc[begin_index:end_index]
    return df

# Get the instance baseline
def getInstanceBaseline(svc, instance, baseline_df, faults_name, begin_timestamp, end_timestamp):
    filename = faults_name + '/' + instance + '.csv'
    df = pd.read_csv(filename)
    # Fetch sliding window
    df = dfTimelimit(df, begin_timestamp, end_timestamp)

    total = 0
    max = 0
    max_col = df.columns[3]
    for column in df.columns[2:-3]:
        piece = abs((pd.Series(formalize(baseline_df[svc].fillna(0)).squeeze())).corr(pd.Series(formalize(df[column].fillna(0)).squeeze())))
        if piece > max:
            max = piece
            max_col = column
    return df[max_col]

# the correlation between the instance and its service
def corrSvcAndInstances(svc, instance, baseline_df, faults_name, begin_timestamp, end_timestamp):
    filename = faults_name + '/' + instance + '.csv'
    df = pd.read_csv(filename)
    df = dfTimelimit(df, begin_timestamp, end_timestamp)
    total = 0
    max = 0
    for column in df.columns[2:-3]:
        piece = abs((pd.Series(formalize(baseline_df[svc].fillna(0)).squeeze())).corr(pd.Series(formalize(df[column].fillna(0)).squeeze())))
        if piece > max:
            max = piece
    return max

# the correlation between the instance and its node
def corrNodeAndInstances(instance, faults_name, begin_timestamp, end_timestamp):
    filename = faults_name + '/' + instance + '.csv'
    df = pd.read_csv(filename)
    df = dfTimelimit(df, begin_timestamp, end_timestamp)
    total = 0
    max = 0.01
    for column in df.columns[2:-3]:
        for node_column in df.columns[-3:]:
            piece = abs((pd.Series(formalize(df[column].fillna(0)).squeeze())).corr(pd.Series(formalize(df[node_column].fillna(0)).squeeze())))
            if piece > max:
                max = piece
    return max

def instance_personalization(svc, anomaly_graph, baseline_df, faults_name, instance, begin_timestamp, end_timestamp):
    filename = faults_name + '/' + instance + '.csv'
    df = pd.read_csv(filename)
    df = dfTimelimit(df, begin_timestamp, end_timestamp)
    ctn_cols = df.columns[2:-3]
    max_corr = 0
    metric = ctn_cols[0]
    total = 0
    for col in ctn_cols:
        temp = abs((pd.Series(formalize(baseline_df[svc].fillna(0)).squeeze())).corr(pd.Series(formalize(df[col].fillna(0)).squeeze())))
        # total += temp
        if temp > max_corr:
            max_corr = temp
            metric = col

    # The total value of statistical services
    edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
        num = num + 1
        edges_weight_avg = edges_weight_avg + data['weight'] 

    svc_instance_data = 0.01
    for u, v, data in anomaly_graph.out_edges(svc, data=True):
        if v == instance:
            svc_instance_data = data['weight']

    # The total value of svc to instance conversion
    edges_weight_avg = edges_weight_avg * svc_instance_data / num + max_corr
    personalization = edges_weight_avg

    return personalization, max_corr

def svc_personalization(svc, anomaly_graph, baseline_df, faults_name, begin_timestamp, end_timestamp):
    # The total value of statistical svc
    edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
        num = num + 1
        edges_weight_avg = edges_weight_avg + data['weight']

    # The total value of svc to instance conversion
    edges_weight_avg = edges_weight_avg / num

    personalization = edges_weight_avg

    return personalization

def node_personalization(node, anomaly_graph, baseline_df, faults_name, begin_timestamp, end_timestamp):
    # Count the total value of instances on the node
    edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(node, data=True):
        num = num + 1
        edges_weight_avg = edges_weight_avg + data['weight'] 

    # Total value of svc to instance conversion
    edges_weight_avg = edges_weight_avg / num
    personalization = edges_weight_avg
    return personalization

# draw anomaly subgraph and execute personalized randow walk
def anomaly_subgraph(DG, anomalies, latency_df, faults_name, alpha, svc_instances_map, instance_svc_map, begin_timestamp, end_timestamp, anomalie_instances, root_cause_level, root_cause, call_set):
    # Get all the svc nodes and instance nodes associated with the exception detection
    edges = []
    nodes = []
    edge_walk = []
    baseline_df = pd.DataFrame()
    edge_df = {}
    # Anomaly source collection
    anomaly_source = []
    source_alpha = 0.2
    # Draw anomaly subgraphs from anomaly nodes
    for anomaly in anomalies:
        edge = anomaly.split('_')
        edge[1] = edge[1][:len(edge[1])-4]
        if edge not in edge_walk:
            edge_walk.append(edge)
        edges.append(tuple(edge))

        svc = edge[1]
        if svc == 'redis-cart' or svc == 'unknown':
            continue
        nodes.append(svc)

        # add anomaly sources
        source = edge[0]
        nodes.append(source)
        anomaly_source.append(source)
        baseline_df[source] = latency_df[anomaly]

        # add the edge[0], i.e, instance，latency impact due to caller instance
        for u, v, data in DG.out_edges(source, data=True):
            if u in v:
                nodes.append(v)
                if v in anomalie_instances:
                    edges.append(tuple([u, v]))
                baseline_df[v] = getInstanceBaseline(u, v, baseline_df, faults_name, begin_timestamp, end_timestamp)

        # Latency as a benchmark for subsequent comparison with its metrics
        baseline_df[svc] = latency_df[anomaly]
        edge_df[svc] = anomaly
        # Add the called party instance node to the node to be processed in the subgraph
        for u, v, data in DG.out_edges(svc, data=True):
            if u in v:
                nodes.append(v)
                if v in anomalie_instances:
                    edges.append(tuple([u, v]))
                baseline_df[v] = getInstanceBaseline(u, v, baseline_df, faults_name, begin_timestamp, begin_timestamp)
                edge_df[v] = anomaly
    # Benchmarking of abnormal metrics
    baseline_df = baseline_df.fillna(0)
    nodes = set(nodes)
    # Modify anomaly node svc, edge name
    nodes = cutSvcNameForAnomalyNodes(nodes)

    # draw anomaly subgraph
    anomaly_graph = nx.DiGraph()
    for node in nodes:
        # Skip if an instance node
        if DG.nodes[node]['type'] == 'instance' or node == 'unknown':
            continue
        # Set incoming edge weights
        for u, v, data in DG.in_edges(node, data=True):
            edge = (u, v)
            # If it is an abnormal edge, assign alpha directly
            if edge in edges:
                data = alpha
            # If it is an instance edge, skip it first and assign it synchronously by its svc assignment
            elif "-" in node:
                continue
            else:
                normal_edge = u + '_' + v + '&p50'
                data = abs(baseline_df[v].corr(latency_df[normal_edge]))
            data = 0 if np.isnan(data) else data
            data = round(data, 3)
            anomaly_graph.add_edge(u, v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']

        # Set out edge weights
        # u is the anomaly node
        for u, v, data in DG.out_edges(node, data=True):
            edge = (u, v)
            if edge in edges:
                data = alpha
                if DG.nodes[v]['type'] == 'instance' :
                    anomaly_graph.add_edge(v, 'node', weight=corrNodeAndInstances(v, faults_name, begin_timestamp, end_timestamp))
                    anomaly_graph.nodes['node']['type'] = 'host'
            else:
                if DG.nodes[v]['type'] == 'instance' :
                    # Assign weights based on similarity of metrics
                    data = corrSvcAndInstances(u, v, baseline_df, faults_name, begin_timestamp, end_timestamp)
                    anomaly_graph.add_edge(v, 'node', weight=corrNodeAndInstances(v, faults_name, begin_timestamp, end_timestamp))
                    anomaly_graph.nodes['node']['type'] = 'host'
                else:
                    if 'redis' in v:
                        continue
                    normal_edge = u + '_' + v
                    # Calculate the correlation between the delay of this node and the anomaly node
                    data = abs(baseline_df[u].corr(latency_df[normal_edge+"&p50"]))
            data = 0 if np.isnan(data) else data
            data = round(data, 3)
            anomaly_graph.add_edge(u, v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']

    for u, v in edges:
        if anomaly_graph.nodes[v]['type'] == 'host' and anomaly_graph.nodes[u]['type'] != 'instance':
            anomaly_graph.remove_edge(u, v)

    personalization = {}
    for node in DG.nodes():
        if node in nodes:
            personalization[node] = 0

    svc_personalization_map = {}
    svc_personalization_count = {}
    # Assigning weights to personalized arrays
    nodes.append('node')
    for node in nodes:
        if node == 'unknown': continue
        if DG.nodes[node]['type'] == 'service':
            personalization[node] = round(svc_personalization(
            node, anomaly_graph, baseline_df, faults_name, begin_timestamp, end_timestamp), 3)
        elif DG.nodes[node]['type'] == 'host': 
            personalization[node] = round(node_personalization(
            node, anomaly_graph, baseline_df, faults_name, begin_timestamp, end_timestamp), 3)
        elif DG.nodes[node]['type'] == 'instance':
            svc = instance_svc_map[node]
            svc_personalization_map.setdefault(svc, 0)
            svc_personalization_count.setdefault(svc, 0)
            p, max_corr = instance_personalization(
            svc, anomaly_graph, baseline_df, faults_name, node, begin_timestamp, end_timestamp)
            # personalization[node] = p / anomaly_graph.degree(node)
            personalization[node] = round(p, 3)

    for node in personalization.keys():
        if np.isnan(personalization[node]):
            personalization[node] = 0

    # The personalized random walk algrithm
    try:
        anomaly_score = nx.pagerank(
            anomaly_graph, alpha=0.85, personalization=personalization, max_iter=10000)
    except:
        anomaly_score = nx.pagerank(
            anomaly_graph, alpha=0.85, personalization=personalization, max_iter=10000, tol=1.0e-1)

    anomaly_score = sorted(anomaly_score.items(),
                           key=lambda x: x[1], reverse=True)
    return anomaly_score


def remove_host_score(anomaly_score, anomaly_graph):
    for node in anomaly_graph.nodes():
        if anomaly_graph.nodes[node]['type'] == 'host':
            for score in anomaly_score:
                if score[0] == node:
                    anomaly_score.remove(score)


def count_rank(root_cause, failure_instance_scores):
    root_cause = root_cause.replace('_', '-', 1)
    os_count = 0
    for i in range(len(failure_instance_scores)):
        if 'os' in failure_instance_scores[i][0] or 'db' in failure_instance_scores[i][0]:
            os_count += 1
            continue
        if root_cause in failure_instance_scores[i][0]:
            return i + 1 - os_count
    return 0

def print_pr(nums):
    pr1 = 0
    pr2 = 0
    pr3 = 0
    pr4 = 0
    pr5 = 0
    pr6 = 0
    pr7 = 0
    pr8 = 0
    pr9 = 0
    pr10 = 0
    fill_nums = []
    for num in nums:
        if num != 0 and num < 10:
            fill_nums.append(num)
    for num in fill_nums:
        if num <= 10:
            pr10 += 1
            if num <= 9:
                pr9 += 1
                if num <= 8:
                    pr8 += 1
                    if num <= 7:
                        pr7 += 1
                        if num <= 6:
                            pr6 += 1
                            if num <= 5:
                                pr5 += 1
                                if num <= 4:
                                    pr4 += 1
                                    if num <= 3:
                                        pr3 += 1
                                        if num <= 2:
                                            pr2 += 1
                                            if num == 1:
                                                pr1 += 1
    pr_1 = round(pr1 / len(fill_nums), 3)
    pr_2 = round(pr2 / len(fill_nums), 3)
    pr_3 = round(pr3 / len(fill_nums), 3)
    pr_4 = round(pr4 / len(fill_nums), 3)
    pr_5 = round(pr5 / len(fill_nums), 3)
    pr_6 = round(pr6 / len(fill_nums), 3)
    pr_7 = round(pr7 / len(fill_nums), 3)
    pr_8 = round(pr8 / len(fill_nums), 3)
    pr_9 = round(pr9 / len(fill_nums), 3)
    pr_10 = round(pr10 / len(fill_nums), 3)
    print('PR@1:' + str(pr_1))
    print('PR@3:' + str(pr_3))
    print('PR@5:' + str(pr_5))
    print('PR@10:' + str(pr_10))
    avg_1 = pr_1
    avg_3 = round((pr_1 + pr_2 + pr_3) / 3, 3)
    avg_5 = round((pr_1 + pr_2 + pr_3 + pr_4 + pr_5) / 5, 3)
    avg_10 = round((pr_1 + pr_2 + pr_3 + pr_4 + pr_5 + pr_6 + pr_7 + pr_8 + pr_9 + pr_10) / 10, 3)
    print('AVG@1:' + str(avg_1))
    print('AVG@3:' + str(avg_3))
    print('AVG@5:' + str(avg_5))
    print('AVG@10:' + str(avg_10))
    return pr_1, pr_3, pr_5, pr_10, avg_1, avg_3, avg_5, avg_10

def my_acc(scoreList, rightOne, n=None):
    node_rank = [_[0] for _ in scoreList]
    if n is None:
        n = len(scoreList)
    s = 0.0
    for i in range(len(rightOne)):
        if rightOne[i] in node_rank:
            rank = node_rank.index(rightOne[i]) + 1
            s += (n - max(0, rank - len(rightOne))) / n
        else:
            s += 0
    s /= len(rightOne)
    return s

def getInstancesName(folder):
    success_rate_file_name = folder +'/' + 'success_rate.csv'
    success_rate_source_data = pd.read_csv(success_rate_file_name)
    headers = success_rate_source_data.columns
    instances = []
    for header in headers:
        if 'timestamp' in header: continue
        instances.append(header)
    instancesSet = set(instances)
    # print(instancesSet)
    return instancesSet

def cutSvcNameForAnomalyNodes(anomaly_nodes):
    anomaly_nodes_cut = []
    for node in anomaly_nodes:
        if "&p50" in node:
            node = node[:-4]
        anomaly_nodes_cut.append(node)
    return anomaly_nodes_cut

def getRootCauseSvc(root_cause):
    if '-' not in root_cause: return root_cause
    return root_cause[:root_cause.find('-')]

def getCandidateList(root_cause_list, count, svc_instances_map, instance_svc_map, DG):
    root_cause_candidate_list = []
    for i in range(min(count, len(root_cause_list))):
        root_cause = root_cause_list[i]
        root_cause_candidate_list.append(root_cause)
        if DG.nodes[root_cause]['type'] == 'instance':
            # Instance root cause candidates plus services
            root_cause_candidate_list.append(instance_svc_map[root_cause])
        elif DG.nodes[root_cause]['type'] == 'service':
            for i in svc_instances_map[root_cause]: root_cause_candidate_list.append(i)
    return root_cause_candidate_list

def rank(classification_count, root_cause_list, label_data):
    rank_list = {}
    for i,root_cause in enumerate(root_cause_list):
        for item in enumerate(classification_count):
            key = item[1][0]
            value = item[1][1]
            try:
                metric_root_cause = label_data.iloc[key - 1]['cmdb_id']
                # if root_cause in metric_root_cause or metric_root_cause in root_cause:
                if root_cause == metric_root_cause:
                    rank_list.setdefault(metric_root_cause, (len(root_cause_list) - i) * value)
                    break
            except:
                pass
        try:
            a = rank_list[metric_root_cause]
            b = rank_list[root_cause]
        except:
            rank_list.setdefault(root_cause, len(root_cause_list) - i)
    return rank_list


def run_metric_sum_corr_enhance(data, dag_cg, datapath, dataset, dk, score_func, root_cause):
    graph, _, _ = run_graph_discovery_metric_sum_corr_enhance(data, dag_cg, datapath, dataset, dk, score_func)
    tp = 'graph_discovery_metric_sum_corr_enhance'
    folder = datetime.now().strftime("%Y-%m-%d-%H.%M.%S") + '&' + root_cause + '/'
    os.mkdir(folder)
    nx.write_gpickle(graph, folder + 'graph' + '-' + tp + '-' + root_cause + '-' + str(datetime.timestamp(datetime.now())) + '.gpickle')
    return run_sum(graph)

def run_metric_sum_personalization_enhance(data, dag_cg, datapath, dataset, dk, score_func, root_cause):
    graph, _, _, personalization = run_graph_discovery_metric_sum_personalization_enhance(data, dag_cg, datapath, dataset, dk, score_func)
    tp = 'graph_discovery_metric_sum_personalization_enhance'
    folder = datetime.now().strftime("%Y-%m-%d-%H.%M.%S") + '&' + root_cause + '/'
    os.mkdir(folder)
    nx.write_gpickle(graph, folder + 'graph' + '-' + tp + '-' + root_cause + '-' + str(datetime.timestamp(datetime.now())) + '.gpickle')
    return run_sum_personalization(graph, personalization)


if __name__ == '__main__':

    folder_list = ['/Users/zhuyuhan/Documents/391-WHU/experiment/researchProject/MicroIRC/data/data2/1']
    label_file_list = ['DejaVu-A1']
    i_t_pr_1 = 0; i_t_pr_3 = 0; i_t_pr_5 = 0; i_t_pr_10 = 0; i_t_avg_1 = 0; i_t_avg_3 = 0; i_t_avg_5 = 0; i_t_avg_10 = 0
    s_t_pr_1 = 0; s_t_pr_3 = 0; s_t_pr_5 = 0; s_t_pr_10 = 0; s_t_avg_1 = 0; s_t_avg_3 = 0; s_t_avg_5 = 0; s_t_avg_10 = 0

    i_t_pr_1_a = 0; i_t_pr_3_a = 0; i_t_pr_5_a = 0; i_t_pr_10_a = 0; i_t_avg_1_a = 0; i_t_avg_3_a = 0; i_t_avg_5_a = 0; i_t_avg_10_a = 0
    s_t_pr_1_a = 0; s_t_pr_3_a = 0; s_t_pr_5_a = 0; s_t_pr_10_a = 0; s_t_avg_1_a = 0; s_t_avg_3_a = 0; s_t_avg_5_a = 0; s_t_avg_10_a = 0

    data_count = len(folder_list)
    for i in range(data_count):
        folder = folder_list[i]

        # params
        minute = 10
        alpha = 0.8
        instance_tolerant = 0.01
        service_tolerant = 0.03
        train = True
        candidate_count = 20
        class_num = 20

        # time_data
        metric_source_data = pd.read_csv(folder + '/' + 'metric.norm.csv')
        metric_source_data = metric_source_data.fillna(0)
        time_data = metric_source_data.iloc[:,0]

        # read root_causes
        label_file_name = folder + '/' + 'label.csv'
        label_data = pd.read_csv(label_file_name, encoding='utf-8')
        label_set = set()
        label_map = {}
        for index, raw in label_data.iterrows():
            label_set.add(raw['cmdb_id'] + raw['fault_description'])
        label_list = sorted(list(label_set))
        for label in list(label_set):
            label_map[label] = label_list.index(label)
        class_num = len(label_set)
        root_causes = label_data['cmdb_id']

        time_list = []

        for row in label_data.itertuples():
            root_cause = row[4]
            root_cause_level = row[6]
            # real_time = label_list[i] + row[1]
            # real_timestamp = int(time.mktime(time.strptime(real_time, "%Y-%m-%d %H:%M:%S")))
            real_timestamp = int(row[2])
            begin_timestamp = real_timestamp - 30 * minute
            end_timestamp = real_timestamp + 30 * minute
            failure_type = row[5]
            lb = label_map[root_cause + str(row[3])]
            t = Time(begin_timestamp, end_timestamp, root_cause, root_cause_level, failure_type, lb)
            time_list.append(t)

        # for row in label_data.itertuples():
        #     root_cause = row[3]
        #     root_cause_level = row[6]
        #     # real_time = label_list[i] + row[1]
        #     # real_timestamp = int(time.mktime(time.strptime(real_time, "%Y-%m-%d %H:%M:%S")))
        #     real_timestamp = int(row[2])
        #     begin_timestamp = real_timestamp - 30 * minute
        #     end_timestamp = real_timestamp + 30 * minute
        #     failure_type = row[5]
        #     lb = label_map[root_cause + str(row[4])]
        #     t = Time(begin_timestamp, end_timestamp, root_cause, root_cause_level, failure_type, lb)
        #     time_list.append(t)

        # get casualIL graph
        graph = parse_graph('/Users/zhuyuhan/Documents/159-WHU/project/DejaVu/exp/SSF/data/A1/graph.yml')

        # build svc call
        # call_file_name = folder + '/' + 'call.csv'
        # call_data = pd.read_csv(call_file_name)
        # call_set = []
        # for head in call_data.columns:
        #     if 'timestamp' in head: continue
        #     call_set.append(head[:head.find('&')])

        # ablation result
        nums_ablation = []
        svc_nums_ablation = []

        nums = []
        svc_nums = []
        instance_level_nums = []
        svc_level_nums = []
        failure_type_map = {}
        acc = 0
        acc_count = 0
        acc_ablation = 0
        acc_ablation_count = 0
        for t in time_list:
            root_cause = t.root_cause
            root_cause_level = t.root_cause_level
            begin_timestamp = t.begin
            end_timestamp = t.end
            failure_type = t.failure_type

            print('#################root_cause:' + root_cause + '#################')
            anomaly_source = root_cause
            # file_dir = folder
            # collect instance names
            # instances = getInstancesName(file_dir)

            # read latency data
            # latency = pd.read_csv(file_dir + '/' + 'call.csv')

            # qps data
            # qps_file_name = file_dir + '/' + 'svc_qps.csv'
            # qps_source_data = pd.read_csv(qps_file_name)
            # qps_source_data = dfTimelimit(qps_source_data, begin_timestamp, end_timestamp)
            # anomalie_instances = birch_ad_with_smoothing(qps_source_data, instance_tolerant)

            # success rate data
            # success_rate_file_name = file_dir + '/' + 'success_rate.csv'
            # success_rate_source_data = pd.read_csv(success_rate_file_name)
            # success_rate_source_data = dfTimelimit(success_rate_source_data, begin_timestamp, end_timestamp)
            # anomalie_instances += birch_ad_with_smoothing(success_rate_source_data, instance_tolerant)

            # node data
            # node_file_name = file_dir + '/' + 'node.csv'
            # node_source_data = pd.read_csv(node_file_name)
            # for head in node_source_data.columns:
            #     if 'node' not in head:
            #         node_source_data = node_source_data.drop([head], axis=1)
            #
            # latency = latency.join(node_source_data)

            # 取当前根因的集群指标信息
            metric_source_data_time_limit = dfTimelimit(metric_source_data, begin_timestamp, end_timestamp)

            # anomaly detection
            # anomalies = birch_ad_with_smoothing(latency, service_tolerant)

            # anomaly_nodes = []
            # for anomaly in anomalies:
            #     edge = anomaly.split('_')
            #     anomaly_nodes.append(edge[1])
            #
            # anomaly_nodes = set(anomaly_nodes)
            # # Build the call graph with examples for subsequent PageRank
            # DG, svc_instances_map, instance_svc_map = attributed_graph(instances, call_set, root_cause)

            # Building anomaly subgraphs and scoring with personalized PageRank
            # anomaly_score = anomaly_subgraph(
            #     DG, anomalies, latency, file_dir, alpha,
            #     svc_instances_map, instance_svc_map,
            #     begin_timestamp, end_timestamp,
            #     anomalie_instances, root_cause_level, root_cause, call_set)
            try:
                # anomaly_score_map = run_metric_sum_corr_enhance(metric_source_data_time_limit, graph, None, None, 'N', 'P2', root_cause)
                anomaly_score_map = run_metric_sum_personalization_enhance(metric_source_data_time_limit, graph, None, None, 'N', 'P2', root_cause)
                num = count_rank(root_cause, anomaly_score_map)
                nums.append(num)
            except:
                print('failed to root cause because of pagerank:' + root_cause)
        print_pr(nums)


    #     print('exception level:' + root_cause_level)
    #     print('params:')
    #     print('minute:' + str(minute))
    #     print('alpha:' + str(alpha))
    #     print('service_tolerant:' + str(service_tolerant))
    #     print('instance_tolerant:' + str(instance_tolerant))
    #     print('acc:' + str(acc / acc_count))
    #     print('acc_ablation:' + str(acc_ablation / acc_ablation_count))
    #     print('instance_pr:')
    #     i_pr_1, i_pr_3, i_pr_5, i_pr_10, i_avg_1, i_avg_3, i_avg_5, i_avg_10 = print_pr(nums)
    #     i_t_pr_1 += i_pr_1
    #     i_t_pr_3 += i_pr_3
    #     i_t_pr_5 += i_pr_5
    #     i_t_pr_10 += i_pr_10
    #     i_t_avg_1 += i_avg_1
    #     i_t_avg_3 += i_avg_3
    #     i_t_avg_5 += i_avg_5
    #     i_t_avg_10 += i_avg_10
    #
    #     print('svc_pr:')
    #     s_pr_1, s_pr_3, s_pr_5, s_pr_10, s_avg_1, s_avg_3, s_avg_5, s_avg_10 = print_pr(svc_nums)
    #     s_t_pr_1 += s_pr_1
    #     s_t_pr_3 += s_pr_3
    #     s_t_pr_5 += s_pr_5
    #     s_t_pr_10 += s_pr_10
    #     s_t_avg_1 += s_avg_1
    #     s_t_avg_3 += s_avg_3
    #     s_t_avg_5 += s_avg_5
    #     s_t_avg_10 += s_avg_10
    #
    #     # ablation
    #     print('instance_pr_ablation:')
    #     i_pr_1, i_pr_3, i_pr_5, i_pr_10, i_avg_1, i_avg_3, i_avg_5, i_avg_10 = print_pr(nums_ablation)
    #     i_t_pr_1_a += i_pr_1
    #     i_t_pr_3_a += i_pr_3
    #     i_t_pr_5_a += i_pr_5
    #     i_t_pr_10_a += i_pr_10
    #     i_t_avg_1_a += i_avg_1
    #     i_t_avg_3_a += i_avg_3
    #     i_t_avg_5_a += i_avg_5
    #     i_t_avg_10_a += i_avg_10
    #
    #     print('svc_pr_ablation:')
    #     s_pr_1, s_pr_3, s_pr_5, s_pr_10, s_avg_1, s_avg_3, s_avg_5, s_avg_10 = print_pr(svc_nums_ablation)
    #     s_t_pr_1_a += s_pr_1
    #     s_t_pr_3_a += s_pr_3
    #     s_t_pr_5_a += s_pr_5
    #     s_t_pr_10_a += s_pr_10
    #     s_t_avg_1_a += s_avg_1
    #     s_t_avg_3_a += s_avg_3
    #     s_t_avg_5_a += s_avg_5
    #     s_t_avg_10_a += s_avg_10
    #
    #     # PR@K in different levels
    #     print('level_instance_pr:')
    #     l_i_pr_1, l_i_pr_3, l_i_pr_5, l_i_pr_10, l_i_avg_1, l_i_avg_3, l_i_avg_5, l_i_avg_10 = print_pr(instance_level_nums)
    #     print('level_svc_pr:')
    #     l_s_pr_1, l_s_pr_3, l_s_pr_5, l_s_pr_10, l_s_avg_1, l_s_avg_3, l_s_avg_5, l_s_avg_10 = print_pr(svc_level_nums)
    #
    #     # PR@K in different anomaly types
    #     for key in failure_type_map:
    #         print('failure_type:' + str(key))
    #         print_pr(failure_type_map[key])
    #
    # print('instance_pr_total:')
    # print('i_t_pr_1:' + str(round(i_t_pr_1 / data_count, 3)))
    # print('i_t_pr_3:' + str(round(i_t_pr_3 / data_count, 3)))
    # print('i_t_pr_5:' + str(round(i_t_pr_5 / data_count, 3)))
    # print('i_t_pr_10:' + str(round(i_t_pr_10 / data_count, 3)))
    # print('i_t_avg_1:' + str(round(i_t_avg_1 / data_count, 3)))
    # print('i_t_avg_3:' + str(round(i_t_avg_3 / data_count, 3)))
    # print('i_t_avg_5:' + str(round(i_t_avg_5 / data_count, 3)))
    # print('i_t_avg_10:' + str(round(i_t_avg_10 / data_count, 3)))
    # print('svc_pr_total:')
    # print('s_t_pr_1:' + str(round(s_t_pr_1 / data_count, 3)))
    # print('s_t_pr_3:' + str(round(s_t_pr_3 / data_count, 3)))
    # print('s_t_pr_5:' + str(round(s_t_pr_5 / data_count, 3)))
    # print('s_t_pr_10:' + str(round(s_t_pr_10 / data_count, 3)))
    # print('s_t_avg_1:' + str(round(s_t_avg_1 / data_count, 3)))
    # print('s_t_avg_3:' + str(round(s_t_avg_3 / data_count, 3)))
    # print('s_t_avg_5:' + str(round(s_t_avg_5 / data_count, 3)))
    # print('s_t_avg_10:' + str(round(s_t_avg_10 / data_count, 3)))
    #
    # print('instance_pr_ablation_total:')
    # print('i_t_pr_1_a:' + str(round(i_t_pr_1_a / data_count, 3)))
    # print('i_t_pr_3_a:' + str(round(i_t_pr_3_a / data_count, 3)))
    # print('i_t_pr_5_a:' + str(round(i_t_pr_5_a / data_count, 3)))
    # print('i_t_pr_10_a:' + str(round(i_t_pr_10_a / data_count, 3)))
    # print('i_t_avg_1_a:' + str(round(i_t_avg_1_a / data_count, 3)))
    # print('i_t_avg_3_a:' + str(round(i_t_avg_3_a / data_count, 3)))
    # print('i_t_avg_5_a:' + str(round(i_t_avg_5_a / data_count, 3)))
    # print('i_t_avg_10_a:' + str(round(i_t_avg_10_a / data_count, 3)))
    # print('svc_pr_ablation_total:')
    # print('s_t_pr_1_a:' + str(round(s_t_pr_1_a / data_count, 3)))
    # print('s_t_pr_3_a:' + str(round(s_t_pr_3_a / data_count, 3)))
    # print('s_t_pr_5_a:' + str(round(s_t_pr_5_a / data_count, 3)))
    # print('s_t_pr_10_a:' + str(round(s_t_pr_10_a / data_count, 3)))
    # print('s_t_avg_1_a:' + str(round(s_t_avg_1_a / data_count, 3)))
    # print('s_t_avg_3_a:' + str(round(s_t_avg_3_a / data_count, 3)))
    # print('s_t_avg_5_a:' + str(round(s_t_avg_5_a / data_count, 3)))
    # print('s_t_avg_10_a:' + str(round(s_t_avg_10_a / data_count, 3)))
