from pathlib import Path

from diskcache import Cache
from loguru import logger
from pyprof import profile
from tqdm import tqdm
import pandas as pd
from random_walk_failure_instance.metric_sage.time import Time
from DejaVu.config import DejaVuConfig
from DejaVu.evaluation_metrics import get_evaluation_metrics_dict
from DejaVu.workflow import format_result_string
from random_walk_failure_instance.metric_sage.model import run_RCA


from failure_dependency_graph import FDGModelInterface
from random_walk_failure_instance.config import RandomWalkFailureInstanceConfig
from random_walk_failure_instance.model import random_walk


# def workflow(config: RandomWalkFailureInstanceConfig):
#     logger.info(
#         f"\n================================================Config=============================================\n"
#         f"{config!s}"
#         f"\n===================================================================================================\n"
#     )
#     logger.info(f"reproducibility info: {config.get_reproducibility_info()}")
#     logger.add(config.output_dir / 'log')
#     config.save(str(config.output_dir / "config"))
#
#     base = FDGModelInterface(config)
#     fdg = base.fdg
#     mp = base.metric_preprocessor
#
#     y_trues = []
#     y_preds = []
#     dimension = [418, 496, 159]
#
#     with profile("random_walk_main", report_printer=lambda _: logger.info(f"\n{_}")) as profiler:
#         folder = "/Users/zhuyuhan/Documents/391-WHU/experiment/researchProject/MicroIRC/data/data2/1"
#         minute = 10
#         # time_data
#         metric_source_data = pd.read_csv(folder + '/' + 'metric.norm.csv')
#         metric_source_data = metric_source_data.fillna(0)
#         time_data = metric_source_data.iloc[:,0]
#
#         # read root_causes
#         label_file_name = folder + '/' + 'label.csv'
#         label_data = pd.read_csv(label_file_name, encoding='utf-8')
#         label_set = set()
#         label_map = {}
#         for index, raw in label_data.iterrows():
#             label_set.add(raw['cmdb_id'] + '&' + raw['fault_description'])
#         label_list = sorted(list(label_set))
#         for label in list(label_set):
#             label_map[label] = label_list.index(label)
#         class_num = len(label_set)
#         root_causes = label_data['cmdb_id']
#
#         time_list = []
#
#         for row in label_data.itertuples():
#             root_cause = row[4]
#             root_cause_level = row[6]
#             # real_time = label_list[i] + row[1]
#             # real_timestamp = int(time.mktime(time.strptime(real_time, "%Y-%m-%d %H:%M:%S")))
#             real_timestamp = int(row[2])
#             # begin_timestamp = real_timestamp - 30 * minute
#             begin_timestamp = real_timestamp - 60 * minute
#             # end_timestamp = real_timestamp + 30 * minute
#             end_timestamp = real_timestamp + 60 * minute
#             failure_type = row[5]
#             lb = label_map[root_cause + '&' + str(row[3])]
#             t = Time(begin_timestamp, end_timestamp, root_cause, root_cause_level, failure_type, lb)
#             time_list.append(t)
#         # for row in label_data.itertuples():
#         #     root_cause = row[3]
#         #     root_cause_level = row[6]
#         #     # real_time = label_list[i] + row[1]
#         #     # real_timestamp = int(time.mktime(time.strptime(real_time, "%Y-%m-%d %H:%M:%S")))
#         #     real_timestamp = int(row[2])
#         #     begin_timestamp = real_timestamp - 30 * minute
#         #     end_timestamp = real_timestamp + 30 * minute
#         #     failure_type = row[5]
#         #     lb = label_map[root_cause + '&' + str(row[4])]
#         #     t = Time(begin_timestamp, end_timestamp, root_cause, root_cause_level, failure_type, lb)
#         #     time_list.append(t)
#         graphsage = trainGraphSage(time_list, class_num, None, dimension[0])
#         nums = []
#         acc_count = 0
#         acc = 0
#         for fid in tqdm(base.test_failure_ids):
#             cache_dir = Path("SSF/tmp/failure_instance_random_walk_cache") / config.data_dir.relative_to("SSF/") / f"{fid=}"
#             logger.info(f"Cache dir: {cache_dir}")
#             cache = Cache(
#                 directory=str(cache_dir),
#                 size_limit=int(1e10),
#             )
#             # 异常时间戳
#             failure_ts = fdg.failure_at(fid)["timestamp"]
#             # fdg图
#             graph = fdg.networkx_graph_at(fid).copy()
#             for failure_class, class_metrics in zip(
#                     fdg.failure_classes, mp(failure_ts, window_size=config.window_size)
#             ):
#                 for failure_instance, instance_metrics in zip(
#                         fdg.failure_instances[failure_class],
#                         class_metrics,
#                 ):
#                     assert instance_metrics.shape == (fdg.metric_number_dict[failure_class], sum(config.window_size))
#                     graph.add_node(failure_instance, values=instance_metrics.numpy())
#             failure_instance_scores = random_walk(
#                 graph,
#                 config=config,
#                 cache=cache,
#             )
#             failure_instance_scores = sorted(list(failure_instance_scores.keys()), key=lambda x: failure_instance_scores[x], reverse=True)
#             time = None
#             for t in time_list:
#                 if t.is_in_time(failure_ts):
#                     time = t
#                     break
#             if time is None:
#                 continue
#             if 'docker' not in time.root_cause and 'service' not in time.root_cause:
#                 continue
#             val = []
#             for i in range(time.begin_index, time.end_index + 1): val.append(i)
#             val_output = graphsage.forward(val, metric_source_data.iloc[:,2:].loc[val], is_node_train_index=False)
#             classification = val_output.data.numpy().argmax(axis=1)
#             failure_instance_scores_final = []
#             for f in failure_instance_scores:
#                 if 'os' not in f and 'db' not in f:
#                     failure_instance_scores_final.append(f)
#
#             failure_instance_scores = {i: len(failure_instance_scores_final) - failure_instance_scores_final.index(i) for i in failure_instance_scores_final}
#             failure_instance_scores = rank(failure_instance_scores, classification, label_map)
#             acc_temp = my_acc_IRC(failure_instance_scores, [root_cause])
#             if acc_temp > 0:
#                 acc_count += 1
#                 acc += acc_temp
#             nums.append(count_rank(time.root_cause, sorted(list(failure_instance_scores.keys()), key=lambda x: failure_instance_scores[x], reverse=True)))
#     print_pr(nums)
#     print("Acc" + str(acc / acc_count))

def workflow(config: RandomWalkFailureInstanceConfig):
    logger.info(
        f"\n================================================Config=============================================\n"
        f"{config!s}"
        f"\n===================================================================================================\n"
    )
    logger.info(f"reproducibility info: {config.get_reproducibility_info()}")
    logger.add(config.output_dir / 'log')
    config.save(str(config.output_dir / "config"))

    base = FDGModelInterface(config)
    fdg = base.fdg
    mp = base.metric_preprocessor

    y_trues = []
    y_preds = []
    dimension = [418, 496, 159]

    with profile("random_walk_main", report_printer=lambda _: logger.info(f"\n{_}")) as profiler:
        for fid in tqdm(base.test_failure_ids):
            cache_dir = Path("SSF/tmp/failure_instance_random_walk_cache") / config.data_dir.relative_to("SSF/") / f"{fid=}"
            logger.info(f"Cache dir: {cache_dir}")
            cache = Cache(
                directory=str(cache_dir),
                size_limit=int(1e10),
            )
            # 异常时间戳
            failure_ts = fdg.failure_at(fid)["timestamp"]
            # fdg图
            graph = fdg.networkx_graph_at(fid).copy()
            for failure_class, class_metrics in zip(
                    fdg.failure_classes, mp(failure_ts, window_size=config.window_size)
            ):
                for failure_instance, instance_metrics in zip(
                        fdg.failure_instances[failure_class],
                        class_metrics,
                ):
                    assert instance_metrics.shape == (fdg.metric_number_dict[failure_class], sum(config.window_size))
                    graph.add_node(failure_instance, values=instance_metrics.numpy())
            failure_instance_scores = random_walk(
                graph,
                config=config,
                cache=cache,
            )
            failure_instance_scores = sorted(list(failure_instance_scores.keys()), key=lambda x: failure_instance_scores[x], reverse=True)
            y_trues.append(set(fdg.root_cause_instances_of(fid)))
            y_preds.append(
                sorted(list(failure_instance_scores.keys()), key=lambda x: failure_instance_scores[x], reverse=True))

    for y_true, y_pred in zip(y_trues, y_preds):
        logger.info(
            f"{';'.join(y_true):<30}"
            f"|{', '.join(y_pred[:5]):<50}"
        )
    metrics = get_evaluation_metrics_dict(y_trues, y_preds, max_rank=fdg.n_failure_instances)
    logger.info(format_result_string(
        metrics,
        profiler,
        DejaVuConfig().from_dict(args_dict=config.as_dict(), skip_unsettable=True)
    ))
    return y_trues, y_preds

def trainGraphSage(time_list, class_num, label_file, dimension, train = False):
    # build svc call
    # call_file_name = folder + '/' + 'call.csv'
    # call_data = pd.read_csv(call_file_name)
    # call_set = []
    # for head in call_data.columns:
    #     if 'timestamp' in head: continue
    #     call_set.append(head[:head.find('&')])
    metric_source_data = pd.read_csv('/Users/zhuyuhan/Documents/391-WHU/experiment/researchProject/MicroIRC/data/data2/1/metric.norm.csv')
    metric_source_data = metric_source_data.fillna(0)
    data = metric_source_data.iloc[:,2:]
    time_data = metric_source_data.iloc[:, 1:2]
    node_num = 0
    for i,row in time_data.iterrows():
        for j, t in enumerate(time_list):
            t.in_time(int(time_data[i:i+1]['timestamp']), i)
    for t in time_list:
        node_num += t.count
    # for i, column in data.items():
    #     x = np.array(column)
    #     x = np.where(np.isnan(x), 0, x)
    #     normalized_x = preprocessing.normalize([x])
    #
    #     X = normalized_x.reshape(-1, 1)
    #     data[i] = X

    return run_RCA(node_num, dimension, data, time_data, time_list, data, None, class_num, label_file ,train, False)

def rank(failure_instance_scores, classification, label_map):
    instance_weight = {}
    for i in classification:
        for k in label_map:
            if label_map[k] == i:
                key = k[:10]
                # instance_weight.setdefault(k[:10], 0)
                # instance_weight[k[:10]] = instance_weight[k[:10]] + 1
                # key = k[:k.index('&')]
                instance_weight.setdefault(key, 0)
                instance_weight[key] = instance_weight[key] + 1
    for k in instance_weight:
        for v in failure_instance_scores:
            if k in v:
                failure_instance_scores[v] = failure_instance_scores[v] * instance_weight[k]
    return failure_instance_scores

def count_rank(root_cause, failure_instance_scores):
    os_count = 0
    for i in range(len(failure_instance_scores)):
        if 'os' in failure_instance_scores[i] or 'db' in failure_instance_scores[i]:
            os_count += 1
            continue
        if root_cause in failure_instance_scores[i]:
            return i + 1 - os_count
    return 0

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

def my_acc_IRC(scoreList, rightOne, n=None):
    node_rank = [_ for _ in scoreList]
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
