"""
  Requests arrival following Poisson distribution
"""

import networkx as gp
from numpy import *
import uuid
import time
from collections import OrderedDict
from random import randint
import pickle
import copy
from confidence import mean_confidence_interval
from algorithms import optimal
from algorithms import baseline
from algorithms import bast
from algorithms import last
from algorithms import rast
# from algorithms import past
from algorithms import pastv2

def re_form_dict(target, num):
    mirror = OrderedDict()
    for ns_len, cost_list in target.items():
        mirror[ns_len] = sum(cost_list)/float(num)
    return mirror
# General requirements
################################################

def _reform_flow_list(sflow_dict):
    total_flow_list = list()
    for rtan, flow_list in sflow_dict.items():
        total_flow_list.extend(flow_list)
    return total_flow_list


NumberOfNode = 50  # lower number of nodes to gain benefits
NumberOfAccessNode = 10
NumberOfEdgeNode = 40


conn_prob = 0.6
maxDelay = 30
MB = 10**6
maxRetry = 500

prop_delay = [10, 20, 30, 40, 50]   # in ms
# prop_delay = [1, 10, 30, 100]
# prop_delay = range(1, 11)

# # bw_list = [1, 10] # in Gbps
trans_rate = [10*MB, 20*MB, 30*MB, 40*MB, 50*MB]    # in Mbps, cpu processing is calculated through traffic rate

# trans_rate = [50*MB, 100*MB]    # in Mbps, cpu processing is calculated through traffic rate


# Lantency budget
# e2e_latency = [10, 20, 30, 100]   # in ms for 5 services
e2e_latency = [50, 100]   # in ms for 5 services

# service_info = list()
# Add co-efficience alpha and beta for specific service here
cons = [(86.982, 5.7892), (39.205, 2.9545), (32.595, 4.5222), (86.982, 5.7892), (39.205, 2.9545)]

#service_type = [{'delay': random.choice(e2e_latency), 'para': (86.982, 5.7892)}, {'delay': random.choice(e2e_latency), 'para': (39.205, 2.9545)},
#                {'delay': random.choice(e2e_latency), 'para': (32.595, 4.5222)}, {'delay': random.choice(e2e_latency), 'para': (86.982, 5.7892)},
#                {'delay': random.choice(e2e_latency), 'para': (39.205, 2.9545)}]

service_type = [{'delay': random.choice(randint(50, 100)), 'para': (86.982, 5.7892)}, {'delay': random.choice(randint(50, 100)), 'para': (39.205, 2.9545)},
                {'delay': random.choice(randint(50, 100)), 'para': (32.595, 4.5222)}, {'delay': random.choice(randint(50, 100)), 'para': (86.982, 5.7892)},
                {'delay': random.choice(randint(50, 100)), 'para': (39.205, 2.9545)}]


pickle.dump(service_type, open("service_type.pickle", "wb"))

# service_type = pickle.load(open("service_type.pickle", "rb"))

# numOfTan = 2      # this is a special parameter

rdgraph = gp.fast_gnp_random_graph(NumberOfNode, conn_prob)

# rdgraph = gp.barabasi_albert_graph(NumberOfNode, 20)


# rdgraph = gp.balanced_tree(2, 4)


# rdgraph = gp.gnp_random_graph(NumberOfNode, conn_prob)

node_list = range(0, NumberOfNode)
access_node_list = random.choice(node_list, NumberOfAccessNode, replace=False)
# pickle.dump(access_node_list, open("an_list.pickle", "wb"))


# # print "access list:", access_node_list
edge_node_list = [node for node in node_list if node not in access_node_list]
# pickle.dump(edge_node_list, open("en_list.pickle", "wb"))

##########################################################################
# # Add node availability
for gnode in rdgraph.nodes():  # type(node) is integer
    rdgraph.node[gnode]['role'] = 0 if gnode in access_node_list else 1           # role can be either access node (0) or edge node (1)

for edge in rdgraph.edges():
    fst_node = edge[0]
    snd_node = edge[1]
    # rdgraph[fst_node][snd_node]['bw'] = 1 * 1000 * MB       # in Mbps
    rdgraph[fst_node][snd_node]['bw'] = random.randint(1, 11) * 1000 * MB  # in Mbps
    rdgraph[fst_node][snd_node]['delay'] = random.choice(prop_delay)  # in ms
    rdgraph[fst_node][snd_node]['load'] = list()


gp.write_gpickle(rdgraph, 'new_sample_graph.gpickle')

# rdgraph = gp.read_gpickle('sample_graph.gpickle')

########################################################################


def access_edge_mapping(graph):
    given_service_list = list()
    for access_node in access_node_list:
        service_info = OrderedDict()
        service_info['id'] = uuid.uuid4()
        service_info['type'] = random.choice(service_type)
        mapping = OrderedDict()
        mapping['access'] = access_node
        visited_edge_dict = OrderedDict()
        # target_edge_node = None
        # target_cost = 1000  # initial value
        for edge_node in edge_node_list:
            # path = gp.dijkstra_path(graph, access_node, edge_node, weight='delay')
            if not gp.has_path(graph, access_node, edge_node):
                continue
            paths = gp.all_shortest_paths(graph, access_node, edge_node, weight='delay')
            # print paths
            path_list = [path for path in paths]
            path = list(path_list)[0]
            cost = path_cost(path, graph)
            if cost > service_info['type']['delay']:
                continue
            visited_edge_dict[edge_node] = OrderedDict()  # remove it afterward
            visited_edge_dict[edge_node]['path'] = list(path_list)
            visited_edge_dict[edge_node]['cost'] = cost
            # if cost < target_cost:
            #     target_cost = cost
            #     target_edge_node = edge_node

        # if target_edge_node is None:
        #     print 'There is no edge node available'
        #     break
        if not visited_edge_dict:
            return given_service_list
        target_edge_node = random.choice(visited_edge_dict.keys())
        mapping['edge'] = target_edge_node
        mapping['path'] = visited_edge_dict[target_edge_node]['path']
        mapping['cost'] = visited_edge_dict[target_edge_node]['cost']
        service_info['mapping'] = mapping
        given_service_list.append(service_info)
    return given_service_list


# there are at least 2 nodes
def path_cost(path_list, graph):
    cost = 0
    for idx, node in enumerate(path_list):
        if (idx + 1) < len(path_list):
            nxt_node = path_list[idx+1]
            # print 'a rm link delay:', graph[node][nxt_node]['delay']
            cost += graph[node][nxt_node]['delay']
        else:
            break
    return cost


# make sure flow attributes must not change later, other everything goes wrong!!!
def generate_request(timer, num, graph):
    service_list = access_edge_mapping(graph)
    flow_dict = OrderedDict()
    # Loop with number of flows
    for f_count in range(0, num):
        flow = OrderedDict()
        flow['id'] = uuid.uuid4()
        flow['rate'] = random.choice(trans_rate)
        flow['service'] = random.choice(service_list)
        flow['lifetime'] = random.exponential(10800) + timer
        # check whether flow is accepted

        flow_dict.update(flow)
    return flow_dict


def flow_mapping(timer, mflow, graph):
    cp_graph = graph.copy()
    is_matched = False
    rendered_service = mflow['service']
    # print rendered_service['mapping']['path']
    path_list = rendered_service['mapping']['path']
    # rendered_mapping = rendered_service['mapping']
    for rendered_path in path_list:
        is_over = False
        for ni, rnode in enumerate(rendered_path):
            if (ni + 1) < len(rendered_path):
                nxt_node = rendered_path[ni+1]
                edge_load = cp_graph[rnode][nxt_node]['load']
                edge_cap = cp_graph[rnode][nxt_node]['bw']
                # if not edge_load:
                #     edge_load.append(mflow)
                curr_load = sum([eflow['rate'] for eflow in edge_load if eflow['lifetime'] >= timer])
                # hist_load = sum([eflow['rate'] for eflow in edge_load])
                # if curr_load < hist_load:
                #     print 'poisson worked!!!'
                if (curr_load + mflow['rate']) <= edge_cap:
                    edge_load.append(mflow)
                else:
                    # print "flow is rejected due to capacity constraint"
                    is_over = True
                    break
        if not is_over:
            is_matched = True
            break
    if not is_matched:
        print "flow is rejected due to capacity constraint"
        # time.sleep(3)
    else:
        graph = cp_graph

    return is_matched


def classify_flow_source(timer, flow_list, pp_van, pp_list):
    mig_an_list = copy.deepcopy(pp_list)
    mig_an_list.append(pp_van)
    mig_an_dict = OrderedDict()
    for sflow in flow_list:
        if sflow['lifetime'] < timer:
            continue
        ran = random.choice(mig_an_list)
        if ran == pp_van:
            # better to revise for further extension
            continue
        else:
            if mig_an_dict.get(ran) is None:
                mig_an_dict[ran] = list()
            mig_an_dict[ran].append(sflow)
    return mig_an_dict

# intial_graph = graph_initial()
# load the graph
# intial_graph = gp.read_gpickle('ideal_graph.gpickle')
# Formulate the distribution with 100s for Poisson process


tan_list = [1, 2, 3, 4, 5, 6]
# numFlowList = [100]
opt_total_cost_dict = OrderedDict()
opt_total_comm_cost_dict = OrderedDict()
opt_total_buff_cost_dict = OrderedDict()
opt_total_exec_time_dict = OrderedDict()
opt_migration_time_dict = OrderedDict()
opt_number_of_fami = OrderedDict()
opt_total_comp_dict = OrderedDict()

base_total_cost_dict = OrderedDict()
base_total_comm_cost_dict = OrderedDict()
base_total_buff_cost_dict = OrderedDict()
base_total_exec_time_dict = OrderedDict()
base_number_of_fami = OrderedDict()
base_total_comp_cost_dict = OrderedDict()

bast_total_cost_dict = OrderedDict()
bast_total_comm_cost_dict = OrderedDict()
bast_total_buff_cost_dict = OrderedDict()
bast_total_exec_time_dict = OrderedDict()
bast_migration_time_dict = OrderedDict()
bast_number_of_fami = OrderedDict()
bast_total_comp_dict = OrderedDict()

last_total_cost_dict = OrderedDict()
last_total_comm_cost_dict = OrderedDict()
last_total_buff_cost_dict = OrderedDict()
last_total_exec_time_dict = OrderedDict()
last_migration_time_dict = OrderedDict()
last_number_of_fami = OrderedDict()
last_total_comp_dict = OrderedDict()

rast_total_cost_dict = OrderedDict()
rast_total_comm_cost_dict = OrderedDict()
rast_total_buff_cost_dict = OrderedDict()
rast_total_exec_time_dict = OrderedDict()
rast_migration_time_dict = OrderedDict()
rast_number_of_fami = OrderedDict()
rast_total_comp_dict = OrderedDict()


past_total_cost_dict = OrderedDict()
past_total_comm_cost_dict = OrderedDict()
past_total_buff_cost_dict = OrderedDict()
past_total_exec_time_dict = OrderedDict()
past_migration_time_dict = OrderedDict()
past_number_of_fami = OrderedDict()
past_total_comp_dict = OrderedDict()

numFlow = 300

for numOfTan in tan_list:
    opt_total_cost_dict[numOfTan] = list()
    opt_total_comm_cost_dict[numOfTan] = list()
    opt_total_buff_cost_dict[numOfTan] = list()
    opt_total_exec_time_dict[numOfTan] = list()
    opt_migration_time_dict[numOfTan] = list()
    opt_number_of_fami[numOfTan] = 0
    opt_total_comp_dict[numOfTan] = list()

    base_total_cost_dict[numOfTan] = list()
    base_total_comm_cost_dict[numOfTan] = list()
    base_total_buff_cost_dict[numOfTan] = list()
    base_total_exec_time_dict[numOfTan] = list()
    base_number_of_fami[numOfTan] = 0

    bast_total_cost_dict[numOfTan] = list()
    bast_total_comm_cost_dict[numOfTan] = list()
    bast_total_buff_cost_dict[numOfTan] = list()
    bast_total_exec_time_dict[numOfTan] = list()
    bast_migration_time_dict[numOfTan] = list()
    bast_number_of_fami[numOfTan] = 0
    bast_total_comp_dict[numOfTan] = list()

    last_total_cost_dict[numOfTan] = list()
    last_total_comm_cost_dict[numOfTan] = list()
    last_total_buff_cost_dict[numOfTan] = list()
    last_total_exec_time_dict[numOfTan] = list()
    last_migration_time_dict[numOfTan] = list()
    last_number_of_fami[numOfTan] = 0
    last_total_comp_dict[numOfTan] = list()

    rast_total_cost_dict[numOfTan] = list()
    rast_total_comm_cost_dict[numOfTan] = list()
    rast_total_buff_cost_dict[numOfTan] = list()
    rast_total_exec_time_dict[numOfTan] = list()
    rast_migration_time_dict[numOfTan] = list()
    rast_number_of_fami[numOfTan] = 0
    rast_total_comp_dict[numOfTan] = list()

    past_total_cost_dict[numOfTan] = list()
    past_total_comm_cost_dict[numOfTan] = list()
    past_total_buff_cost_dict[numOfTan] = list()
    past_total_exec_time_dict[numOfTan] = list()
    past_migration_time_dict[numOfTan] = list()
    past_number_of_fami[numOfTan] = 0
    past_total_comp_dict[numOfTan] = list()

    retry = 0
    while retry < (maxRetry-1):
        print 'retry:', retry
        intial_graph = gp.read_gpickle('new_sample_graph.gpickle')
        # if retry == 4:
        #     for edge in intial_graph.edges():
        #         first_node = edge[0]
        #         second_node = edge[1]
        #         print intial_graph[first_node][second_node]['load']
        # import time
        # time.sleep(5)
        # net_graph = intial_graph  # achtung!!! some trials failed very early after this for some algorithms
        # net_graph = graph_initial()
        service_list = access_edge_mapping(intial_graph)
        if not service_list:
            continue
        # print service_list
        numOfRequest = 500        # increasing the bw can increase the number of requests
        count = 1
        time_unit = 0  # increase 100s after each iteration
        # mig_trigger = 1000
        flow_set = OrderedDict()
        while True:
            NoFlow = random.poisson(1)
            if not NoFlow:
                time_unit += 30
                continue
            is_overloaded = False
            for i in range(0, NoFlow):
                flow = OrderedDict()
                flow['id'] = uuid.uuid4()
                flow['rate'] = random.choice(trans_rate)
                flow['service'] = random.choice(service_list)
                flow['lifetime'] = random.exponential(10800) + time_unit
                # map the flow to underlying network
                if not flow_mapping(time_unit, flow, intial_graph):
                    print 'Stop at:', count
                    is_overloaded = True
                    break
                tsid = flow['service']['id']
                if flow_set.get(tsid) is None:
                    flow_set[tsid] = list()
                flow_set[tsid].append(flow)
            if is_overloaded:
                retry -= 1
                break
            if count >= numFlow:            # Migration condition
                # start migration
                print 'Migration starts at flow:', count
                vsins = random.choice(service_list)
                van = vsins['mapping']['access']
                pp_an_list = access_node_list.tolist()
                pp_an_list.remove(van)
                tan_list = random.choice(pp_an_list, numOfTan, replace=False).tolist()
                if flow_set.get(vsins['id']) is None:
                    retry -= 1
                    break
                mig_flow_list = flow_set[vsins['id']]
                # classify flows
                mig_an_dict = classify_flow_source(time_unit, mig_flow_list, van, tan_list)
                pp_tan_list = copy.deepcopy(tan_list)
                for tan in tan_list:
                    if mig_an_dict.get(tan) is None:
                        pp_tan_list.remove(tan)
                if not pp_tan_list:
                    print 'skip due to idle access node'
                    retry -= 1
                    break

                print 'Number of visited flows:', len(mig_flow_list)
                print "TAN list:", pp_tan_list
                # print mig_an_dict
                print "migrated flows:", [len(mig_an_dict[i]) for i in pp_tan_list]

                # refactor link usage since flows gonna migrate
                for edge in intial_graph.edges():
                    src_node = edge[0]
                    dst_node = edge[1]
                    for flow_info in _reform_flow_list(mig_an_dict):
                        if flow_info in intial_graph[src_node][dst_node]['load']:
                            intial_graph[src_node][dst_node]['load'].remove(flow_info)

                # print "service path:", vsins['mapping']
                # opt_start_time = time.time()
                # opt_total_cost, opt_comm_cost, opt_buff_cost, opt_total_mig_time, opt_total_comp_cost = optimal.Optimal(time_unit, vsins, pp_tan_list, edge_node_list, mig_an_dict, intial_graph).execute()
                # opt_exec_time = time.time() - opt_start_time
                # 
                # print "Opt delay:", opt_exec_time

                base_start_time = time.time()
                base_total_cost, base_comm_cost, base_buff_cost = baseline.Baseline(time_unit, vsins, pp_tan_list, edge_node_list, mig_an_dict, intial_graph).execute()
                base_exec_time = time.time() - base_start_time

                # bast_start_time = time.time()
                # bast_total_cost, bast_comm_cost, bast_buff_cost, bast_total_mig_time, bast_total_comp_cost = bast.Bast(time_unit, vsins, pp_tan_list, edge_node_list, mig_an_dict, intial_graph).execute()
                # bast_exec_time = time.time() - base_start_time
                # 
                # last_start_time = time.time()
                # last_total_cost, last_comm_cost, last_buff_cost, last_total_mig_time, last_total_comp_cost = last.Last(time_unit, vsins, pp_tan_list, edge_node_list, mig_an_dict, intial_graph).execute()
                # last_exec_time = time.time() - last_start_time
                # 
                # rast_start_time = time.time()
                # rast_total_cost, rast_comm_cost, rast_buff_cost, rast_total_mig_time, rast_total_comp_cost = rast.Rast(time_unit, vsins, pp_tan_list, edge_node_list, mig_an_dict, intial_graph).execute()
                # rast_exec_time = time.time() - rast_start_time
                # 
                # past_start_time = time.time()
                # past_total_cost, past_comm_cost, past_buff_cost, past_total_mig_time, past_total_comp_cost = pastv2.Past(time_unit, vsins, pp_tan_list, edge_node_list, mig_an_dict, intial_graph).execute()
                # past_exec_time = time.time() - past_start_time
                # 
                # print "Past delay:", past_exec_time

                # time.sleep(3)

                # if opt_total_cost:
                #     opt_total_cost_dict[numOfTan].append(opt_total_cost)
                #     opt_total_comm_cost_dict[numOfTan].append(opt_comm_cost)
                #     opt_total_buff_cost_dict[numOfTan].append(opt_buff_cost)
                #     opt_total_exec_time_dict[numOfTan].append(opt_exec_time)
                #     opt_migration_time_dict[numOfTan].append(opt_total_mig_time)
                #     opt_total_comp_dict[numOfTan].append(opt_total_comp_cost)
                # else:
                #     opt_number_of_fami[numOfTan] += 1

                if base_total_cost:
                    base_total_cost_dict[numOfTan].append(base_total_cost)
                    base_total_comm_cost_dict[numOfTan].append(base_comm_cost)
                    base_total_buff_cost_dict[numOfTan].append(base_buff_cost)
                    base_total_exec_time_dict[numOfTan].append(base_exec_time)
                else:
                    # print "Base got failed"
                    # time.sleep(3)
                    base_number_of_fami[numOfTan] += 1

                # if bast_total_cost:
                #     bast_total_cost_dict[numOfTan].append(bast_total_cost)
                #     bast_total_comm_cost_dict[numOfTan].append(bast_comm_cost)
                #     bast_total_buff_cost_dict[numOfTan].append(bast_buff_cost)
                #     bast_total_exec_time_dict[numOfTan].append(bast_exec_time)
                #     bast_migration_time_dict[numOfTan].append(bast_total_mig_time)
                #     bast_total_comp_dict[numOfTan].append(bast_total_comp_cost)
                # else:
                #     bast_number_of_fami[numOfTan] += 1

                # if last_total_cost:
                #     last_total_cost_dict[numOfTan].append(last_total_cost)
                #     last_total_comm_cost_dict[numOfTan].append(last_comm_cost)
                #     last_total_buff_cost_dict[numOfTan].append(last_buff_cost)
                #     last_total_exec_time_dict[numOfTan].append(last_exec_time)
                #     last_migration_time_dict[numOfTan].append(last_total_mig_time)
                #     last_total_comp_dict[numOfTan].append(last_total_comp_cost)
                # else:
                #     last_number_of_fami[numOfTan] += 1
                # 
                # if rast_total_cost:
                #     rast_total_cost_dict[numOfTan].append(rast_total_cost)
                #     rast_total_comm_cost_dict[numOfTan].append(rast_comm_cost)
                #     rast_total_buff_cost_dict[numOfTan].append(rast_buff_cost)
                #     rast_total_exec_time_dict[numOfTan].append(rast_exec_time)
                #     rast_migration_time_dict[numOfTan].append(rast_total_mig_time)
                #     rast_total_comp_dict[numOfTan].append(rast_total_comp_cost)
                # else:
                #     rast_number_of_fami[numOfTan] += 1
                # 
                # if past_total_cost:
                #     past_total_cost_dict[numOfTan].append(past_total_cost)
                #     past_total_comm_cost_dict[numOfTan].append(past_comm_cost)
                #     past_total_buff_cost_dict[numOfTan].append(past_buff_cost)
                #     past_total_exec_time_dict[numOfTan].append(past_exec_time)
                #     past_migration_time_dict[numOfTan].append(past_total_mig_time)
                #     past_total_comp_dict[numOfTan].append(past_total_comp_cost)
                # else:
                #     past_number_of_fami[numOfTan] += 1

                break
            time_unit += 30
            count += NoFlow

        retry = retry + 1
    print "Actual number of reties:", retry
    # time.sleep(3)

# opt_total_cost_result = list()
# opt_total_cost_err = list()
# 
# for num in opt_total_cost_dict:
#     result, err = mean_confidence_interval(opt_total_cost_dict[num], confidence=0.95)
#     opt_total_cost_result.append(result)
#     opt_total_cost_err.append(err)
# 
# opt_comm_cost_result = list()
# opt_comm_cost_err = list()
# 
# for num in opt_total_comm_cost_dict:
#     result, err = mean_confidence_interval(opt_total_comm_cost_dict[num], confidence=0.95)
#     opt_comm_cost_result.append(result)
#     opt_comm_cost_err.append(err)
# 
# opt_buff_cost_result = list()
# opt_buff_cost_err = list()
# 
# for num in opt_total_buff_cost_dict:
#     result, err = mean_confidence_interval(opt_total_buff_cost_dict[num], confidence=0.95)
#     opt_buff_cost_result.append(result)
#     opt_buff_cost_err.append(err)
# 
# 
# opt_exec_time_result = list()
# opt_exec_time_err = list()
# for num in opt_total_exec_time_dict:
#     result, err = mean_confidence_interval(opt_total_exec_time_dict[num], confidence=0.95)
#     opt_exec_time_result.append(result)
#     opt_exec_time_err.append(err)
# 
# opt_mig_time_result = list()
# opt_mig_time_err = list()
# for num in opt_migration_time_dict:
#     result, err = mean_confidence_interval(opt_migration_time_dict[num], confidence=0.95)
#     opt_mig_time_result.append(result)
#     opt_mig_time_err.append(err)
# 
# opt_comp_cost_result = list()
# opt_comp_cost_err = list()
# for num in opt_total_comp_dict:
#     result, err = mean_confidence_interval(opt_total_comp_dict[num], confidence=0.95)
#     opt_comp_cost_result.append(result)
#     opt_comp_cost_err.append(err)
# 
# print "Optimal"
# print "Optimal total cost:", opt_total_cost_result
# print "Optimal communication cost:", opt_comm_cost_result
# print "Optimal buffering cost:", opt_buff_cost_result
# print "Total migration time:", opt_mig_time_result
# print "Execution time:", opt_exec_time_result
# print "Fami:", opt_number_of_fami
# print "Comp cost:", opt_comp_cost_result
# 
# pickle.dump(opt_total_cost_result, open("opt_total_cost_result.pickle", "wb"))
# pickle.dump(opt_total_cost_err, open("opt_total_cost_err.pickle", "wb"))
# 
# pickle.dump(opt_comm_cost_result, open("opt_comm_cost_result.pickle", "wb"))
# pickle.dump(opt_comm_cost_err, open("opt_comm_cost_err.pickle", "wb"))
# 
# pickle.dump(opt_buff_cost_result, open("opt_buff_cost_result.pickle", "wb"))
# pickle.dump(opt_buff_cost_err, open("opt_buff_cost_err.pickle", "wb"))
# 
# pickle.dump(opt_mig_time_result, open("opt_mig_time_result.pickle", "wb"))
# pickle.dump(opt_mig_time_err, open("opt_mig_time_err.pickle", "wb"))
# 
# pickle.dump(opt_exec_time_result, open("opt_exec_time_result.pickle", "wb"))
# pickle.dump(opt_exec_time_err, open("opt_exec_time_err.pickle", "wb"))
# 
# pickle.dump(opt_comp_cost_result, open("opt_comp_cost_result.pickle", "wb"))
# pickle.dump(opt_comp_cost_err, open("opt_comp_cost_err.pickle", "wb"))
# 
# bast_total_cost_result = list()
# bast_total_cost_err = list()
# 
# for num in bast_total_cost_dict:
#     result, err = mean_confidence_interval(bast_total_cost_dict[num], confidence=0.95)
#     bast_total_cost_result.append(result)
#     bast_total_cost_err.append(err)
# 
# bast_comm_cost_result = list()
# bast_comm_cost_err = list()
# 
# for num in bast_total_comm_cost_dict:
#     result, err = mean_confidence_interval(bast_total_comm_cost_dict[num], confidence=0.95)
#     bast_comm_cost_result.append(result)
#     bast_comm_cost_err.append(err)
# 
# bast_buff_cost_result = list()
# bast_buff_cost_err = list()
# 
# for num in bast_total_buff_cost_dict:
#     result, err = mean_confidence_interval(bast_total_buff_cost_dict[num], confidence=0.95)
#     bast_buff_cost_result.append(result)
#     bast_buff_cost_err.append(err)
# 
# bast_exec_time_result = list()
# bast_exec_time_err = list()
# for num in bast_total_exec_time_dict:
#     result, err = mean_confidence_interval(bast_total_exec_time_dict[num], confidence=0.95)
#     bast_exec_time_result.append(result)
#     bast_exec_time_err.append(err)
# 
# bast_mig_time_result = list()
# bast_mig_time_err = list()
# for num in bast_migration_time_dict:
#     result, err = mean_confidence_interval(bast_migration_time_dict[num], confidence=0.95)
#     bast_mig_time_result.append(result)
#     bast_mig_time_err.append(err)
# 
# bast_comp_cost_result = list()
# bast_comp_cost_err = list()
# 
# for num in bast_total_comp_dict:
#     result, err = mean_confidence_interval(bast_total_comp_dict[num], confidence=0.95)
#     bast_comp_cost_result.append(result)
#     bast_comp_cost_err.append(err)
# 
# print "Bast"
# print "Bast total cost:", bast_total_cost_result
# print "Bast communication cost:", bast_comm_cost_result
# print "Bast buffering cost:", bast_buff_cost_result
# print "Total migration time:", bast_mig_time_result
# print "Bast execution time:", bast_exec_time_result
# print "Fami:", bast_number_of_fami
# print "Comp cost:"
# 
# pickle.dump(bast_total_cost_result, open("bast_total_cost_result.pickle", "wb"))
# pickle.dump(bast_total_cost_err, open("bast_total_cost_err.pickle", "wb"))
# 
# pickle.dump(bast_comm_cost_result, open("bast_comm_cost_result.pickle", "wb"))
# pickle.dump(bast_comm_cost_err, open("bast_comm_cost_err.pickle", "wb"))
# 
# pickle.dump(bast_buff_cost_result, open("bast_buff_cost_result.pickle", "wb"))
# pickle.dump(bast_buff_cost_err, open("bast_buff_cost_err.pickle", "wb"))
# 
# pickle.dump(bast_mig_time_result, open("bast_mig_time_result.pickle", "wb"))
# pickle.dump(bast_mig_time_err, open("bast_mig_time_err.pickle", "wb"))
# 
# pickle.dump(bast_exec_time_result, open("bast_exec_time_result.pickle", "wb"))
# pickle.dump(bast_exec_time_err, open("bast_exec_time_err.pickle", "wb"))
# 
# pickle.dump(bast_comp_cost_result, open("bast_comp_cost_result.pickle", "wb"))
# pickle.dump(bast_comp_cost_err, open("bast_comp_cost_err.pickle", "wb"))

base_total_cost_result = list()
base_total_cost_err = list()

for num in base_total_cost_dict:
    result, err = mean_confidence_interval(base_total_cost_dict[num], confidence=0.95)
    base_total_cost_result.append(result)
    base_total_cost_err.append(err)

base_comm_cost_result = list()
base_comm_cost_err = list()

for num in base_total_comm_cost_dict:
    result, err = mean_confidence_interval(base_total_comm_cost_dict[num], confidence=0.95)
    base_comm_cost_result.append(result)
    base_comm_cost_err.append(err)

base_buff_cost_result = list()
base_buff_cost_err = list()

for num in base_total_buff_cost_dict:
    result, err = mean_confidence_interval(base_total_buff_cost_dict[num], confidence=0.95)
    base_buff_cost_result.append(result)
    base_buff_cost_err.append(err)

base_exec_time_result = list()
base_exec_time_err = list()
for num in base_total_exec_time_dict:
    result, err = mean_confidence_interval(base_total_exec_time_dict[num], confidence=0.95)
    base_exec_time_result.append(result)
    base_exec_time_err.append(err)


print "Base"
print "Base total cost:", base_total_cost_result
print "Base communication cost:", base_comm_cost_result
print "Base buffering cost:", base_buff_cost_result
print "Base execution time:", base_exec_time_result
print "Fami:", base_number_of_fami

pickle.dump(base_total_cost_result, open("base_total_cost_result.pickle", "wb"))
pickle.dump(base_total_cost_err, open("base_total_cost_err.pickle", "wb"))

pickle.dump(base_comm_cost_result, open("base_comm_cost_result.pickle", "wb"))
pickle.dump(base_comm_cost_err, open("base_comm_cost_err.pickle", "wb"))

pickle.dump(base_buff_cost_result, open("base_buff_cost_result.pickle", "wb"))
pickle.dump(base_buff_cost_err, open("base_buff_cost_err.pickle", "wb"))

pickle.dump(base_exec_time_result, open("base_exec_time_result.pickle", "wb"))
pickle.dump(base_exec_time_err, open("base_exec_time_err.pickle", "wb"))


# last_total_cost_result = list()
# last_total_cost_err = list()
# 
# for num in last_total_cost_dict:
#     result, err = mean_confidence_interval(last_total_cost_dict[num], confidence=0.95)
#     last_total_cost_result.append(result)
#     last_total_cost_err.append(err)
# 
# last_comm_cost_result = list()
# last_comm_cost_err = list()
# 
# for num in last_total_comm_cost_dict:
#     result, err = mean_confidence_interval(last_total_comm_cost_dict[num], confidence=0.95)
#     last_comm_cost_result.append(result)
#     last_comm_cost_err.append(err)
# 
# last_buff_cost_result = list()
# last_buff_cost_err = list()
# 
# for num in last_total_buff_cost_dict:
#     result, err = mean_confidence_interval(last_total_buff_cost_dict[num], confidence=0.95)
#     last_buff_cost_result.append(result)
#     last_buff_cost_err.append(err)
# 
# last_exec_time_result = list()
# last_exec_time_err = list()
# for num in last_total_exec_time_dict:
#     result, err = mean_confidence_interval(last_total_exec_time_dict[num], confidence=0.95)
#     last_exec_time_result.append(result)
#     last_exec_time_err.append(err)
# 
# last_mig_time_result = list()
# last_mig_time_err = list()
# for num in last_migration_time_dict:
#     result, err = mean_confidence_interval(last_migration_time_dict[num], confidence=0.95)
#     last_mig_time_result.append(result)
#     last_mig_time_err.append(err)
# 
# last_comp_cost_result = list()
# last_comp_cost_err = list()
# 
# for num in last_total_comp_dict:
#     result, err = mean_confidence_interval(last_total_comp_dict[num], confidence=0.95)
#     last_comp_cost_result.append(result)
#     last_comp_cost_err.append(err)
# 
# print "last"
# print "last total cost:", last_total_cost_result
# print "last communication cost:", last_comm_cost_result
# print "last buffering cost:", last_buff_cost_result
# print "last migration time:", last_mig_time_result
# print "Last execution time:", last_exec_time_result
# print "last computation cost:", last_comp_cost_result
# print "Fami:", last_number_of_fami
# 
# pickle.dump(last_total_cost_result, open("last_total_cost_result.pickle", "wb"))
# pickle.dump(last_total_cost_err, open("last_total_cost_err.pickle", "wb"))
# 
# pickle.dump(last_comm_cost_result, open("last_comm_cost_result.pickle", "wb"))
# pickle.dump(last_comm_cost_err, open("last_comm_cost_err.pickle", "wb"))
# 
# pickle.dump(last_buff_cost_result, open("last_buff_cost_result.pickle", "wb"))
# pickle.dump(last_buff_cost_err, open("last_buff_cost_err.pickle", "wb"))
# 
# pickle.dump(last_mig_time_result, open("last_mig_time_result.pickle", "wb"))
# pickle.dump(last_mig_time_err, open("last_mig_time_err.pickle", "wb"))
# 
# pickle.dump(last_exec_time_result, open("last_exec_time_result.pickle", "wb"))
# pickle.dump(last_exec_time_err, open("last_exec_time_err.pickle", "wb"))
# 
# 
# pickle.dump(last_comp_cost_result, open("last_comp_cost_result.pickle", "wb"))
# pickle.dump(last_comp_cost_err, open("last_comp_cost_err.pickle", "wb"))
# 
# rast_total_cost_result = list()
# rast_total_cost_err = list()
# 
# for num in rast_total_cost_dict:
#     result, err = mean_confidence_interval(rast_total_cost_dict[num], confidence=0.95)
#     rast_total_cost_result.append(result)
#     rast_total_cost_err.append(err)
# 
# rast_comm_cost_result = list()
# rast_comm_cost_err = list()
# 
# for num in rast_total_comm_cost_dict:
#     result, err = mean_confidence_interval(rast_total_comm_cost_dict[num], confidence=0.95)
#     rast_comm_cost_result.append(result)
#     rast_comm_cost_err.append(err)
# 
# rast_buff_cost_result = list()
# rast_buff_cost_err = list()
# 
# for num in rast_total_buff_cost_dict:
#     result, err = mean_confidence_interval(rast_total_buff_cost_dict[num], confidence=0.95)
#     rast_buff_cost_result.append(result)
#     rast_buff_cost_err.append(err)
# 
# rast_exec_time_result = list()
# rast_exec_time_err = list()
# for num in rast_total_exec_time_dict:
#     result, err = mean_confidence_interval(rast_total_exec_time_dict[num], confidence=0.95)
#     rast_exec_time_result.append(result)
#     rast_exec_time_err.append(err)
# 
# rast_mig_time_result = list()
# rast_mig_time_err = list()
# for num in rast_migration_time_dict:
#     result, err = mean_confidence_interval(rast_migration_time_dict[num], confidence=0.95)
#     rast_mig_time_result.append(result)
#     rast_mig_time_err.append(err)
# 
# rast_comp_cost_result = list()
# rast_comp_cost_err = list()
# 
# for num in rast_total_comp_dict:
#     result, err = mean_confidence_interval(rast_total_comp_dict[num], confidence=0.95)
#     rast_comp_cost_result.append(result)
#     rast_comp_cost_err.append(err)
# 
# pickle.dump(rast_total_cost_result, open("rast_total_cost_result.pickle", "wb"))
# pickle.dump(rast_total_cost_err, open("rast_total_cost_err.pickle", "wb"))
# 
# pickle.dump(rast_comm_cost_result, open("rast_comm_cost_result.pickle", "wb"))
# pickle.dump(rast_comm_cost_err, open("rast_comm_cost_err.pickle", "wb"))
# 
# pickle.dump(rast_buff_cost_result, open("rast_buff_cost_result.pickle", "wb"))
# pickle.dump(rast_buff_cost_err, open("rast_buff_cost_err.pickle", "wb"))
# 
# pickle.dump(rast_mig_time_result, open("rast_mig_time_result.pickle", "wb"))
# pickle.dump(rast_mig_time_err, open("rast_mig_time_err.pickle", "wb"))
# 
# pickle.dump(rast_exec_time_result, open("rast_exec_time_result.pickle", "wb"))
# pickle.dump(rast_exec_time_err, open("rast_exec_time_err.pickle", "wb"))
# 
# pickle.dump(rast_comp_cost_result, open("rast_comp_cost_result.pickle", "wb"))
# pickle.dump(rast_comp_cost_err, open("rast_comp_cost_err.pickle", "wb"))
# 
# # =============================================================================
# 
# past_total_cost_result = list()
# past_total_cost_err = list()
# 
# for num in past_total_cost_dict:
#     result, err = mean_confidence_interval(past_total_cost_dict[num], confidence=0.95)
#     past_total_cost_result.append(result)
#     past_total_cost_err.append(err)
# 
# past_comm_cost_result = list()
# past_comm_cost_err = list()
# 
# for num in past_total_comm_cost_dict:
#     result, err = mean_confidence_interval(past_total_comm_cost_dict[num], confidence=0.95)
#     past_comm_cost_result.append(result)
#     past_comm_cost_err.append(err)
# 
# past_buff_cost_result = list()
# past_buff_cost_err = list()
# 
# for num in past_total_buff_cost_dict:
#     result, err = mean_confidence_interval(past_total_buff_cost_dict[num], confidence=0.95)
#     past_buff_cost_result.append(result)
#     past_buff_cost_err.append(err)
# 
# past_exec_time_result = list()
# past_exec_time_err = list()
# for num in past_total_exec_time_dict:
#     result, err = mean_confidence_interval(past_total_exec_time_dict[num], confidence=0.95)
#     past_exec_time_result.append(result)
#     past_exec_time_err.append(err)
# 
# past_mig_time_result = list()
# past_mig_time_err = list()
# for num in past_migration_time_dict:
#     result, err = mean_confidence_interval(past_migration_time_dict[num], confidence=0.95)
#     past_mig_time_result.append(result)
#     past_mig_time_err.append(err)
# 
# past_comp_cost_result = list()
# past_comp_cost_err = list()
# 
# for num in past_total_comp_dict:
#     result, err = mean_confidence_interval(past_total_comp_dict[num], confidence=0.95)
#     past_comp_cost_result.append(result)
#     past_comp_cost_err.append(err)
# 
# pickle.dump(past_total_cost_result, open("past_total_cost_result.pickle", "wb"))
# pickle.dump(past_total_cost_err, open("past_total_cost_err.pickle", "wb"))
# 
# pickle.dump(past_comm_cost_result, open("past_comm_cost_result.pickle", "wb"))
# pickle.dump(past_comm_cost_err, open("past_comm_cost_err.pickle", "wb"))
# 
# pickle.dump(past_buff_cost_result, open("past_buff_cost_result.pickle", "wb"))
# pickle.dump(past_buff_cost_err, open("past_buff_cost_err.pickle", "wb"))
# 
# pickle.dump(past_mig_time_result, open("past_mig_time_result.pickle", "wb"))
# pickle.dump(past_mig_time_err, open("past_mig_time_err.pickle", "wb"))
# 
# pickle.dump(past_exec_time_result, open("past_exec_time_result.pickle", "wb"))
# pickle.dump(past_exec_time_err, open("past_exec_time_err.pickle", "wb"))
# 
# pickle.dump(past_comp_cost_result, open("past_comp_cost_result.pickle", "wb"))
# pickle.dump(past_comp_cost_err, open("past_comp_cost_err.pickle", "wb"))
# 
# print "past"
# print "past total cost:", past_total_cost_result
# print "past communication cost:", past_comm_cost_result
# print "past buffering cost:", past_buff_cost_result
# print "past migration time:", past_mig_time_result
# print "past execution time:", past_exec_time_result
# print "past computation cost:", past_comp_cost_result
# print "Fami:", past_number_of_fami





