from copy import deepcopy
from numpy import *
import networkx as net
import time

# set of flows

# set of access nodes

# set of edge nodes

ALPHA = 1
BETA = 10
GAMMA = 10


def _reform_flow_list(flow_dict):
    total_flow_list = list()
    for rtan, flow_list in flow_dict.items():
        total_flow_list.extend(flow_list)
    return total_flow_list


def split_path(path):
    split_list = list()
    for idx, val in enumerate(path):
        if (idx + 1) < len(path):
            nxt_val = path[idx + 1]
            split_list.append((val, nxt_val))
    return split_list


class Baseline(object):
    def __init__(self, timer, visited_service_instance, tan_list, edge_list, mig_flow_dict, graph):
        self.timer = timer
        self.vsins = visited_service_instance
        self.an = visited_service_instance['mapping']['access']
        self.en = visited_service_instance['mapping']['edge']
        self.tan_list = tan_list
        self.edge_list = edge_list
        self.graph = deepcopy(graph)
        self.flow_dict = mig_flow_dict
        self.flow_list = _reform_flow_list(mig_flow_dict)
        self.service_delay = visited_service_instance['type']['delay']
        self.spa = visited_service_instance['type']['para']

    def execute(self):
        total_comm_cost = 0
        total_buffering_cost = 0
        ten = self.en
        for etan in self.tan_list:
            if not net.has_path(self.graph, etan, ten):
                break
            comm_cost = self._mig_comm_cost(etan, ten)
            if comm_cost is None:
                print "Baseline - Try to determine number of failed migration"
                total_comm_cost = 0
                break
            total_comm_cost += comm_cost
        if total_comm_cost:
            total_cost = ALPHA*total_comm_cost/float(10**6) + BETA*total_buffering_cost/float(10**9)
            print 'Baseline'
            print "Total cost with baseline:", total_cost
            print "Total communication cost:", total_comm_cost/float(10**6)
            print "Total buffering cost:", total_buffering_cost/float(10**9)
            # print "All possible edges:", self.all_possible_edges()
        else:
            return None, None, None
        # time.sleep(3)
        return total_cost, total_comm_cost/float(10**6), total_buffering_cost/float(10**9)

    # there are at least 2 nodes
    def _path_latency_cos(self, path):
        cost = 0
        for idx, node in enumerate(path):
            if (idx + 1) < len(path):
                nxt_node = path[idx + 1]
                # print 'a rm link delay:', graph[node][nxt_node]['delay']
                cost += self.graph[node][nxt_node]['delay']
            else:
                break
        return cost

    def _orig_comm_cost(self):
        path = net.dijkstra_path(self.graph, self.an, self.en, 'delay')
        total_flow_rate = sum([flow['rate'] for flow in self.flow_list])
        return total_flow_rate*(len(path)-1)

    def _mig_comm_cost(self, stan, sten):
        path = net.dijkstra_path(self.graph, stan, sten, 'delay')
        flow_list = self.flow_dict[stan]
        total_flow_rate = sum([flow['rate'] for flow in flow_list])
        # should check whether the link is overloaded
        # print "Baseline - total rate:", total_flow_rate
        # print "Baseline - Path length:", len(path) - 1
        if (self._path_latency_cos(path) > self.service_delay) or self.is_link_overloaded(path, total_flow_rate, flow_list):
            return None
        return total_flow_rate*(len(path)-1)

    def is_link_overloaded(self, path, total_rate, flow_list):
        is_over = False
        path_list = split_path(path)
        for seg in path_list:
            curr_load = self._curr_link_node(seg)
            if total_rate + curr_load > self._link_cap(seg):
                is_over = True
                break
            else:
                src_node = seg[0]
                dst_node = seg[1]
                # for flow in flow_list:
                #     if flow in self.graph[src_node][dst_node]['load']:
                #         print "Baseline something wrong! Revise is_link_overloaded"
                #         time.sleep(10)
                # print "Base flow list:", flow_list
                # print "Current edge load:", self.graph[src_node][dst_node]['load']
                self.graph[src_node][dst_node]['load'].extend(flow_list)
        return is_over

    def _path_list(self, stan, sten):
        path = net.dijkstra_path(self.graph, stan, sten, 'delay')
        path_list = split_path(path)
        return path_list

    def _curr_link_node(self, edge):
        src_npde = edge[0]
        dst_node = edge[1]
        total_load = self.graph[src_npde][dst_node]['load']
        return sum([flow['rate'] for flow in total_load if flow['lifetime'] >= self.timer])

    def _est_link_node(self, edge, stan):
        vs_flow_list = self.flow_dict[stan]
        total_flow_rate = sum([flow['rate'] for flow in vs_flow_list])
        src_node = edge[0]
        dst_node = edge[1]
        total_load = self.graph[src_node][dst_node]['load']
        curr_load = sum([flow['rate'] for flow in total_load if flow['lifetime'] >= self.timer])
        return total_flow_rate + curr_load

    def _link_cap(self, edge):
        src_npde = edge[0]
        dst_node = edge[1]
        return self.graph[src_npde][dst_node]['bw']

    def buffering_cost(self, stan, sten):
        # transmission cost
        nflow = len(self.flow_dict[stan])
        a = self.spa[0]
        b = self.spa[1]
        trans_time = a + b*nflow             # in ms
        spath = net.dijkstra_path(self.graph, sten, self.en, weight='delay')
        prop_time = self._path_latency_cos(spath)
        total_time = trans_time + prop_time
        total_flow_rate = sum([flow['rate'] for flow in self.flow_dict[stan]])
        return total_flow_rate*total_time

    def total_mig_time(self, stan, sten):
        nflow = len(self.flow_dict[stan])
        a = self.spa[0]
        b = self.spa[1]
        trans_time = a + b * nflow  # in ms
        spath = net.dijkstra_path(self.graph, sten, self.en, weight='delay')
        prop_time = self._path_latency_cos(spath)
        total_time = trans_time + prop_time
        return total_time

    def state_transmission_time(self, stan, sten):
        nflow = len(self.flow_dict[stan])
        a = self.spa[0]
        b = self.spa[1]
        trans_time = a + b * nflow  # in ms
        return trans_time
    # def _path_bw_cost(self):

    # def _e2e_latency_constraint(self, e):
    #     src_node = e[0]
    #     dst_node = e[1]
    #     return self.graph[src_node][dst_node]['delay']

    def all_possible_edges(self):
        all_edge_list = list()
        for vtan in self.tan_list:
            for vten in self.edge_list:
                path_list = self._path_list(vtan, vten)
                for tp_edge in path_list:
                    tp_edge = (tp_edge[0], tp_edge[1])
                    rev_tp_egde = (tp_edge[1], tp_edge[0])
                    if tp_edge in all_edge_list or rev_tp_egde in all_edge_list:
                        continue
                    else:
                        all_edge_list.append(tp_edge)
        return all_edge_list

    def _link_bw_constraint(self, e):
        src_node = e[0]
        dst_node = e[1]
        return self.graph[src_node][dst_node]['bw']
