"""
   Bandwith-aware State Transfer (BAST)
"""
from numpy import *
import networkx as net
import time
from collections import OrderedDict
from copy import deepcopy

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


class Bast(object):
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
        self.ten_list = list()

    def execute(self):
        print "Bast orig edge node:", self.en
        total_cost, buff_cost, total_mig_time, total_comp_cost = self.find_ten()
        if not total_cost:
            print "Bast - Try to determine number of failed migration"
            return None, None, None, None, None
        all_cost = ALPHA*(total_cost/float(10**6)) + BETA*(buff_cost/float(10**9)) + GAMMA*total_comp_cost
        print 'Bast'
        print "Total cost with Bast:", all_cost
        print "Communication cost:", total_cost/float(10**6)
        print "Buffering cost:", buff_cost/float(10**9)
        print "Comp cost:", total_comp_cost
        print "Orig en:", self.en
        print "Ten list:", self.ten_list
        print "TAN list:", self.tan_list
        # time.sleep(3)
        return all_cost, total_cost/float(10**6), buff_cost/float(10**9), total_mig_time, total_comp_cost

    def find_ten(self):
        # find ten list that is satisfied with e2e latency constraint
        total_comm_cost = 0
        total_buff_cost = 0
        total_mig_time = 0
        total_comp_cost = 0
        for ctan in self.tan_list:
            cten, comm_cost, buff_cost, total_time, comp_cost = self._find_ten(ctan)
            if cten is None:
                print "Something wrong or link overload! Revise BAST"
                total_comm_cost = 0
                total_buff_cost = 0
                break
            total_comm_cost += comm_cost
            total_buff_cost += buff_cost
            total_mig_time += total_time
            total_comp_cost += comp_cost
            print "BAST Comm Path:", net.dijkstra_path(self.graph, ctan, cten, weight='delay')

        return total_comm_cost, total_buff_cost, total_mig_time, total_comp_cost

    def _find_ten(self, vtan):
        final_target = None
        buff_cost = 0
        comm_cost = 0
        total_time = 0
        comp_cost = 0
        total_rate = sum([flow['rate'] for flow in self.flow_dict[vtan]])
        # print 'Bast - total rate:', total_rate
        visited_edge_dict = OrderedDict()
        trial_dict = OrderedDict()
        for vten in self.edge_list:
            if not net.has_path(self.graph, vtan, vten):
                continue
            if not net.has_path(self.graph, vten, self.en):
                continue
            path = net.dijkstra_path(self.graph, vtan, vten, weight='delay')
            cost = self._path_latency_cos(path)
            if (cost > self.service_delay) or self.is_link_overloaded(path, total_rate):           # constraints   # noqa
                continue
            visited_edge_dict[vten] = OrderedDict()  # remove it afterward
            visited_edge_dict[vten]['path'] = path
            visited_edge_dict[vten]['cost'] = cost
            trial_dict[vten] = len(path)
        if trial_dict:
            final_target = min(trial_dict, key=trial_dict.get)
            path_len = trial_dict[final_target]
            comm_cost = total_rate * (path_len - 1)
            if final_target == self.en:
                buff_cost = 0
                total_time = 0
                comp_cost = 0
            else:
                buff_cost, total_time = self.buffering_cost(vtan, final_target)
                comp_cost = 0 if final_target in self.ten_list else 1
            self.update_graph(vtan, visited_edge_dict[final_target]['path'])
            self.ten_list.append(final_target)

        # print "Bast - target", final_target, comm_cost
        if not trial_dict:
            print 'Bast: failed to compromise latency constraint and link capacity'
        return final_target, comm_cost, buff_cost, total_time, comp_cost

    def is_link_overloaded(self, path, total_rate):
        is_over = False
        path_list = split_path(path)
        for seg in path_list:
            curr_load = self._curr_link_node(seg)
            if total_rate + curr_load > self._link_cap(seg):
                is_over = True
                break
        return is_over

    def update_graph(self, vtan, path):
        path_list = split_path(path)
        for seg in path_list:
                src_node = seg[0]
                dst_node = seg[1]
                # print "Src and dst", src_node, dst_node
                # print self.graph[src_node][dst_node]['load']
                self.graph[src_node][dst_node]['load'].extend(self.flow_dict[vtan])
                
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
        return total_flow_rate*(len(path)-1)

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

    # def buffering_cost(self, stan, prop_time):
    #     # transmission cost
    #     nflow = len(self.flow_dict[stan])
    #     a = self.spa[0]
    #     b = self.spa[1]
    #     trans_time = a + b * nflow  # in ms
    #     total_time = trans_time + prop_time
    #     total_flow_rate = sum([flow['rate'] for flow in self.flow_dict[stan]])
    #     return (total_flow_rate * total_time), total_time

    def buffering_cost(self, stan, sten):
        # transmission cost
        nflow = len(self.flow_dict[stan])
        a = self.spa[0]
        b = self.spa[1]
        trans_time = a + b*nflow             # in ms
        spath = net.dijkstra_path(self.graph, sten, self.en, weight='delay')
        print "Bast state path:", spath
        prop_time = self._path_latency_cos(spath)
        total_time = trans_time + prop_time
        total_flow_rate = sum([flow['rate'] for flow in self.flow_dict[stan]])
        return total_flow_rate*total_time, total_time

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

    def _link_bw_constraint(self, e):
        src_node = e[0]
        dst_node = e[1]
        return self.graph[src_node][dst_node]['bw']
