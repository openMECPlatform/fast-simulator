"""
   Proportional State Transfer (Past): choose the edge node that has lowest k = buff_cost/(Max_comm - curr_comm)
   Use Tabu search
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
BETA = 1

LOOP_ITER_MAX = 60
TABU_ITER_MAX = 100


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


class Past(object):
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
        self.tabu_list = list()

    # def execute(self):
    #     total_cost, buff_cost, total_mig_time = self.find_ten()
    #     if not total_cost:
    #         print "Bast - Try to determine number of failed migration"
    #         return None, None, None
    #     all_cost = ALPHA*(total_cost / float(10 ** 6)) + BETA*(buff_cost / float(10 ** 9))
    #     print 'Past'
    #     print "Total cost:", all_cost
    #     print "Communication cost:", total_cost / float(10 ** 6)
    #     print "Buffering cost:", buff_cost / float(10 ** 9)
    #     # time.sleep(3)
    #     return all_cost, total_cost / float(10 ** 6), buff_cost / float(10 ** 9), total_mig_time

    def execute(self):
        # This method returns initial ns_dict and initial cost
        init_candidate, init_cost, init_comm_cost, init_buff_cost, init_total_mig_time = self.initial_algorithm()
        print 'initial cost:', init_cost
        if not init_candidate:
            print "Algorithm failed at first step."
            return None, None, None, None
        curr_solution = deepcopy(init_candidate)
        final_best_cost = init_cost
        final_comm_cost = init_comm_cost
        final_buff_cost = init_buff_cost
        final_total_mig_time = init_total_mig_time
        # print final_best_cost
        # final_best_candidate = copy.deepcopy(init_candidate)
        loop_index = 0
        while loop_index < LOOP_ITER_MAX:
            # match_dict is a move
            mapping_dict, move, total_cost, comm_cost, buff_cost, total_mig_time = self.find_neighborhood_solution(curr_solution)
            if mapping_dict is None:
                # print 'Tabu: I am None!!!'
                # print 'Loop index:', loop_index
                # import time
                # time.sleep(5)
                loop_index = loop_index + 1
                continue
            # bst_cost = solution_info_dict['total_cost']
            if move in self.tabu_list:
                # override if meet aspiration condition
                if total_cost < final_best_cost:
                    # this makes sure there is only one mapping vnf_index:node_index in tabu list
                    # print 'Stop by to see tabu list decreases'
                    # import time
                    # time.sleep(10)
                    self.tabu_list.remove(move)
            else:
                if (len(self.tabu_list) + 1) < TABU_ITER_MAX:
                    self.tabu_list.append(move)    # {nf:node} is a move
                else:
                    print 'Tabu++: Break due to over tabu list'
                    # import time
                    # time.sleep(5)
                    break        # stop if tabu list size exceeds TABU_ITER_MAX
            loop_index = loop_index + 1
            # print loop_index
            if total_cost < final_best_cost:
                print "Tabu worked"
                # time.sleep(5)
                final_best_cost = total_cost
                # final_best_candidate = copy.deepcopy(mapping_dict)
                final_comm_cost = comm_cost
                final_buff_cost = buff_cost
                final_total_mig_time = total_mig_time
                loop_index = 0
        print "=========="
        print 'Tabu list:', len(self.tabu_list)
        # should return best candidate
        print 'Past'
        print "Total cost:", final_best_cost
        print "Communication cost:", final_comm_cost
        print "Buffering cost:", final_buff_cost
        return final_best_cost, final_comm_cost, final_buff_cost, final_total_mig_time

    def initial_algorithm(self):
        mapping_dict = OrderedDict()
        total_comm_cost = 0
        total_buff_cost = 0
        total_mig_time = 0
        trial_graph = deepcopy(self.graph)
        for ctan in self.tan_list:
            cten, comm_cost, buff_cost, total_time = self._find_ten(ctan, trial_graph)
            if cten is None:
                print "Something wrong or link overload! Revise PAST"
                return None, None, None, None, None
            total_comm_cost += comm_cost
            total_buff_cost += buff_cost
            total_mig_time += total_time
            mapping_dict[ctan] = cten
        all_cost = ALPHA * (total_comm_cost / float(10 ** 6)) + BETA * (total_buff_cost / float(10 ** 9))
        return mapping_dict, all_cost, (total_comm_cost / float(10 ** 6)), (total_buff_cost / float(10 ** 9)), total_mig_time

    def find_neighborhood_solution(self, mapping_dict):
        move = OrderedDict()
        trial_graph = deepcopy(self.graph)
        mtan = random.choice(mapping_dict.keys())
        new_mapping_dict = deepcopy(mapping_dict)
        mten = self.randomize_ten(trial_graph, mtan)
        if mten is None:
            return None, None, None, None, None, None
        new_mapping_dict[mtan] = mten
        total_cost, total_comm_cost, total_buff_cost, total_mig_time = self.total_cal_cost(trial_graph, new_mapping_dict)
        if not total_cost:
            # print "not total cost at all"
            return None, None, None, None, None, None
        move[mtan] = mten
        return new_mapping_dict, move, total_cost, (total_comm_cost / float(10 ** 6)), (total_buff_cost / float(10 ** 9)), total_mig_time

    # def find_neighborhood_solution_with_en(self, mapping_dict):
    #     move = OrderedDict()
    #     trial_graph = self.graph.copy()
    #     mten = random.choice(self.en)
    #     new_mapping_dict = copy.deepcopy(mapping_dict)
    #     mten = self.randomize_ten(trial_graph, mtan)
    #     if mten is None:
    #         return None, None, None, None, None, None
    #     new_mapping_dict[mtan] = mten
    #     total_cost, total_comm_cost, total_buff_cost, total_mig_time = self.total_cal_cost(trial_graph, new_mapping_dict)
    #     if not total_cost:
    #         # print "not total cost at all"
    #         return None, None, None, None, None, None
    #     move[mtan] = mten
    #     return new_mapping_dict, move, total_cost, (total_comm_cost / float(10 ** 6)), (total_buff_cost / float(10 ** 9)), total_mig_time_

    def total_cal_cost(self, graph, mapping_dict):
        total_comm_cost = 0
        total_buff_cost = 0
        total_mig_time = 0
        for mtan, mten in mapping_dict.items():
            # print "tan-ten before:", mtan, mten
            comm_cost, buff_cost, total_time = self.mapped_cal_cost(graph, mtan, mten)
            if not comm_cost:
                # print "not comm cost at all:", comm_cost
                # print "tan-ten:", mtan, mten
                return 0, 0, 0, 0
            total_comm_cost += comm_cost
            total_buff_cost += buff_cost
            total_mig_time += total_time
        total_cost = ALPHA * (total_comm_cost / float(10 ** 6)) + BETA * (total_buff_cost / float(10 ** 9))
        return total_cost, total_comm_cost, total_buff_cost, total_mig_time

    def mapped_cal_cost(self, graph, mtan, mten):
        total_rate = sum([flow['rate'] for flow in self.flow_dict[mtan]])
        path = net.dijkstra_path(self.graph, mtan, mten, weight='delay')
        cost = self._path_latency_cos(path)
        if (cost > self.service_delay) or self.is_link_overloaded(graph, path, total_rate):
            return 0, 0, 0
        comm_cost = total_rate * (len(path) - 1)
        # print "Comm cost:", comm_cost
        # print "verified tan-ten:", mtan, mten
        if mten == self.en:
            buff_cost = 0
            total_time = 0
        else:
            buff_cost, total_time = self.buffering_cost(mtan, mten)
        self.update_graph(graph, mtan, path)
        return comm_cost, buff_cost, total_time

    def randomize_ten(self, graph, rtan):
        total_rate = sum([flow['rate'] for flow in self.flow_dict[rtan]])
        ten_list = list()
        for rten in self.edge_list:
            if not net.has_path(self.graph, rtan, rten):
                continue
            if not net.has_path(self.graph, rten, self.en):
                continue
            path = net.dijkstra_path(self.graph, rtan, rten, weight='delay')
            cost = self._path_latency_cos(path)
            if (cost > self.service_delay) or self.is_link_overloaded(graph, path, total_rate):
                continue
            ten_list.append(rten)
        if not ten_list:
            return None
        return random.choice(ten_list)

    def _find_ten(self, vtan, tgraph):
        final_target = None
        comm_cost = 0
        buff_cost = 0
        total_time = 0
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
            # print "LAST cost:", cost
            if (cost > self.service_delay) or self.is_link_overloaded(tgraph, path, total_rate):         # constraints
                # print "I am here!"
                # time.sleep(10)
                continue
            visited_edge_dict[vten] = path
            trial_dict[vten] = cost
        if visited_edge_dict:
            final_target = random.choice(trial_dict.keys())
            lat_cost = trial_dict[final_target]
            if final_target == self.en:
                buff_cost = 0
                total_time = 0
            else:
                buff_cost, total_time = self.buffering_cost(vtan, final_target)
            # print "LAST path:", test_dict[final_target]
            # print "LAST path list", path_dict[final_target]
            comm_cost = total_rate * (len(visited_edge_dict[final_target]) - 1)
            # print "LAST path:", test_dict[final_target],
            self.update_graph(tgraph, vtan, visited_edge_dict[final_target])
        return final_target, comm_cost, buff_cost, total_time

    def is_link_overloaded(self, graph, path, total_rate):
        is_over = False
        path_list = split_path(path)
        for seg in path_list:
            curr_load = self._curr_link_node(graph, seg)
            if total_rate + curr_load > self._link_cap(seg):
                is_over = True
                break
        return is_over

        # there are at least 2 nodes

    def update_graph(self, graph, vtan, path):
        path_list = split_path(path)
        for seg in path_list:
                src_node = seg[0]
                dst_node = seg[1]
                graph[src_node][dst_node]['load'].extend(self.flow_dict[vtan])

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
        return total_flow_rate * (len(path) - 1)

    def _mig_comm_cost(self, stan, sten):
        path = net.dijkstra_path(self.graph, stan, sten, 'delay')
        flow_list = self.flow_dict[stan]
        total_flow_rate = sum([flow['rate'] for flow in flow_list])
        return total_flow_rate * (len(path) - 1)

    def _path_list(self, stan, sten):
        path = net.dijkstra_path(self.graph, stan, sten, 'delay')
        path_list = split_path(path)
        return path_list

    def _curr_link_node(self, graph, edge):
        src_npde = edge[0]
        dst_node = edge[1]
        total_load = graph[src_npde][dst_node]['load']
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