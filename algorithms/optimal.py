from gurobipy import Model, GRB
from numpy import *
import networkx as net
import time
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


# def new_sum(sum_list):
#     bit_sum = 0
#     for bit in sum_list:
#         bit_sum |= bit
#     return bit_sum


class Optimal(object):
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
        model = Model('State migration')
        # print 'Tan list', self.tan_list
        # print 'edge list', self.edge_list
        # Decision variables
        x = model.addVars(self.tan_list, self.edge_list, vtype=GRB.BINARY, name='x')               # edge selection constraint
        y = model.addVars(self.edge_list, vtype=GRB.BINARY, name='y')

        # Constraints
        model.addConstrs((sum(x[ctan, ten] for ten in self.edge_list) == 1 for ctan in self.tan_list), 'edge selection constraint')

        model.addConstrs((x[ctan, ten]*self._e2e_latency(ctan, ten) <= self.service_delay for ctan in self.tan_list for ten in self.edge_list), 'e2e service ')

        # model.addConstrs((x[ctan, ten]*self._est_link_node(edge, ctan) <= self._link_cap(edge) for ctan in self.tan_list for ten in self.edge_list for edge in self._path_list(ctan, ten)), 'link capacity constraint')

        model.addConstrs((self._curr_link_node(edge) + sum(x[ctan, ten]*self.edge_usage(edge, ctan, ten) for ctan in self.tan_list for ten in self.edge_list) <= self._link_cap(edge) for edge in self.all_possible_edges()), 'link capacity constraint')           # noqa

        # print "test mig cost:", self._mig_comm_cost(random.choice(self.tan_list), random.choice(self.edge_list))
        # print "Orig cost:", self._orig_comm_cost()
        # objectives

        # model.addConstrs((y[ten] == 1 if sum(x[otan, ten] for otan in self.tan_list) >= 1 else 0 for ten in self.edge_list), 'edge node constraint')

        model.addConstrs((y[ten] == any([x[otan, ten] for otan in self.tan_list]) for ten in self.edge_list), 'edge node constraint')


        # print 'original objective:', self._orig_comm_cost()
        # model.setObjective(sum(x[otan, ten]*self._mig_comm_cost(otan, ten) for otan in self.tan_list for ten in self.edge_list), GRB.MINIMIZE)

        # total_comm_cost =\
        #     (self._orig_comm_cost() - sum(x[otan, ten]*self._mig_comm_cost(otan, ten) for otan in self.tan_list for ten in self.edge_list))/float(10**6)   # in Mbps

        total_comm_cost = sum(
                x[otan, ten] * self._mig_comm_cost(otan, ten) for otan in self.tan_list for ten in
                self.edge_list) / float(10 ** 6)  # in Mbps

        total_state_transfer_cost = sum(
            x[stan, sten]*self.buffering_cost(stan, sten) for stan in self.tan_list for sten in self.edge_list)/float(10**9)   # in Mb (b --> Mb, ms --> s)

        # model.setObjective(total_comm_cost - total_state_transfer_cost, GRB.MAXIMIZE)

        total_comp_cost = sum(y[ten]*self.is_orig(ten) for ten in self.edge_list)

        model.setObjective(ALPHA*total_comm_cost + BETA*total_state_transfer_cost + GAMMA*total_comp_cost, GRB.MINIMIZE)


        # model.setObjective(total_comm_cost, GRB.MAXIMIZE)

        # model.setObjective(total_state_transfer_cost, GRB.MAXIMIZE)

        model.optimize()

        if model.solCount == 0:
            print("Model is infeasible")

        obj = model.getObjective()
        #
        res_total_comm_cost = \
            (self._orig_comm_cost() - sum(
                getattr(x[otan, ten], 'X') * self._mig_comm_cost(otan, ten) for otan in self.tan_list for ten in
                self.edge_list)) / float(10 ** 6)  # in Mbps

        res_mig_comm_cost = sum(getattr(x[otan, ten], 'X') * self._mig_comm_cost(otan, ten) for otan in self.tan_list for ten in self.edge_list)/float(10 ** 6)  # in Mbps

        res_total_state_transfer_cost =\
            sum(getattr(x[stan, sten], 'X')*self.buffering_cost(stan, sten) for stan in self.tan_list for sten in self.edge_list)/float(10**9)   # in Mb (b --> Mb, ms --> s)

        res_mig_time = \
            sum(getattr(x[stan, sten], 'X') * self.total_mig_time(stan, sten) for stan in self.tan_list for sten in
                self.edge_list)

        # res_total_comp_cost = sum(
        #     sum(getattr(x[stan, sten], 'X') for stan in self.tan_list) for sten in self.edge_list
        # )

        res_total_comp_cost = sum(getattr(y[ten], 'X') * self.is_orig(ten) for ten in self.edge_list)

        res_state_trans_time = \
            sum(getattr(x[stan, sten], 'X') * self.state_transmission_time(stan, sten) for stan in self.tan_list for sten in
                self.edge_list)

        print "Optimal"
        print 'Total cost:', obj.getValue()
        #
        # print 'Total comm cost:', res_total_comm_cost
        # print 'Orig cost:', self._orig_comm_cost()/float(10**6)
        print "Comm cost:", res_mig_comm_cost
        #
        print 'Buffering cost:', res_total_state_transfer_cost

        print "Comp cost:", res_total_comp_cost
        # print 'Total migration time:', res_mig_time
        # print 'State Transmission time:', res_state_trans_time

        # print 'Benefit:', self._orig_comm_cost() - obj.getValue()
        # for v in model.getVars():
        #     print('%s %g' % (v.varName, v.x))
        # time.sleep(2)

        return obj.getValue(), res_mig_comm_cost, res_total_state_transfer_cost, res_mig_time, res_total_comp_cost

    def is_orig(self, ten):
        return 0 if (ten == self.en) else 1

    def _e2e_latency(self, stan, sten):
        if not net.has_path(self.graph, stan, sten):
            return 10**10
        path = net.dijkstra_path(self.graph, stan, sten, 'delay')
        return self._path_latency_cos(path)

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
        if not net.has_path(self.graph, stan, sten):
            return list()
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
        if sten == self.en:
            return 0
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
        if sten == self.en:
            return 0
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

    def edge_usage(self, edge, stan, sten):
        src_node = edge[0]
        dst_node = edge[1]
        tp_edge = (src_node, dst_node)
        rev_tp_edge = (dst_node, src_node)
        path_list = self._path_list(stan, sten)
        if tp_edge in path_list or rev_tp_edge in path_list:
            return sum([flow['rate'] for flow in self.flow_dict[stan]])
        else:
            return 0

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

    # def _path_bw_cost(self):

    # def _e2e_latency_constraint(self, e):
    #     src_node = e[0]
    #     dst_node = e[1]
    #     return self.graph[src_node][dst_node]['delay']

    def _link_bw_constraint(self, e):
        src_node = e[0]
        dst_node = e[1]
        return self.graph[src_node][dst_node]['bw']
