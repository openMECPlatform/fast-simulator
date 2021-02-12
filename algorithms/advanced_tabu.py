# Advanced Tabu includes 2 phases:
# Phase 1: find the largest sub-chain to reuse
# Phase 2: apply Tabu search
# Tabu KPI for selection is the Paid factor
from collections import OrderedDict
import networkx as nx
from numpy import random
import uuid
import copy

# Set weight for problem optimization
ALPHA = 0.6  # weight for computation cost
GAMMA = 0.6   # chain config cost should be greater than routing cost
BETA = 0.4  # weight for routing cost
DELTA = 0.4  # weight for reliability cost
TABU_ITER_MAX = 300   # Tabu size: stop after algorithm reaches this size
LOOP_ITER_MAX = 100   # Number of iterations is executed for each tabu search


class AdvTabu(object):
    def __init__(self, nf_prop, req_dict, graph, sys_ns_dict, timer):
        self.graph = graph
        # self.ns_dict = system_ns_dict
        self.nf_prop = nf_prop
        self.req_dict = req_dict
        self.tabu_list = list()
        self.sfc_dict = self.req_dict['sfc']
        self.timer = timer
        self.req_id = uuid.uuid4()
        self.req_info = self.req_dict['info']
        self.req_requirements = self.req_info['requirements']
        self.e2e_path = list()
        self.sys_ns_dict = sys_ns_dict
        self.NSlen = len(self.sfc_dict)
        self.inst_mapping = OrderedDict()    # used for reuse, this is later used to update graph
        self.shared_path_dict = OrderedDict()

    def execute_tabu(self):
        # This method returns initial ns_dict and initial cost
        init_candidate, init_cost = self.find_first_solution(strategy='random')
        print 'orig-cost:', init_cost
        if init_candidate is None:
            print "Algorithm failed at first step."
            return None, None
        self.e2e_path = init_cost['detailed_path'][:]
        # print 'first path', self.e2e_path
        # print 'first candidate', init_candidate
        curr_solution = copy.deepcopy(init_candidate)
        final_best_cost = init_cost['total_cost']
        # print final_best_cost
        final_best_candidate = copy.deepcopy(init_candidate)
        final_best_result_dict = copy.deepcopy(init_cost)
        # print 'First result'
        # print final_best_candidate
        # print final_best_cost
        loop_index = 0
        while loop_index < LOOP_ITER_MAX:
            # match_dict is a move
            match_dict, bst_candidate, solution_info_dict = self.find_best_neighborhood(curr_solution, policy='random')
            # match_tpl = (vnf_index, node_index, instance_index)
            # print 'end tracking'
            if bst_candidate is None:
                print 'Tabu++: I am None!!!'
                print 'Loop index:', loop_index
                # import time
                # time.sleep(5)
                loop_index = loop_index + 1
                continue
            bst_cost = solution_info_dict['total_cost']

            # print 'new-cost  curr-cost init-cost:', bst_cost, final_best_cost, solution_info_dict

            # print bst_candidate, solution_info_dict['detailed_path']
            # bst_candidate, bst_cost = self.find_best_neighborhood(curr_solution, policy='paid')
            # A solution belong to the current visiting VNF back to its original physical node
            if self.in_tabu_list(match_dict):   # Tabu list is a list of vnf_index:node:instance
                # override if meet aspiration condition
                if bst_cost < final_best_cost:
                    # this makes sure there is only one mapping vnf_index:node_index in tabu list
                    # print 'Stop by to see tabu list decreases'
                    # import time
                    # time.sleep(10)
                    self.tabu_list_remove(match_dict)
            else:
                if (len(self.tabu_list) + 1) < TABU_ITER_MAX:
                    self.tabu_list.append(match_dict)    # {nf:node} is a move
                else:
                    print 'Tabu++: Break due to over tabu list'
                    # import time
                    # time.sleep(5)
                    break        # stop if tabu list size exceeds TABU_ITER_MAX
            loop_index = loop_index + 1
            # print loop_index
            if bst_cost < final_best_cost:
                final_best_cost = bst_cost
                final_best_candidate = copy.deepcopy(bst_candidate)
                final_best_result_dict = copy.deepcopy(solution_info_dict)
                self.e2e_path = solution_info_dict['detailed_path'][:]
                curr_solution = copy.deepcopy(bst_candidate)
                loop_index = 0
                # if policy is paid
                # print "Tabu++ worked here"
                # import time
                # time.sleep(10)

                # print bst_cost
                # print 'Inner final path', solution_info_dict['detailed_path']
                # print 'Inner final candidate', final_best_candidate
        print "=========="
        print 'Tabu++ list:', len(self.tabu_list)
        # should return best candidate
        return final_best_candidate, final_best_result_dict

    # Find first solution: apply local search for this case (one-by-one)
    def find_first_solution(self, strategy):
        # total_cost is calculated from the problem formulation
        solution_info_dict = OrderedDict()
        solution_info_dict['total_cost'] = 0
        solution_info_dict['config_cost'] = 0
        solution_info_dict['routing_cost'] = 0
        solution_info_dict['config_cost'] = 0
        solution_info_dict['comp_cost'] = 0
        solution_info_dict['rel_cost'] = 1
        solution_info_dict['detailed_path'] = list()
        curr_solution = OrderedDict()
        est_graph = copy.deepcopy(self.graph)
        # node:{vnf_index: instance_id} if reuse, else: node:{vnf_index: None}
        if strategy == "greedy":
            for index, nf_index in enumerate(self.sfc_dict.keys()):
                src_dict = OrderedDict()
                if index:
                    prev_vnf = self.sfc_dict.keys()[index-1]
                    src_dict[prev_vnf] = copy.deepcopy(curr_solution[prev_vnf])
                # print 'VNF at first', nf_index
                # print 'Src dict at first', src_dict
                # import time
                # time.sleep(3)
                node_candidate = list()
                for node in est_graph.nodes():
                    if nf_index in est_graph.node[node]['allowed_vnf_list']:
                        # curr_node_cap = self.graph.node[node]['curr_load']
                        # total_node_cap = self.graph.node[node]['cpu']
                        # if (vnf_load + curr_node_cap) <= total_node_cap:
                        node_candidate.append(node)

                # Run comp cost function
                comp_cost_dict, config_cost_dict, match_dict = self.pre_comp_config_cost_func(nf_index, src_dict, node_candidate, est_graph)
                routing_cost_dict, path_dict = self.routing_cost_func(node_candidate[:], curr_solution, est_graph)
                # print 'Routing cost at first', routing_cost_dict
                rel_cost_dict = self.rel_cost_func(nf_index, node_candidate)
                # print 'Reliability cost', rel_cost_dict
                # print 'NS system', self.sys_ns_dict
                local_node_candidate = OrderedDict()
                for node in node_candidate:
                    if routing_cost_dict.get(node) is None or comp_cost_dict.get(node) is None:
                        continue
                    local_node_candidate[node] = ALPHA*comp_cost_dict[node] + BETA*routing_cost_dict[node] + GAMMA*config_cost_dict[node] + DELTA*(1 - rel_cost_dict[node])   # noqa
                if not local_node_candidate:
                    print 'Tabu++: What is the fault reason:', local_node_candidate
                    print 'At VNF-th', nf_index
                    print 'Current solution', curr_solution
                    print 'Tabu++: routing cost', routing_cost_dict
                    print 'Tabu++: comp cost', comp_cost_dict
                    import time
                    time.sleep(10)
                    return None, None
                else:
                    # print 'Total cost at first', local_node_candidate
                    min_total_cost = min([cost for node, cost in local_node_candidate.items()])
                    candidate_list = [node for node, cost in local_node_candidate.items() if cost == min_total_cost]
                    final_candidate = candidate_list[0]
                    curr_solution[nf_index] = {final_candidate: match_dict[final_candidate]}
                    solution_info_dict['total_cost'] = solution_info_dict['total_cost'] + min_total_cost
                    solution_info_dict['config_cost'] = solution_info_dict['config_cost'] + config_cost_dict[final_candidate]
                    solution_info_dict['routing_cost'] = solution_info_dict['routing_cost'] + routing_cost_dict[final_candidate]
                    solution_info_dict['comp_cost'] = solution_info_dict['comp_cost'] + comp_cost_dict[final_candidate]
                    solution_info_dict['rel_cost'] = solution_info_dict['rel_cost'] * rel_cost_dict[final_candidate]
                    solution_info_dict['detailed_path'].extend(path_dict[final_candidate])

                    self.update_graph({nf_index: {final_candidate: copy.deepcopy(match_dict[final_candidate])}}, est_graph, path_dict[final_candidate])      # noqa

            # Calculate the chain configuration cost here:
            # curr_solution is the mapping between vnf_index and node_index
        if strategy == 'random':
            for index, nf_index in enumerate(self.sfc_dict.keys()):
                src_dict = OrderedDict()
                if index:
                    prev_vnf = self.sfc_dict.keys()[index - 1]
                    src_dict[prev_vnf] = copy.deepcopy(curr_solution[prev_vnf])
                    print 'prev_node_dict', src_dict[prev_vnf]
                # print 'VNF at first', nf_index
                # print 'Src dict at first', src_dict
                # import time
                # time.sleep(3)
                node_candidate = list()
                for node in est_graph.nodes():
                    if nf_index in est_graph.node[node]['allowed_vnf_list']:
                        # curr_node_cap = self.graph.node[node]['curr_load']
                        # total_node_cap = self.graph.node[node]['cpu']
                        # if (vnf_load + curr_node_cap) <= total_node_cap:
                        node_candidate.append(node)

                rdm_node_candidate = [random.choice(node_candidate)]
                # print 'tabu++: stop here...', rdm_node_candidate, node_candidate
                # import time
                # time.sleep(3)

                # Run comp cost function
                comp_cost_dict, config_cost_dict, match_dict = self.pre_comp_config_cost_func(nf_index, src_dict,
                                                                                              rdm_node_candidate, est_graph)

                # routing cost here did not include from MEA node
                routing_cost_dict, path_dict = self.routing_cost_func(rdm_node_candidate[:], curr_solution, est_graph)
                rel_cost_dict = self.rel_cost_func(nf_index, rdm_node_candidate)
                local_node_candidate = OrderedDict()
                for node in rdm_node_candidate:
                    if routing_cost_dict.get(node) is None or comp_cost_dict.get(node) is None:
                        continue
                    local_node_candidate[node] = ALPHA * comp_cost_dict[node] + BETA * routing_cost_dict[node] + GAMMA * config_cost_dict[node] + DELTA * (1 - rel_cost_dict[node])  # noqa
                if not local_node_candidate:
                    print 'Tabu++: What is the fault reason:', local_node_candidate
                    print 'At VNF-th', nf_index
                    print 'Current solution', curr_solution
                    print 'Tabu++: routing cost', routing_cost_dict
                    print 'Tabu++: comp cost', comp_cost_dict
                    import time
                    time.sleep(10)
                    return None, None
                else:
                    final_candidate = random.choice(local_node_candidate.keys())
                    exp_total_cost = local_node_candidate[final_candidate]
                    curr_solution[nf_index] = {final_candidate: match_dict[final_candidate]}
                    solution_info_dict['total_cost'] = solution_info_dict['total_cost'] + exp_total_cost
                    solution_info_dict['config_cost'] = solution_info_dict['config_cost'] + config_cost_dict[final_candidate]
                    solution_info_dict['routing_cost'] = solution_info_dict['routing_cost'] + routing_cost_dict[final_candidate]
                    solution_info_dict['comp_cost'] = solution_info_dict['comp_cost'] + comp_cost_dict[final_candidate]
                    solution_info_dict['rel_cost'] = solution_info_dict['rel_cost'] * rel_cost_dict[final_candidate]
                    solution_info_dict['detailed_path'].extend(path_dict[final_candidate])

                    prev_node_dict = {final_candidate: match_dict[final_candidate]}
                    # print 'First update graph', prev_node_dict
                    self.update_graph({nf_index: {final_candidate: copy.deepcopy(match_dict[final_candidate])}}, est_graph, path_dict[final_candidate])  # noqa

        return curr_solution, solution_info_dict

    def post_cal_total_cost(self, new_solution):
        solution_info_dict = OrderedDict()
        solution_info_dict['total_cost'] = 0
        solution_info_dict['config_cost'] = 0
        solution_info_dict['routing_cost'] = 0
        solution_info_dict['config_cost'] = 0
        solution_info_dict['comp_cost'] = 0
        solution_info_dict['rel_cost'] = 1
        solution_info_dict['detailed_path'] = list()
        curr_solution = OrderedDict()
        bst_graph = copy.deepcopy(self.graph)
        for index, vnf_index in enumerate(new_solution.keys()):
            # pnode = next(iter(new_solution[vnf_index]))
            # pins = new_solution[vnf_index][pnode]     # pins can be None
            src_dict = OrderedDict()
            if index:
                prev_vnf = self.sfc_dict.keys()[index - 1]
                src_dict[prev_vnf] = copy.deepcopy(curr_solution[prev_vnf])
            node_candidate_dict = new_solution[vnf_index]
            node_candidate = node_candidate_dict.keys()
            pnode = node_candidate[0]
            comp_cost_dict, config_cost_dict = self.post_comp_config_cost_func(vnf_index, src_dict, node_candidate_dict, bst_graph)
            routing_cost_dict, path_dict = self.routing_cost_func(node_candidate, curr_solution, bst_graph)
            # comp_cost is None when running out of CPU resources
            # routing_cost is None when running out of BW
            if comp_cost_dict.get(pnode) is None or routing_cost_dict.get(pnode) is None:
                return None          # also means that loop will be automatically broken
            rel_cost_dict = self.rel_cost_func(vnf_index, node_candidate)
            curr_solution[vnf_index] = node_candidate_dict
            curr_cost = ALPHA*comp_cost_dict[pnode] + BETA*routing_cost_dict[pnode] + GAMMA*config_cost_dict[pnode] + DELTA*(1 - rel_cost_dict[pnode])     # noqa
            solution_info_dict['total_cost'] = solution_info_dict['total_cost'] + curr_cost
            solution_info_dict['config_cost'] = solution_info_dict['config_cost'] + config_cost_dict[pnode]
            solution_info_dict['routing_cost'] = solution_info_dict['routing_cost'] + routing_cost_dict[pnode]
            solution_info_dict['comp_cost'] = solution_info_dict['comp_cost'] + comp_cost_dict[pnode]
            solution_info_dict['rel_cost'] = solution_info_dict['rel_cost'] * rel_cost_dict[pnode]
            solution_info_dict['detailed_path'].extend(path_dict[pnode])

            self.update_graph({vnf_index: copy.deepcopy(node_candidate_dict)}, bst_graph, path_dict[pnode])
        # Calculate the config_cost here
        # solution_info_dict['config_cost'] = self.chain_config_cost(visited_solution, self.sys_ns_dict)
        # Calculate final cost
        # solution_info_dict['total_cost'] = solution_info_dict['total_cost'] + GAMMA*solution_info_dict['config_cost']
        return solution_info_dict

    def find_match(self, orig_solution, visited_vnf, visited_node):
        visited_solution = copy.deepcopy(orig_solution)
        prev_node_dict = visited_solution[visited_vnf]
        curr_node_load = self.graph.node[visited_node]['curr_load']
        total_node_cap = self.graph.node[visited_node]['cpu']
        if self.graph.node[visited_node]['instances'].get(visited_vnf) is None:
            if {visited_node: None} != prev_node_dict:
                if (self.nf_prop['proc_cap'][visited_vnf] + curr_node_load) > total_node_cap:
                    return None
                else:
                    # print 'choose an empty node'
                    visited_solution[visited_vnf] = {visited_node: None}
            # return None
        else:
            local_inst_list = list()
            for inst_id, inst_list in self.graph.node[visited_node]['instances'][visited_vnf].items():
                if {visited_node: inst_id} != prev_node_dict:
                    total_load = sum([inst_info_dict['req_load'] for inst_info_dict in inst_list if
                                      inst_info_dict['lifetime'] >= self.timer])
                    if self.req_requirements['proc_cap'] + total_load <= self.nf_prop['proc_cap'][visited_vnf]:
                        local_inst_list.append(inst_id)
            if not local_inst_list:
                if (self.nf_prop['proc_cap'][visited_vnf] + curr_node_load) > total_node_cap:
                    return None
                else:
                    if {visited_node: None} != prev_node_dict:
                        visited_solution[visited_vnf] = {visited_node: None}
                    else:
                        return None
                # return None
            else:
                print 'Tabu++ for best solution: Instance with visited node is found'
                visited_solution[visited_vnf] = {visited_node: random.choice(local_inst_list)}

        return visited_solution

    # apply proposed Tabu strategies here
    # There are 2 main strategies: randomly pick VNF and randomly pick node candidate
    # def find_best_neighborhood(self, curr_solution, policy):
    #     bst_cost_dict = OrderedDict()
    #     picked_vnf = None
    #     picked_index = None
    #     # apply random VNF first
    #     if policy == 'random':
    #         picked_vnf = random.choice(self.sfc_dict.keys())
    #
    #     # from nf_index in nf_dict, find index in chain
    #     picked_index = self.sfc_dict.keys().index(picked_vnf) if picked_vnf is not None else picked_index
    #     visited_node_dict = curr_solution[self.sfc_dict.keys()[picked_index]]
    #     node_candidate = list()
    #     for node in self.graph.nodes():
    #         # if node != visited_node:
    #         if picked_vnf in self.graph.node[node]['allowed_vnf_list']:
    #             node_candidate.append(node)
    #     candidate_list = list()
    #     for vnode in node_candidate:
    #         new_solution = self.find_match(curr_solution, picked_vnf, vnode)
    #         if new_solution is None:
    #             continue
    #         solution_info_dict = self.post_cal_total_cost(new_solution)
    #         if solution_info_dict is not None:
    #             temp_candidate_dict = OrderedDict()
    #             temp_candidate_dict['solution'] = copy.deepcopy(new_solution)
    #             temp_candidate_dict['solution_info_dict'] = copy.deepcopy(solution_info_dict)
    #             # print 'Temp solution', new_solution
    #             # print 'Temp total cost', temp_candidate_dict['solution_info_dict']['total_cost']
    #             # print 'Temp routing cost', temp_candidate_dict['solution_info_dict']['routing_cost']
    #             candidate_list.append(temp_candidate_dict)
    #         else:
    #             continue
    #     if not candidate_list:
    #         return None, None, None
    #     else:
    #         final_bst_cost = min([candidate['solution_info_dict']['total_cost'] for candidate in candidate_list])
    #         final_candidate_list = [candidate for candidate in candidate_list if candidate['solution_info_dict']['total_cost'] == final_bst_cost]
    #         # Can think about strict constrain here
    #         final_candidate = final_candidate_list[0]
    #         final_solution = final_candidate['solution']
    #         # print 'Final solution', final_solution
    #         # print 'Final total cost', final_bst_cost
    #         # print 'Routing cost', final_candidate['solution_info_dict']['routing_cost']
    #         final_solution_info_dict = final_candidate['solution_info_dict']
    #
    #     return {picked_vnf: final_solution[picked_vnf]}, final_solution, final_solution_info_dict
    #     # Since the final result did not change, the same trial is run at the end


        # apply proposed Tabu strategies here
        # There are 2 main strategies: randomly pick VNF and randomly pick node candidate

    def find_best_neighborhood(self, curr_solution, policy):
        bst_cost_dict = OrderedDict()
        picked_vnf = None
        picked_index = None
        # apply random VNF first
        if policy == 'random':
            picked_vnf = random.choice(self.sfc_dict.keys())

        # from nf_index in nf_dict, find index in chain
        print 'Tabu++: visted VNF', picked_vnf
        picked_index = self.sfc_dict.keys().index(picked_vnf)
        node_candidate = list()
        for node in self.graph.nodes():
            # if node != visited_node:
            if picked_vnf in self.graph.node[node]['allowed_vnf_list']:
                node_candidate.append(node)

        trial_node = random.choice(node_candidate)    # does not make any sense to the chain configuration
        new_solution = self.find_match(curr_solution, picked_vnf, trial_node)
        if new_solution is None:
            return None, None, None

        else:
            final_solution = copy.deepcopy(new_solution)
            final_solution_info_dict = self.post_cal_total_cost(new_solution)
            if final_solution_info_dict is None:
                return None, None, None

        return {picked_vnf: final_solution[picked_vnf]}, final_solution, final_solution_info_dict
        # Since the final result did not change, the same trial is run at the end

    def ordered_path_list(self, paid_list):
        paid_dict = OrderedDict()
        for paid_index, paid in enumerate(paid_list):
            paid_dict[paid_index] = paid
        order_path_tup = sorted(paid_dict.items(), key=lambda kv: kv[1])
        return order_path_tup

    def in_tabu_list(self, match_dict):
        check = False
        for tabu_dict in self.tabu_list:
            if tabu_dict == match_dict:
                check = True
                break
        return check

    def tabu_list_remove(self, match_dict):
        temp_tabu_list = self.tabu_list
        for tabu_index, tabu_dict in enumerate(temp_tabu_list):
            if tabu_dict == match_dict:
                self.tabu_list.pop(tabu_index)

    # Calculate the routing cost cost
    def routing_cost_func(self, node_candidate, curr_solution, graph):
        path_dict = OrderedDict()
        # comm_cost includes key (target node) and value(comm_cost)
        curr_routing_cost = OrderedDict()
        source_node = None
        if curr_solution:
            curr_len = len(curr_solution)
            source_node_dict = curr_solution.values()[curr_len - 1]
            source_node = source_node_dict.keys()[0]
        for node in node_candidate:
            if node == source_node or not curr_solution:
                curr_routing_cost[node] = 0
                path_dict[node] = list()
            else:
                # This can return a list of paths, strictly condition needed
                # this will be a number of nodes for routing cost
                path_list = nx.all_shortest_paths(graph, source=source_node, target=node)
                # path_list = nx.all_shortest_paths(self.graph, source=source_node, target=node, weight='delay')
                # Add constrains for link capacity. Traffic rate is also considered as link rate
                filtered_path = list()
                # Determine the current link usage the existing source and destination for link
                # Find link with lowest latency: path = [1 5 7]
                # visited_path = list()
                for path in path_list:
                    illegal_path = False
                    for pindex, pnode in enumerate(path):
                        if pindex < len(path) - 1:
                            p_snode = pnode
                            p_dnode = path[pindex+1]
                            # determine the BW usage between them. Check whether there are same NS
                            # across 2 physical nodes
                            if not nx.has_path(graph, p_snode, p_dnode):
                                print 'Tabu++: There is no direct link. Revise comm_cost_func'
                                return

                            self.update_curr_link_usage(p_snode, p_dnode, graph)
                            if graph[p_snode][p_dnode]['curr_load'] + self.req_requirements['rate'] > graph[p_snode][p_dnode]['maxBW']:
                                illegal_path = True
                                break
                    if not illegal_path:

                        filtered_path.append(path)

                # nx.dijkstra_path(rdgraph, source=0, target=5, weight='avail')
                # remember here paths can have same cost but different length
                if not filtered_path:
                    continue
                else:
                    min_path_length = min([len(path) for path in filtered_path])
                    final_path_list = [path for path in filtered_path if len(path) == min_path_length]
                    path_dict[node] = final_path_list[0]
                    curr_routing_cost[node] = len(final_path_list[0])/float(len(final_path_list[0])+self.NSlen)
        return curr_routing_cost, path_dict

    # This is used to calculate chain configuration cost
    # if need new resource for dst_node or src_node, config_cost = 1
    # config_cost = 0 when both src_node and dst_node are reused
    # Perfect match will be: VNF-index: node-index: instance-index
    def chain_config_cost(self, dst_nf, src_dict, node_candidate):
        # calculate number of consecutive VNFs
        config_cost = OrderedDict()
        for dst_node in node_candidate:
            if not src_dict:
                config_cost[dst_node] = 0
            else:
                src_vnf = src_dict.keys()[0]
                src_node = src_dict[src_vnf]
                for ns_id, ns_info_dict in self.sys_ns_dict.items():
                    mapping_dict = ns_info_dict['mapping']
                    # share_list = list()
                    for mp_index, mp_nf in enumerate(mapping_dict.keys()):
                        mp_node_id = mapping_dict[mp_nf]
                        if src_vnf == mp_nf and src_node == mp_node_id:
                                if mp_index < len(mapping_dict) - 1:
                                    nxt_mp_nf = mapping_dict.keys()[mp_index + 1]
                                    nxt_node = mapping_dict[nxt_mp_nf]
                                    if {dst_nf: dst_node} == {nxt_mp_nf: nxt_node}:
                                        print 'Tabu++: COUPLE map detected!!!'
                                        config_cost[dst_node] = 0
                                break
                    if config_cost.get(dst_node) == 0:
                        break
                if config_cost.get(dst_node) is None:
                    config_cost[dst_node] = 1
        return config_cost

    def update_curr_link_usage(self, src_node, dst_node, graph):
        graph[src_node][dst_node]['curr_load'] = 0
        if graph[src_node][dst_node].get('req'):
            for req in graph[src_node][dst_node]['req']:
                if req['lifetime'] >= self.timer:
                    graph[src_node][dst_node]['curr_load'] =\
                        graph[src_node][dst_node]['curr_load'] + req['rate']

    # Combine comp cost and config cost - chain aware
    def pre_comp_config_cost_func(self, nf_index, src_dict, node_candidate, graph):
        req_load = self.req_requirements['proc_cap']
        vnf_load = self.nf_prop['proc_cap'][nf_index]
        # comm_cost includes key (target node) and value(comm_cost)
        curr_comp_cost = OrderedDict()
        config_cost = OrderedDict()
        final_dst_node = OrderedDict()
        node_match = OrderedDict()
        # Determine a set of possible instances on a visited node
        for node in node_candidate:
            inst_existed = False
            if graph.node[node]['instances'].get(nf_index):
                nf_inst_dict = graph.node[node]['instances'][nf_index]
                node_match[node] = list()
                print 'Checked node', node
                for inst_index, inst_info_list in nf_inst_dict.items():
                    total_load = sum([inst_info_dict['req_load'] for inst_info_dict in inst_info_list if inst_info_dict['lifetime'] >= self.timer])
                    if req_load + total_load <= self.nf_prop['proc_cap'][nf_index]:
                        curr_comp_cost[node] = 0
                        inst_existed = True
                        # node_match[node].append({'id': inst_index, 'curr_load': total_load})
                        node_match[node].append(inst_index)
                    else:
                        print 'Overloaded node', node
                        print 'current load', total_load
                        print 'Req load', req_load
                        print 'expected load', (total_load+req_load)
                        print 'VNF cap', self.nf_prop['proc_cap'][nf_index]
                        # import time
                        # time.sleep(3)

            if not inst_existed:
                # Limit the number of node by overal node capacity
                curr_node_load = graph.node[node]['curr_load']
                total_node_cap = graph.node[node]['cpu']
                if (vnf_load + curr_node_load) > total_node_cap:
                    continue
                # curr_node_load = 0.01 if curr_node_load == 0 else curr_node_load
                exp_node_load = curr_node_load + vnf_load
                curr_comp_cost[node] = vnf_load / float(exp_node_load)  # This is node-level index
                final_dst_node[node] = None        # There is no instance to reuse
                config_cost[node] = 0 if not src_dict else 1

                if node in node_match:
                    node_match.pop(node)
        # find the best instance, which is the chain-aware,  to reuse here
        if node_match:     # this is just for existing instance
            print 'Node match:', node_match
            inst_candidate = self.matched_config_cost(nf_index, src_dict, node_match)
            print 'Matched instance', inst_candidate
            if inst_candidate:
                # choose the min inst load
                for cd_node, cd_inst_dict in inst_candidate.items():
                    if not cd_inst_dict:
                        print 'Tabu++: can reuse but {src-int, dst-inst} not in any chain'
                        config_cost[cd_node] = 0 if not src_dict else 1
                        unmatched_inst_list = node_match[cd_node]
                        final_dst_node[cd_node] = self.unmatched_config_cost(nf_index, cd_node, unmatched_inst_list)
                        print 'Final unmatched candidate', cd_node
                        continue
                    print 'Tabu++: couple detected!!!'
                    local_ins_dict = OrderedDict()
                    for inst_id, ns_list in cd_inst_dict.items():
                        local_ins_dict[inst_id] = len(ns_list)
                    target_inst_id = max(local_ins_dict, key=local_ins_dict.get)  # the most shared instance---key word here   # noqa
                    final_dst_node[cd_node] = target_inst_id
                    config_cost[cd_node] = 0

        if len(final_dst_node) != len(config_cost):
            print 'Tabu++: Error in comp_config_cost!'
            print final_dst_node
            print config_cost
        return curr_comp_cost, config_cost, final_dst_node
        # config_cost = 0, final_dst_node != None: reused instance, reused sub-chain
        # config_cost = 1, final_dst_node == None: new instance, new chain
        # config_cost = 1, final_dst_node != None: reused instance, new chain

    # Combine comp cost and config cost - chain aware
    def post_comp_config_cost_func(self, nf_index, src_dict, node_dict, graph):
        vnf_load = self.nf_prop['proc_cap'][nf_index]
        # comm_cost includes key (target node) and value(comm_cost)
        curr_comp_cost = OrderedDict()
        config_cost = OrderedDict()
        for visited_node, visited_instance in node_dict.items():            # instance can be None here
            if visited_instance is None:
                curr_node_load = graph.node[visited_node]['curr_load']
                total_node_cap = graph.node[visited_node]['cpu']
                if (vnf_load + curr_node_load) > total_node_cap:
                    continue
                exp_node_load = curr_node_load + vnf_load
                curr_comp_cost[visited_node] = vnf_load / float(exp_node_load)  # This is node-level index
                config_cost[visited_node] = 1 if src_dict else 0
            else:
                curr_comp_cost[visited_node] = 0
                match_dict = self.matched_config_cost(nf_index, src_dict, {visited_node: [visited_instance]})
                if not match_dict[visited_node]:
                    config_cost[visited_node] = 1 if src_dict else 0
                else:
                    print 'Tabu++ for the best solution: Couple detected!!!'
                    # import time
                    # time.sleep(10)
                    config_cost[visited_node] = 0

        return curr_comp_cost, config_cost

    def matched_config_cost(self, nf_index, src_dict, match_node):
        # find which node matched with existing sub-chain
        # inst here was already verified with enough resources
        inst_candidate = OrderedDict()
        for node, inst_list in match_node.items():
            inst_candidate[node] = OrderedDict()
            if src_dict:
                for inst_index in inst_list:   # inst_id changed here
                    # inst_index = inst_dict['id']
                    dst_dict = {nf_index: {node: inst_index}}
                    for ns_id, ns_info_dict in self.sys_ns_dict.items():   # ns_id changed here
                        mapping_dict = ns_info_dict['mapping']
                        for map_index, orig_nf in enumerate(mapping_dict.keys()):
                            # if src_dict:
                                # print 'check the match'
                                # print {orig_nf: mapping_dict[orig_nf]}
                                # print src_dict
                                # import time
                                # time.sleep(2)
                            print 'mapping dict:', {orig_nf: mapping_dict[orig_nf]}
                            print 'src dict:', src_dict
                            if {orig_nf: mapping_dict[orig_nf]} == src_dict:
                                print 'I am here for source check'
                                # print 'Destination dict', dst_dict
                                # print 'Src', src_dict
                                if map_index < len(mapping_dict) - 1:
                                    nxt_orig_nf = mapping_dict.keys()[map_index+1]
                                    # print 'Possible Mirror', {nxt_orig_nf: mapping_dict[nxt_orig_nf]}
                                    # import time
                                    # time.sleep(3)
                                    if {nxt_orig_nf: mapping_dict[nxt_orig_nf]} == dst_dict:
                                        print 'Tabu++: Match detected both for source and dest.!!!'
                                        # import time
                                        # time.sleep(5)

                                        if inst_candidate[node].get(inst_index) is None:
                                            inst_candidate[node][inst_index] = list()
                                        inst_candidate[node][inst_index].append(ns_id)
                                        break

        return inst_candidate

    def unmatched_config_cost(self, nf_index, target_node, inst_list):
        # find which node matched with existing sub-chain
        # inst here was already verified with enough resources
        inst_candidate = dict()
        for inst_index in inst_list:
            inst_candidate[inst_index] = list()
            move = {nf_index: {target_node: inst_index}}
            for ns_id, ns_info_dict in self.sys_ns_dict.items():   # ns_id changed here
                mapping_dict = ns_info_dict['mapping']
                for orig_nf, node_dict in mapping_dict.items():
                    # print 'trial', {orig_nf: node_dict}
                    # print 'actual', move
                    if {orig_nf: node_dict} == move:
                            print 'Dest. node matched detected!!!! But source node is different'
                            inst_candidate[inst_index].append(ns_id)
                            break
            if not inst_candidate.get(inst_index):
                print self.sys_ns_dict
                print 'why?????'
        if not inst_candidate:
            return None
        else:
            max_ns = max([len(ns_list) for inst_id, ns_list in inst_candidate.items()])
            # print 'Maximum shared!!!!!!!!!!!!!!!!!!!!!!!!!!', max_ns
            most_shared_list = [inst_id for inst_id, ns_list in inst_candidate.items() if len(ns_list) == max_ns]
            return most_shared_list[0]

    # Calculate the reliability cost. Re-examine it
    def rel_cost_func(self, nf_index, node_candidate):
        rel_cost = OrderedDict()
        for node in node_candidate:
            node_rel = self.graph.node[node]['rel']
            origin_nf_rel = self.nf_prop['rel'][nf_index]
            nf_rel = origin_nf_rel * node_rel
            rel_cost[node] = nf_rel
        return rel_cost

    # The good thing of paid is calculate the reliability of all VNFs on the same node
    # type_of_search=single, cluster
    # mention the case when the candidate gets over resources
    def paid_engine(self, candidate, type_of_search):
        sum_node_weight = 0
        # candidate is a dict (vnf_index, node)
        if type_of_search == 'single':
            conv_candidate = self.find_coloc(candidate.values())
            update_rel_node = OrderedDict()
            for target_node, vnf_list in conv_candidate.items():
                update_rel_node[target_node] = self.graph.node[target_node]['rel']
                for vnf_index in vnf_list:
                    update_rel_node[target_node] = update_rel_node[target_node] * self.nf_prop['rel'][vnf_index]

            for nf_index, target_node in candidate.items():
                # determine the reuse factor here
                # load_index = req_load/curr_load      # This is VNF-level index
                req_load = self.req_requirements['proc_cap']
                inst_dict = OrderedDict()
                if self.graph.node[target_node]['instances'].get(nf_index):
                    nf_inst_dict = self.graph.node[target_node]['instances'][nf_index]
                    for inst_index, inst_info_list in nf_inst_dict.items():
                        total_load = sum([inst_info_dict['req_load'] for inst_info_dict in inst_info_list if inst_info_dict['lifetime'] >= self.timer])
                        if req_load + total_load <= self.nf_prop['proc_cap'][nf_index]:
                            inst_dict[inst_index] = total_load
                if inst_dict:
                    min_load = min([load for inst_index, load in inst_dict.items()])
                else:
                    vnf_load = self.nf_prop['proc_cap'][nf_index]
                    curr_node_load = self.graph.node[target_node]['curr_load']
                    total_node_cap = self.graph.node[target_node]['cpu']
                    if (vnf_load + curr_node_load) > total_node_cap:
                        return None
                    min_load = 0
                reuse_factor = req_load/float(min_load+req_load)
                sum_node_weight = sum_node_weight + update_rel_node[target_node]/reuse_factor

        return sum_node_weight

    def find_coloc(self, candidate):
        conv_candidate = OrderedDict()
        for index, node in enumerate(candidate):
            if node not in conv_candidate:
                conv_candidate[node] = list()
                conv_candidate[node].append(index)
            else:
                conv_candidate[node].append(index)
        return conv_candidate

    def add_link_usage(self, src_node, dst_node, graph):
        # BW(src_node, dst_node) does not specify the endpoints
        if nx.has_path(graph, src_node, dst_node):
            # print graph[src_node][dst_node]['curr_load']
            self.update_curr_link_usage(src_node, dst_node, graph)
            if graph[src_node][dst_node]['curr_load'] + self.req_requirements['rate'] > \
                    graph[src_node][dst_node]['maxBW']:
                print 'Tabu++: the link capacity is over!!! Revise add_link_usage'
                # print src_node, dst_node
            graph[src_node][dst_node]['curr_load'] = graph[src_node][dst_node]['curr_load'] + \
                                                          self.req_requirements['rate']
            if graph[src_node][dst_node].get('req') is None:
                graph[src_node][dst_node]['req'] = list()
            graph[src_node][dst_node]['req'].append(
                {'id': self.req_id, 'lifetime': self.req_info['lifetime'], 'rate': self.req_requirements['rate']})
        else:
            print 'Tabu++: there is no direct link. Revise add_link_usage'

    def update_graph(self, ns_candidate, graph=None, path_list=None):
        # For Tabu++, target VNF instance is known
        if graph is None:
            graph = self.graph
        if path_list is None:
            path_list = self.e2e_path
        # Update physical node
        for vnf_index, node_dict in ns_candidate.items():
            if not node_dict:
                print 'Tabu++: Node dict error. Revise update graph'
                return
            node = node_dict.keys()[0]
            vnf_inst = node_dict[node]

            inst_info = OrderedDict()
            inst_info['lifetime'] = self.req_info['lifetime']
            inst_info['req_load'] = self.req_requirements['proc_cap']
            inst_info['ns_id'] = self.req_id

            if vnf_inst is None:
                if graph.node[node]['curr_load'] + self.nf_prop['proc_cap'][vnf_index] > graph.node[node]['cpu']:
                    print 'Tabu++: Load in physical node is over. Revise update_graph'
                    # print index
                    return
                graph.node[node]['curr_load'] =\
                    graph.node[node]['curr_load'] + self.nf_prop['proc_cap'][vnf_index]
                inst_id = uuid.uuid4()
                node_dict[node] = inst_id       # Update ns_candidate
                graph.node[node]['instances'][vnf_index] = OrderedDict()
                graph.node[node]['instances'][vnf_index][inst_id] = list()
                graph.node[node]['instances'][vnf_index][inst_id].append(inst_info)
            else:
                nf_inst_list = graph.node[node]['instances'][vnf_index][vnf_inst]
                total_load = sum([inst_info_dict['req_load'] for inst_info_dict in nf_inst_list if
                                  inst_info_dict['lifetime'] >= self.timer])
                if self.req_requirements['proc_cap'] + total_load <= self.nf_prop['proc_cap'][vnf_index]:
                    nf_inst_list.append(inst_info)
                else:
                    print 'Tabu++: VNF instance load is over. Revise update_graph'

        # Update physical link
        for node_index, node in enumerate(path_list):
            if node_index < len(path_list) - 1:
                p_snode = node
                p_dnode = path_list[node_index + 1]
                if p_snode == p_dnode:
                    continue
                self.add_link_usage(p_snode, p_dnode, graph)

    def reform_ns_candidate(self, ns_candidate):
        # ns_candidate is a list of node: VNF index:(node - instance index)
        mapping_dict = OrderedDict()
        for vnf_index, node_dict in ns_candidate.items():
            mapping_dict[vnf_index] = node_dict
        return mapping_dict

    def reform_list(self, shared_list):
        tp_list = list()
        for list_index, vnf_index in enumerate(shared_list):
            if list_index < len(shared_list) - 1:
                tp = (vnf_index, shared_list[list_index+1])
                tp_list.append(tp)
        return tp_list

    def get_graph(self):
        return self.graph

    def get_path(self):
        return self.e2e_path
