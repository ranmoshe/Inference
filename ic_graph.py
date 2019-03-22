'''
IC* graph for 3 categories
'''
import itertools
import networkx as nx

from sample_set import SampleSet

class HybridGraph(nx.Graph):
    def add_directed_edge(self, out_node, in_node):
        self.add_edge(out_node, in_node, out=out_node)

class IC_Graph():
    SIGNIFICANCE_LEVEL = 0.1

    def __init__(self, sampleSet):
        self.smp = sampleSet
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.smp.matrix.columns)
        self.nodes = self.graph.nodes

    @staticmethod
    def _subsets(groupList):
        for i in range(0, len(groupList)+1):
            for subset in itertools.combinations(groupList, i):
                yield list(subset)

    def print_results(self, res, node1, node2, subset):
        print(f'node1: {node1}, node2: {node2}, subset: {subset}')
        print(res)
        print(self.smp.probability(list(set([node1]+[node2]+subset))))
        print('==========================================')

    def _find_minimal_group_for_independence(self, node1, node2, rest):
        '''
        H0 is that node1, node2 are independent given the subset.
        P value should be smaller than the significance level in order to disprove H0
        If P val is bigger, method return True => nodes are independent
        '''
        for subset in self._subsets(rest):
            try:
#                if node1 == 'gender' and node2 == 'num_of_module_created' and subset == ['70_uploads']:
#                    import ipdb; ipdb.set_trace()
                res = self.smp.mutual_information([node1], [node2], subset)
            except Exception as e:
                import ipdb; ipdb.set_trace()
                pass
#            if node1 == 'dog' and node2 == 'mouse':
#                self.print_results(res, node1, node2, subset)
            p_val = res['p_val']
            if p_val > self.SIGNIFICANCE_LEVEL:
                return True, p_val, subset
        return False, None, None

    def ic_step_1(self):
        independents = []
        node_arr = [node for node in self.nodes]
        for idx1 in range(len(node_arr)):
            for idx2 in range(idx1+1, len(node_arr)):
                node1 = node_arr[idx1]
                node2 = node_arr[idx2]
                rest = [node_arr[i] for i in range(len(node_arr)) if i not in (idx1, idx2)]
                is_independent, significance, condGroup = self._find_minimal_group_for_independence(node1, node2, rest)
                if is_independent:
                    independents.append((node1, node2, condGroup))
                else:
                    self.graph.add_edge(node1, node2)
        return independents

    def ic_step_2(self, independents):
        for node1, node2, condGroup in independents:
            common_neighbors = list(set(self.graph.neighbors(node1)) & set(self.graph.neighbors(node1)))
            for neighbor in common_neighbors:
                if not neighbor in condGroup:
                    self.graph.remove_edge(node1, neighbor)
                    self.graph.remove_edge(node2, neighbor)
                    self.graph.add_edges_from([(node1, neighbor), (node2, neighbor)], out=neighbor)

    @staticmethod
    def get_directed_all_directions(directed):
        edges2d = [[(edge[0], edge[1]), (edge[1], edge[0])] for edge in directed]
        return list(itertools.chain.from_iterable(edges2d))

    def get_edges_with_types(self):
        directed = [(t, t[2]) for t in self.graph.edges.data('out') if t[2] is not None]
        directed_star = [(t, t[2]) for t in self.graph.edges.data('out_star') if t[2] is not None]
        directed_edges = [t[0] for t in directed + directed_star]
        out_nodes = [t[1] for t in directed + directed_star]
        directed_all_directions = self.get_directed_all_directions(directed_edges)
        non_directed_edges = [edge for edge in self.graph.edges if edge not in directed_all_directions]
        return (directed_edges, list(set(out_nodes)), non_directed_edges)

    @staticmethod
    def get_neighbor(node, edge):
        node_idx = edge.index(node)
        neighbor_idx = 1 if node_idx == 0 else 0
        return edge[neighbor_idx]

    def ic_step_3_r1(self):
        _, out_nodes, non_directed_edges = self.get_edges_with_types()
        for node in out_nodes:
            for edge in non_directed_edges:
                if node in edge:
                    neighbor = self.get_neighbor(node, edge)
                    self.graph.remove_edge(node, neighbor)
                    self.graph.add_edges_from([(node, neighbor)], out_star=neighbor)

    def ic_step_3_r2(self):
        pass

    def build_graph(self):
        independents = self.ic_step_1()
        self.ic_step_2(independents)
        self.ic_step_3_r1()
        self.ic_step_3_r2()

    def ic_graph(self):
        return None
