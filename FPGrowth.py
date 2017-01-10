# -*- coding:utf-8 -*-
###########################
# Author:Nanhai.yang
###########################

from collections import defaultdict, namedtuple


class FPGrowth(object):

    def __init__(self, min_support=0.1, min_confidence=0.1):
        self._min_support = min_support
        self._min_confidence = min_confidence
        self._transaction_len = 0
        self.items = defaultdict(int)

    @staticmethod
    def conditional_tree_from_paths(paths):
        tree = FPTree()
        condition_item = None
        items = set()
        for path in paths:
            if condition_item is None:
                condition_item = path[-1].get_item()
            point = tree.get_root()
            for node in path:
                next_point = point.search(node.get_item())
                if not next_point:
                    items.add(node.get_item())
                    count = node.get_count() if node.get_item() == condition_item else 0
                    next_point = FPNode(tree, node.get_item(), count)
                    point.add_child(next_point)
                    tree.update_route(next_point)
                point = next_point
        assert condition_item is not None
        for path in tree.prefix_path(condition_item):
            count = path[-1].get_count()
            for node in reversed(path[:-1]):
                node.set_count(node.get_count() + count)
        return tree

    def _clean_transaction(self, transaction):
        transaction = list(filter(lambda v: v in self.items, transaction))
        # order by the support count
        transaction.sort(key=lambda v: self.items[v], reverse=True)
        return transaction

    def find_with_suffix(self, tree, suffix):
        for item, nodes in tree.items():
            support = sum(n.get_count() for n in nodes)
            if support >= self._min_support * self._transaction_len and item not in suffix:
                found_set = [item] + suffix
                yield (found_set, support)
                cond_tree = self.conditional_tree_from_paths(tree.prefix_path(item))
                for s in self.find_with_suffix(cond_tree, found_set):
                    yield s

    def fit(self, transactions):
        self._transaction_len = len(transactions)
        for transaction in transactions:
            for item in transaction:
                self.items[item] += 1

        support_count = self._min_support * self._transaction_len
        self.items = dict((item, support) for item, support in self.items.items() if support >= support_count)
        master = FPTree()
        for transaction in transactions:
            master.add(self._clean_transaction(transaction))

        for item_set in self.find_with_suffix(master, []):
            yield item_set

    def predict(self):
        pass


class FPTree(object):

    Route = namedtuple('Route', 'head tail')

    def __init__(self):
        self._root = FPNode(self, None, None)
        self._routes = {}

    def get_root(self):
        return self._root

    def add(self, transaction):
        point = self._root
        for item in transaction:
            next_point = point.search(item)
            if next_point:
                next_point.increment_count()
            else:
                next_point = FPNode(self, item)
                point.add_child(next_point)
                self.update_route(next_point)
            point = next_point

    def update_route(self, point):
        assert point.get_tree() is self
        try:
            route = self._routes[point.get_item()]
            route[1].set_neighbor(point)
            self._routes[point.get_item()] = self.Route(route[0], point)
        except KeyError:
            self._routes[point.get_item()] = self.Route(point, point)

    def nodes(self, item):
        try:
            node = self._routes[item][0]
        except KeyError:
            return
        while node:
            yield node
            node = node.get_neighbor()

    def items(self):
        for item in self._routes:
            yield (item, self.nodes(item))

    def prefix_path(self, item):
        def collect_path(node):
            path = []
            while node and not node.is_root():
                path.append(node)
                node = node.get_parent()
            print(item, path)
            return path
        return (collect_path(node) for node in self.nodes(item))


class FPNode(object):

    def __init__(self, tree, item, count=1):
        self._tree = tree
        self._item = item
        self._count = count
        self._parent = None
        self._children = {}
        self._neighbor = None

    def __contains__(self, item):
        return item in self._children

    def __repr__(self):
        if self.is_root():
            return "<%s (root)>" % type(self).__name__
        return "<%s %r (%r)>" % (type(self).__name__, self.get_item(), self.get_count())

    def get_item(self):
        return self._item

    def get_count(self):
        return self._count

    def set_count(self, n):
        self._count = n

    def get_parent(self):
        return self._parent

    def get_neighbor(self):
        return self._neighbor

    def get_tree(self):
        return self._tree

    def is_root(self):
        return self._item is None and self._count is None

    def is_leaf(self):
        return len(self._children) == 0

    def increment_count(self):
        assert self._count is not None
        self._count += 1

    def set_parent(self, parent):
        self._parent = parent

    def set_neighbor(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("only FPNode can add as neighbor")
        if value and value.get_tree() is not self.get_tree():
            raise ValueError("Cannot have a neighbor from another tree.")
        self._neighbor = value

    def add_child(self, child):
        assert isinstance(child, FPNode)
        if child.get_item() not in self._children:
            self._children[child.get_item()] = child
            child.set_parent(self)

    def search(self, item):
        try:
            return self._children[item]
        except KeyError:
            return None

    def print_tree(self, depth=0):
        print('  ' * depth) + repr(self)
        for child in self.children:
            child.print_tree(depth + 1)

if __name__ == "__main__":
    tran = [['A', 'B', 'C', 'D'],
                    ['B', 'C', 'E'],
                    ['A', 'B', 'C', 'E'],
                    ['B', 'D', 'E'],
                    ['A', 'B', 'C', 'D']]
    fp_growth = FPGrowth(0.2)
    for y in fp_growth.fit(tran):
        print(y)
