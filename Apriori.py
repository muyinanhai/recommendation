# -*- coding:utf-8 -*-
#############################################
# author: Nanhai.yang
#############################################
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser
import codecs


class Apriori(object):

    def __init__(self, min_support, min_confidence):
        """
        :param min_support:
        :param min_confidence:
        """
        self._min_support = min_support
        self._min_confidence = min_confidence
        self._transaction_len = 0
        self._to_ret = {}
        self._result = {}
        self._frequent_set = defaultdict(int)

    @staticmethod
    def _sub_set(arr, max_len=None):
        """
        :param arr:
        :return: all combination of arr
        """
        if max_len is None:
            max_len = len(arr)
        return chain(*[combinations(arr, i+1) for i in range(max_len)])

    @staticmethod
    def _join_set(arr, length):
        """
        :param arr:
        :param length:
        :return:
        """
        return set([i.union(j) for i in arr for j in arr if len(i.union(j)) == length])

    def _find_frequent_item_set(self, item_set, transaction_list):
        tmp_item_set = set()
        local_set = defaultdict(int)

        for item in item_set:
            for transaction in transaction_list:
                if item.issubset(transaction):
                    self._frequent_set[item] += 1
                    local_set[item] += 1
        for item, count in local_set.items():
            support = float(count) / self._transaction_len
            if support >= self._min_support:
                tmp_item_set.add(item)
        return tmp_item_set

    def fit(self, item_set, transaction_list):
        """
        :param item_set:
        :param transaction_list:
        :return:
        """
        self._transaction_len = len(transaction_list)
        large_set = dict()
        k = 2
        current_set = set([item for item in item_set])
        while True:
            current_set = self._find_frequent_item_set(current_set, transaction_list)
            if len(current_set) == 0:
                break
            large_set[k-1] = current_set
            current_set = self._join_set(current_set, k)
            k += 1

        for key, val in large_set.items():
            for item in val:
                self._to_ret[frozenset(item)] = self._frequent_set[item] / self._transaction_len

        # generate rules
        for key, val in large_set.items():
            if key == 1:
                continue
            for item in val:
                # get all subsets except the one same with item
                subsets = map(frozenset, [x for x in self._sub_set(item, len(item)-1)])
                for element in subsets:
                    remain = item.difference(element)
                    confidence = self._frequent_set[item]/self._frequent_set[element]
                    if confidence >= self._min_confidence:
                        self._result[element] = self._result.get(element, {})
                        self._result[element][remain] = confidence
        return self._to_ret, self._result

    def _ranking(self, x):
        """
        :param x:
        :return:
        """
        subsets = self._sub_set(x)
        recommend = {}
        for subset in subsets:
            subset = frozenset(subset)
            if subset not in self._to_ret.keys():
                continue
            remain = self._result.get(subset, {})
            for recommend_set, confidence in remain.items():
                for single_recommend in recommend_set:
                    # skip the item have occur in x
                    if single_recommend in x:
                        continue
                    recommend[single_recommend] = recommend.get(single_recommend, 0) + confidence * self._to_ret[subset] * len(subset)/len(x)
        recommend = {key:val for key, val in recommend.items() if val>self._min_confidence}
        return sorted(recommend.items(), key=lambda d: d[1], reverse=True)

    def predict(self, transaction_list):
        """
        :param transaction_list:
        :return:
        """
        transaction_list = [self._clean_transaction(transaction) for transaction in transaction_list]
        recommend = map(self._ranking,transaction_list)
        return [ r for r in recommend]

    def print_result(self):
        for item, support in sorted(self._to_ret.items(), key=lambda  s: s[1]):
            print("item: %s , %.3f" % (str(item), support))
        print("\n--------------------------------------------------------- RULES:")
#       from dict to tuple
        tmp_result = []
        for key, val in self._result.items():
            for sub_key, sub_val in val.items():
                tmp_result.append(((tuple(key), tuple(sub_key)),sub_val))
        for rule, confidence in sorted(tmp_result, key=lambda c: c[1]):
            pre, post = rule
            print("Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence))


def load_data(file_name):
    fin = codecs.open(file_name, "r", "utf-8")
    item_set = set()
    transaction_list = []
    for line in fin:
        line = line.strip().split(',')
        recodes = frozenset(line)
        transaction_list.append(recodes)
        for item in recodes:
            item_set.add(frozenset([item]))
    return item_set, transaction_list

if __name__ == '__main__':
    opt_parser = OptionParser()
    opt_parser.add_option('-f', '--inputFile',
                          dest='input',
                          help='filename',
                          default=None)
    opt_parser.add_option('-s', '--minSupport',
                          dest='min_support',
                          help='minimum support value',
                          default=0.15)
    opt_parser.add_option('-c', '--minConfidence',
                          dest='min_confidence',
                          help='minimum confidence value',
                          default=0.6)
    (options, args) = opt_parser.parse_args()
#    #if options.input is None:
    #    print("No input files, exit")
    #    exit(0)
#    #iter_set, transactions = load_data(options.input)
    iter_set = set(['A', 'B', 'C', 'D', 'E'])
    iter_set = [frozenset([x]) for x in iter_set]
    transactions = [['A', 'B', 'C', 'D'],
                    ['B', 'C', 'E'],
                    ['A', 'B', 'C', 'E'],
                    ['B', 'D', 'E'],
                    ['A', 'B', 'C', 'D']]
    transactions = [frozenset(x) for x in transactions]
    alg = Apriori(0.1, 0.1)
    alg.fit(iter_set, transactions)
    # alg.print_result()
    print("-----------------------------------------------")
    y_pre = alg.predict([['A','B']])
    for y in y_pre:
        print(y)