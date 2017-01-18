# -*- coding:utf-8 -*-
#############################################
# author: Nanhai.yang
#############################################

from __future__ import print_function
import abc
import numpy as np
import types


#############################################
# Similarity Part
# Used for calculating user(item) distance
#############################################
class Distance(object):
    """
    compute distance for sparse matrix
    """
    def __init__(self, solver='cos'):
        self._solver = solver

    @staticmethod
    def _cosine(x, y):
        try:
            interaction_keys = [key for key in x if key in y]
            if len(interaction_keys) == 0:
                return 0
            x_inter = np.sum([val * val for key, val in x.items()])
            y_inter = np.sum([val*val for key, val in y.items()])
            xy_inter = np.sum([x[key]*y[key] for key in interaction_keys])
            if x_inter == 0 or y_inter == 0:
                return 0
            else:
                return xy_inter/np.sqrt(x_inter * y_inter)
        except ValueError:
            print("value not valid, dict required")
            return -1

    @staticmethod
    def _jaccard(x, y):
        try:
            x_cnt = len(x)
            y_cnt = len(y)
            xy_cnt = np.sum([1 for i in x if i in y])
            return xy_cnt/(x_cnt + y_cnt - xy_cnt)
        except ValueError:
            print("value not valid, dict required")
            return -1

    @staticmethod
    def _pearson(x, y):
        try:
            interaction_keys = [key for key in x if key in y]
            if len(interaction_keys) == 0:
                return 0
            x_inter = np.array([val for key, val in x.items() if key in interaction_keys])
            y_inter = np.array([val for key, val in y.items() if key in interaction_keys])
            x_mean = np.mean(x_inter)
            y_mean = np.mean(y_inter)
            x_dev = np.sum((x_inter-x_mean)**2)
            y_dev = np.sum((y_inter-y_mean)**2)
            return np.sum((x_inter - x_mean) * (y_inter - y_mean)) / np.sqrt(x_dev * y_dev)
        except ValueError:
            print("value not valid, pearson dict required")
            return -1

    def get_distance(self, x, y):
        if self._solver == 'cos':
            return self._cosine(x, y)
        elif self._solver == 'jaccard':
            return self._jaccard(x, y)
        elif self._solver == 'pearson':
            return self._pearson(x, y)
        else:
            raise ValueError("solver not valid")


#############################################
# CF Part
#############################################
class CollaborativeFiltering(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, solver='cos'):
        self._model = {}
        self._data = {}
        self._items = {}
        self._distance = Distance(solver).get_distance

    @abc.abstractmethod
    def fit(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x):
        raise NotImplementedError

    def _get_nearest_neighbor(self, user, n_neighbors=-1):
        similarity = [(other, self._distance(self._data[other], self._data[user])) for other in self._data.keys() if
                      other != user]
        similarity.sort(key=lambda d: d[1])
        if n_neighbors > 0:
            return similarity[:n_neighbors]
        return similarity


#############################################
# UserBased Part
#############################################
class UserBased(CollaborativeFiltering):

    def load_data(self, x):
        if isinstance(x, dict):
            self._data = x
            for user in self._data:
                for item in self._data[user]:
                    self._items[item] = None
            return
        for user, item, rating in x:
            self._data.setdefault(user, {})
            self._data[user][item] = rating
            self._items[item] = None

    def fit(self, x):
        """
        :param x: x.shape equals (,3) and the order of the cols is user, item ,rating, or a dict format
        :return:
        """
        self.load_data(x)
        for user in self._data:
            self._model[user] = self._get_nearest_neighbor(user)

    def _predict_user_item(self, user, item):
        if item in self._data[user]:
            return 1.0
        similarity = [sim * self._data[other][item] for other, sim in self._model[user] if item in self._data[other]
                      and sim >= 0]
        if len(similarity) == 0:
            return 0
        return np.mean(similarity)

    def _predict_user_item_weighted(self, user, item):
        """
        :param user:
        :param item:
        :return:
        """
        if item in self._data[user]:
            return self._data[user][item]
        mean_rate = np.mean([score for score in self._data[user].values()])
        weight = 0
        normalizer = 0
        for neighbor, similarity in self._model[user]:
            if item not in self._data[neighbor]:
                continue
            neighbor_mean_rate = np.mean([r for r in self._model[neighbor].values()])
            weight += similarity * (self._data[neighbor][item] - neighbor_mean_rate)
            normalizer += np.abs(similarity)
        if normalizer == 0:
            return 0
        return mean_rate + (weight / normalizer)

    def predict(self, x):
        # only user
        if len(x) == 0:
            return []
        result = []
        if len(x[0]) == 1:
            for user in x:
                user = user[0]
                predicts = []
                for item in self._items:
                    if item in self._data[user]:
                        continue
                    predicts.append((item, self._predict_user_item(user, item)))
                predicts.sort(key=lambda d: d[1], reverse=True)
                result.append(predicts)
            return result
        elif len(x[0]) == 2:
            for user, item in x:
                if item in self._data[user]:
                    continue
                result.append(self._predict_user_item(user, item))
            return result


#############################################################################
# ItemBased Part, you only need to
# reorder the columns of the input from user,item,rating to item,user,rating
#############################################################################


#############################################################################
# Test Part
#############################################################################
if __name__ == '__main__':
    dataset = {
        'Lisa Rose': {'Lady in the Water': 2.5,
                      'Snakes on a Plane': 3.5,
                      'Just My Luck': 3.0,
                      'Superman Returns': 3.5,
                      'You, Me and Dupree': 2.5,
                      'The Night Listener': 3.0},
        'Gene Seymour': {'Lady in the Water': 3.0,
                         'Snakes on a Plane': 3.5,
                         'Just My Luck': 1.5,
                         'Superman Returns': 5.0,
                         'The Night Listener': 3.0,
                         'You, Me and Dupree': 3.5},

        'Michael Phillips': {'Lady in the Water': 2.5,
                             'Snakes on a Plane': 3.0,
                             'Superman Returns': 3.5,
                             'The Night Listener': 4.0},
        'Claudia Puig': {'Snakes on a Plane': 3.5,
                         'Just My Luck': 3.0,
                         'The Night Listener': 4.5,
                         'Superman Returns': 4.0,
                         'You, Me and Dupree': 2.5},
        'Mick LaSalle': {'Lady in the Water': 3.0,
                         'Snakes on a Plane': 4.0,
                         'Just My Luck': 2.0,
                         'Superman Returns': 3.0,
                         'The Night Listener': 3.0,
                         'You, Me and Dupree': 2.0},
        'Jack Matthews': {'Lady in the Water': 3.0,
                          'Snakes on a Plane': 4.0,
                          'The Night Listener': 3.0,
                          'Superman Returns': 5.0,
                          'You, Me and Dupree': 3.5},
        'Toby': {'Snakes on a Plane': 4.5,
                 'You, Me and Dupree': 1.0,
                 'Superman Returns': 4.0}}
    ub = UserBased("pearson")
    ub.fit(dataset)
    print(ub.predict([["Toby"]]))
