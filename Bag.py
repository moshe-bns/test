import torch
import numpy as np
import random
from enum import Enum

from Main_run import Mode, mode


class Proportion(Enum):
    STRONG_MAJORITY =1
    MAJORITY =2
    EQUALLY_LIKELY = 3
    LOW_MAJORITY = 4


# proportion_to_prob = {
#     Proportion.STRONG_MAJORITY : random.choice([0.8, 0.85, 0.9, 0.95]),
#     Proportion.MAJORITY : random.choice([0.6, 0.65, 0.7, 0.75]),
#     Proportion.EQUALLY_LIKELY : random.choice([0.45, 0.5, 0.55]),
#     Proportion.LOW_MAJORITY : random.choice([0.25, 0.3, 0.35, 0.4]),
#                       }
class BagParam:
    def __init__(self, bag_index, bag_size, sample_dict=None):
        self.bag_index = bag_index
        self.bag_size = bag_size
        self.sample_dict = sample_dict

class Bag:
    def __init__(self, bag_idx, indices_in_data, targets_proportions_in_bag= None):
        if type(targets_proportions_in_bag) is dict:
            assert round (sum(targets_proportions_in_bag.values()),2) == 1
        self.bag_idx = bag_idx
        self.indices_in_data = indices_in_data
        self.targets_proportions_in_bag = targets_proportions_in_bag

    def get_bag_idx(self):
        return self.bag_idx

    def get_indices_in_data(self):
        return self.indices_in_data

    def get_true_proportion(self, target_class= None):
        if target_class is None:
            return self.targets_proportions_in_bag
        if target_class in self.targets_proportions_in_bag:
            return self.targets_proportions_in_bag[target_class]
        return 0


class BagFactory:
    def __init__(self, data, targets):
        assert len(data) == len(targets)
        self.data = data
        self.targets = targets
        self.counter_bag_idx = 0

    def get_new_bag_index(self, use = False):
        if use:
            self.counter_bag_idx += 1
        return self.counter_bag_idx -1

    def reset_bag_index_cocunter(self):
        self.counter_bag_idx =0


    def create_bags_from_bag_param(self, bag_params):
        if mode == Mode.CLASSIFICATION:
            return self.create_bag_for_classification(bag_params.bag_size, bag_params.sample_dict)
        return self.create_bag_for_regression(bag_params.bag_size)

    def create_bag_for_classification(self, bag_size, sample_dict, bag_idx=None):
        assert round(sum(sample_dict.values()), 2) == 1
        bag_idx = self.get_new_bag_index(use=True) if bag_idx is None else bag_idx
        integer_adjusted_sample_dict={}
        asks_for_example = []
        total_examples = 0
        for c in sample_dict:
            if int(sample_dict[c]*bag_size)==round(sample_dict[c]*bag_size,2):
                integer_adjusted_sample_dict [c] = int(sample_dict[c]*bag_size)/float(bag_size)
            else:
                integer_adjusted_sample_dict[c] = int(sample_dict[c]*bag_size)/float(bag_size)
                asks_for_example.append(c)
            total_examples += int(sample_dict[c] * bag_size)
        remain = bag_size-total_examples
        if asks_for_example:
            chosen = np.random.choice(asks_for_example, remain, replace=False).tolist()
            for c in chosen:
                integer_adjusted_sample_dict[c] = (int(sample_dict[c]*bag_size)+1)/float(bag_size)

        data_idx = []
        for c, proportion in integer_adjusted_sample_dict.items():
            idx = np.where(self.targets == c)[0]
            sample_size = int(round(bag_size*integer_adjusted_sample_dict[c],2))
            replace = False if (len(idx)>sample_size) else True
            data_idx += np.random.choice(idx, sample_size, replace=replace).tolist()
        assert len(data_idx) == bag_size
        return Bag(bag_idx, data_idx, integer_adjusted_sample_dict)

    def create_bag_for_regression(self, bag_size, bag_idx=None):
        bag_idx = self.get_new_bag_index(use=True) if bag_idx is None else bag_idx
        optional_idx = np.arange(len(self.data))
        replace = False if (len(optional_idx) > bag_size) else True
        data_idx = np.random.choice(optional_idx, bag_size, replace=replace).tolist()
        return Bag(bag_idx, data_idx, self.targets[data_idx].mean())

    # def create_bag_by_majority(self, majority_type, major_class, all_classes, bag_size, bag_idx=None):
    #     major_class_proportion = proportion_to_prob[majority_type]
    #     sample_dict = {major_class:major_class_proportion}
    #     other_props = (1-major_class)/(len(all_classes)-1)
    #     remain_classes = [c for c in all_classes if c is not major_class]
    #     if remain_classes[-1]:
    #         for c in remain_classes[:-1]:
    #             sample_dict[c] = other_props
    #         sample_dict[remain_classes[-1]] = 1 - major_class - len(remain_classes[:-1]) * other_props
    #     return self.create_bag_for_classification( bag_size, sample_dict, bag_idx)


