from abc import ABC,abstractmethod
import torch, torchvision, numpy
import numpy as np
import torch.nn.functional as F
import sys

class AutomatedConstraintParams:
    def __init__(self, bag_index, constraint_type, constraint_class = None, bag2=None, boundary_spread = None, **kwargs):
        self.bag_index = bag_index
        self.constraint_type = constraint_type
        self.constraint_class = constraint_class
        self.bag2= bag2
        self.boundary_spread = boundary_spread
        self.kwargs = kwargs

    def create_constraint(self, bags):
        from Main_run import mode, Mode
        if mode == Mode.CLASSIFICATION:
            return self.constraint_type(bags[self.bag_index], constraint_class=self.constraint_class,
                                    boundary_spread = self.boundary_spread, **self.kwargs)
        else:
            return self.constraint_type(bags[self.bag_index],boundary_spread = self.boundary_spread,**self.kwargs)

class Constraint(ABC):
    def __init__(self, bag, constraint_class = None):
        from Main_run import Mode, mode
        
        mode = mode
        if mode == mode.CLASSIFICATION:
            assert constraint_class is not None
        self.bag = bag
        self.constraint_class = constraint_class
        pass

    @abstractmethod
    def calc_loss(self, predictions, targets, idx, device):
        pass

    def get_bags(self):
        return {self.bag}

class ExactProportionConstraint(Constraint):
    def __init__(self, bag, constraint_class = None, **kwargs):
        super().__init__(bag, constraint_class)
        self.true_proportion = self.bag.get_true_proportion()
        if constraint_class is not None:
            self.true_proportion = self.true_proportion[constraint_class]

    def calc_loss(self, predictions, targets, idx, device):
        from Main_run import Mode, mode
        bag_idx = self.bag.get_indices_in_data()
        rel_idx = np.isin(idx, bag_idx)
        if not rel_idx.sum():
            return 0
        if mode == Mode.CLASSIFICATION:
            probabilities = torch.softmax(predictions[rel_idx], dim=1)
            predicted_proportion = torch.mean(probabilities[:, self.constraint_class])
        else:
            predicted_proportion = torch.mean(predictions[rel_idx])
        return F.mse_loss(predicted_proportion , self.true_proportion)# torch.mul(predicted_proportion - self.true_proportion, predicted_proportion - self.true_proportion)

class EntrophyConstraint(Constraint):
    def __init__(self, alpha = 0.1,**kwargs):
        self.alpha = alpha
    def calc_loss(self, predictions, targets, idx, device):
        term = F.softmax(predictions, dim=1) * F.log_softmax(predictions, dim=1)
        entrophy = -1.0 *float(self.alpha) * term.sum()
        assert entrophy>=0
        return entrophy

class KnownLabelsConstraint(Constraint):
    name = 'KnowLabels'
    def __init__(self, bag, **kwargs):
        super().__init__(bag, -1)
        self.constraint_class = None


    def calc_loss(self, predictions, targets, idx, device):
        from Main_run import Mode, mode
        bag_idx = self.bag.get_indices_in_data()
        rel_idx = np.isin(idx, bag_idx)
        rel_targets = targets[rel_idx]
        if not len(rel_targets):
            return 0
        if mode == Mode.CLASSIFICATION:
            probabilities = torch.softmax(predictions[rel_idx], dim=1)
            return F.cross_entropy(probabilities, rel_targets)
        else:
            return F.l1_loss(predictions[rel_idx], rel_targets)


class MajorityConstraint(Constraint):
    name = 'Majority'
    def __init__(self, bag):
        super().__init__(bag, -1)
        self.constraint_class = None
        self.major_class = [c for c in self.bag.get_true_proportion() if
                            self.bag.get_true_proportion()[c] == max(self.bag.get_true_proportion().values())][0]

    def calc_loss(self, predictions, targets, idx, device):
        from Main_run import Mode, mode
        bag_idx = self.bag.get_indices_in_data()
        rel_idx = np.isin(idx, bag_idx)
        if not rel_idx.sum():
            return 0
        probabilities = torch.softmax(predictions[rel_idx], dim=1)
        rel_targets = torch.ones(len(probabilities), dtype=targets.dtype)*int(self.major_class)
        return F.cross_entropy(probabilities, rel_targets)

class AverageConstraint(Constraint):
    name = 'Average'
    def __init__(self, bag):
        super().__init__(bag, -1)
        self.constraint_class = None

    def calc_loss(self, predictions, targets, idx, device):
        from Main_run import Mode, mode
        bag_idx = self.bag.get_indices_in_data()
        rel_idx = np.isin(idx, bag_idx)
        if not rel_idx.sum():
            return 0
        rel_targets = torch.ones(len(predictions[rel_idx]), dtype=targets.dtype)*self.bag.get_true_proportion()
        return F.mse_loss(predictions[rel_idx], rel_targets)

class RandomLabelSameProportion(Constraint):
    name = 'RLSP'
    def __init__(self, bag):
        super().__init__(bag, -1)
        self.constraint_class = None

    def calc_loss(self, predictions, targets, idx, device):
        from Main_run import Mode, mode
        bag_idx = self.bag.get_indices_in_data()
        rel_idx = np.isin(idx, bag_idx)
        bag_sample_size = rel_idx.sum()
        if bag_sample_size == 0 :
            return 0
        probabilities = torch.softmax(predictions[rel_idx], dim=1)
        sample_size_intger = {c: int(self.bag.get_true_proportion(c)*bag_sample_size) for c in self.bag.get_true_proportion()}
        need_extra_sample = [c for c in self.bag.get_true_proportion() if self.bag.get_true_proportion(c)*bag_sample_size -
                                 int(self.bag.get_true_proportion(c)*bag_sample_size)>0]
        target_class = []
        remain = bag_sample_size - sum(list(sample_size_intger.values()))
        if remain:
            target_class = np.random.choice(need_extra_sample, remain ).tolist()
        for c, size in sample_size_intger.items():
            target_class += [c] * size
        np.random.shuffle(target_class)
        rel_targets = torch.tensor(target_class, dtype=targets.dtype)
        assert len(rel_targets) == bag_sample_size
        return F.cross_entropy(probabilities, rel_targets)


class UpperBoundProportionConstraint(Constraint):
    def __init__(self, bag, upper_bound = None, constraint_class=None, boundary_spread=0):
        super().__init__(bag, constraint_class)
        true_proportion = self.bag.get_true_proportion()
        if constraint_class is not None:
            true_proportion = true_proportion[constraint_class]
        self.upper_bound = upper_bound if upper_bound is not None else true_proportion + boundary_spread

    def calc_loss(self, predictions, targets, idx, device):
        from Main_run import Mode, mode
        bag_idx = self.bag.get_indices_in_data()
        rel_idx = np.isin(idx, bag_idx)
        if not rel_idx.sum():
            return 0
        if mode ==Mode.CLASSIFICATION:
            probabilities = torch.softmax(predictions[rel_idx], dim=1)
            predicted_proportion = torch.mean(probabilities[:, self.constraint_class])
        else:
            predicted_proportion = torch.mean(predictions[rel_idx])
        loss = torch.max(torch.zeros(1), predicted_proportion-self.upper_bound)
        assert not torch.isnan(loss)
        return loss

class LowerBoundProportionConstraint(Constraint):
    def __init__(self, bag, lower_bound = None, constraint_class=None, boundary_spread = 0):
        super().__init__(bag, constraint_class)
        true_proportion = self.bag.get_true_proportion()
        if constraint_class is not None:
            true_proportion = true_proportion[constraint_class]
        self.lower_bound = lower_bound if lower_bound is not None else true_proportion - boundary_spread

    def calc_loss(self, predictions, targets, idx, device):
        from Main_run import Mode, mode
        bag_idx = self.bag.get_indices_in_data()
        rel_idx = np.isin(idx, bag_idx)
        if not rel_idx.sum():
            return 0
        if mode ==Mode.CLASSIFICATION:
            probabilities = torch.softmax(predictions[rel_idx], dim=1)
            predicted_proportion = torch.mean(probabilities[:, self.constraint_class])
        else:
            predicted_proportion = torch.mean(predictions[rel_idx])
        loss= torch.max(torch.zeros(1), self.lower_bound- predicted_proportion)
        assert not torch.isnan(loss)
        return loss

class TwoBagsAddProportionConstraint(Constraint):
    # bag1 <= bag2 + addition_factor
    def __init__(self, bag1, bag2, addition_factor = None, constraint_class=None, boundary_spread = 0):
        super().__init__(bag1, constraint_class)
        assert addition_factor>=-1 and addition_factor<=1
        self.bag2 = bag2
        true_prop_bag1 = self.bag.get_true_proportion()
        if constraint_class is not None:
            true_prop_bag1 = true_prop_bag1[constraint_class]
        true_prop_bag2 = self.bag2.get_true_proportion()
        if constraint_class is not None:
            true_prop_bag2 = true_prop_bag2[constraint_class]
        diff = true_prop_bag1-true_prop_bag2
        self.addition_factor = addition_factor if addition_factor is not None else  diff + boundary_spread
        
    def calc_loss(self, predictions, targets, idx, device):
        from Main_run import Mode, mode
        bag1_idx = self.bag.get_indices_in_data()
        rel1_idx = np.isin(idx, bag1_idx)
        if mode == Mode.CLASSIFICATION:
            probabilities_bag1 = torch.softmax(predictions[rel1_idx], dim=1)
            predicted_proportion_bag1 = torch.mean(probabilities_bag1[:, self.constraint_class])
        else:
            predicted_proportion_bag1 = torch.mean(predictions[rel1_idx])

        bag2_idx = self.bag2.get_indices_in_data()
        rel2_idx = np.isin(idx, bag2_idx)
        if mode == Mode.CLASSIFICATION:
            probabilities_bag2 = torch.softmax(predictions[rel2_idx], dim=1)
            predicted_proportion_bag2 = torch.mean(probabilities_bag2[:, self.constraint_class])
        else:
            predicted_proportion_bag2 = torch.mean(predictions[rel2_idx])
        return torch.max(torch.zeros(1),  predicted_proportion_bag1-predicted_proportion_bag2-self.addition_factor)

    def get_bags(self):
        return {self.bag, self.bag2}

class TwoBagsMulProportionConstraint(Constraint):
    # bag1 <= bag2*addition_factor
    def __init__(self, bag1, bag2, mul_factor=None, constraint_class=None, boundary_spread=1):
        super().__init__(bag1, constraint_class)
        self.bag2 = bag2
        true_prop_bag1 = self.bag.get_true_proportion()
        if constraint_class is not None:
            true_prop_bag1 = true_prop_bag1[constraint_class]
        true_prop_bag2 = self.bag2.get_true_proportion()
        if constraint_class is not None:
            true_prop_bag2 = true_prop_bag2[constraint_class]
        diff = float(true_prop_bag1)/true_prop_bag2 
        self.mul_factor = mul_factor if mul_factor is not None else diff*boundary_spread

    def calc_loss(self, predictions, targets, idx, device):
        from Main_run import Mode, mode
        bag1_idx = self.bag.get_indices_in_data()
        rel1_idx = np.isin(idx, bag1_idx)
        if mode == Mode.CLASSIFICATION:
            probabilities_bag1 = torch.softmax(predictions[rel1_idx], dim=1)
            predicted_proportion_bag1 = torch.mean(probabilities_bag1[:, self.constraint_class])
        else:
            predicted_proportion_bag1 = torch.mean(predictions[rel1_idx])

        bag2_idx = self.bag2.get_indices_in_data()
        rel2_idx = np.isin(idx, bag2_idx)
        if mode == Mode.CLASSIFICATION:
            probabilities_bag2 = torch.softmax(predictions[rel2_idx], dim=1)
            predicted_proportion_bag2 = torch.mean(probabilities_bag2[:, self.constraint_class])
        else:
            predicted_proportion_bag2 = torch.mean(predictions[rel2_idx])
        return torch.max(torch.zeros(1), predicted_proportion_bag1 - predicted_proportion_bag2*self.mul_factor)

    def get_bags(self):
        return {self.bag, self.bag2}




class ProbabilityCrossEntrophysConstraint(Constraint):
    name = 'ProbCE'
    def __init__(self, bag, **kwargs):
        super().__init__(bag, -1)
        self.constraint_class = None


    def calc_loss(self, predictions, targets, idx, device):
        from Main_run import Mode, mode
        bag_idx = self.bag.get_indices_in_data()
        rel_idx = np.isin(idx, bag_idx)
        rel_targets = targets[rel_idx]
        if not len(rel_targets):
            return 0
        known_probabilities = torch.tensor([list(self.bag.get_true_proportion().values())]*predictions[rel_idx].shape[0])
        predicted_probabilities = torch.log(torch.softmax(predictions[rel_idx], dim=1))
        nbce = -1.0*known_probabilities*predicted_probabilities
        return nbce.sum()
