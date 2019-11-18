from sklearn.svm import classes

from Bag import *
import torch
from BinarySimulation import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
from NN_Options import *
from Trainer import *
from Tester import *
from Bags_dataset import *
from Loss import *
from Constrain import *
from KRunner import *
from enum import Enum
from sklearn.preprocessing import normalize
import pickle

class Mode(Enum):
    REGRESSION=1
    CLASSIFICATION=2

mode = Mode.CLASSIFICATION
torch.set_printoptions(sci_mode=False)

def remove_irrelevant_classes(data_set, all_classes):
    relevant_indices = np.isin(data_set.targets, all_classes)
    data_set.data = data_set.data[relevant_indices]
    data_set.targets = data_set.targets[relevant_indices]
    return data_set


def create_single_BC_add_const_bag_02(num_of_bags,bag_size,true_p):
    const_true_p = 0.2
    from Bag import BagParam
    config = []
    true_p = round(true_p,2)
    for i in range(num_of_bags):
        config.append(BagParam(i, bag_size, {0:1-true_p, 1:true_p}))
        config.append(BagParam(num_of_bags+i, bag_size, {0:1-const_true_p, 1:const_true_p}))
    return [config]

def create_single_BC_add_const_bag_053(num_of_bags,bag_size,true_p):
    const_true_p = 0.53
    from Bag import BagParam
    config = []
    true_p = round(true_p,2)
    for i in range(num_of_bags):
        config.append(BagParam(i, bag_size, {0:1-true_p, 1:true_p}))
        config.append(BagParam(num_of_bags+i, bag_size, {0:1-const_true_p, 1:const_true_p}))
    return [config]

if __name__=='__main__':

    ##  General Params
    device = torch.device('cpu')
    k_times = 15
    batch_size = 32
    n_epochs = 50
    learning_rate = 0.01
    moment = 0.5
    test_batch_size = 1000


    ##  End Of General Params

    ## Data Generation
    if mode == Mode.CLASSIFICATION:
        all_classes = [0, 1]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data_set = remove_irrelevant_classes(datasets.MNIST('../data', train=True, download=True,
                                                                  transform=transform), all_classes)
        x_train = train_data_set.data
        y_train = train_data_set.targets
        test_data_set = remove_irrelevant_classes(datasets.MNIST('../data', train=False, download=True,
                                                                 transform=transform), all_classes)
        dataset = BagsDatasetVision
    else:
        boston = load_boston()
        transform = None
        boston_df = pd.DataFrame(normalize(boston['data']))
        boston_df['PRICE'] =boston['target']
        x_train, x_test, y_train, y_test = train_test_split(boston_df.iloc[:,0:13], boston_df['PRICE'], test_size=0.2, random_state=0)
        x_train = torch.from_numpy(np.array(x_train, dtype=np.float64)).float()
        y_train = torch.from_numpy(np.array(y_train, dtype=np.float64)).float()
        x_test = torch.from_numpy(np.array(x_test, dtype=np.float64)).float()
        y_test = torch.from_numpy(np.array(y_test, dtype=np.float64)).float()
        test_data_set = SimpleCustomData(x_test,y_test)
        dataset = BagsDataset
        all_classes = x_train[0]

    ## End Of Data Generation

    ## Bags Configuration
    bag0 = BagParam(0, 15, {0:0.33, 1:0.67})
    bag01 = BagParam(108, 100, {0:0.2, 1:0.8})
    bag02 = BagParam(102, 100, {0:0.8, 1:0.2})
    bag03 = BagParam(106, 100, {0:0.4, 1:0.6})
    bag04 = BagParam(104, 100, {0:0.6, 1:0.4})
    bag1 = BagParam(1, 100, {0:0.4, 1:0.6})
    # bag2 = BagParam(2, 10, {0:0.1, 1:0.9})
    bag2 = BagParam(2, 100, {0:0.4, 1:0.6})
    bag3 = BagParam(3, 16, {0:0.2, 1:0.8})
    bag4 = BagParam(4, 100, {0:0.6, 1:0.4})
    bag5 = BagParam(5, 40, {0:0.5, 1:0.5})
    bag6 = BagParam(6, 40, {0:0.75, 1:0.25})
    bag7 = BagParam(7, 32, {0:0.65, 1:0.35})
    bag8 = BagParam(8, 5, {0:1, 1:0})

    bag9 = BagParam(9, 15, {0:0.3, 1:0.2, 2:0.5})
    bag10 = BagParam(10, 160,  {0:0.25, 1:0.3, 2:0.45})
    bag11 = BagParam(11, 240,  {0:0.5, 1:0.1, 2:0.4})
    bag12 = BagParam(12, 520,  {0:0.1, 1:0.65, 2:0.25})
    two_class_bags_config = [bag01, bag02, bag03, bag04,bag0,bag1,bag2,bag3,bag4,bag5,bag6,bag7,bag8]
    three_class_bags_config = [bag9, bag10,bag11,bag12]

    reg_bag0 = BagParam(0, 30)
    reg_bag1 = BagParam(1, 40)
    reg_bag2 = BagParam(2, 30)
    reg_bag_config = [reg_bag0, reg_bag1, reg_bag2]
    bag_factory = BagFactory(x_train, y_train)
    ## End Of Bags Configuration

    ## Constraints Configuration

    ## two classes
    if len(all_classes)==2:
        boundary_spread = 0.1

        two_known_labels_constraint0 = AutomatedConstraintParams(0, KnownLabelsConstraint)
        two_known_labels_constraint1 = AutomatedConstraintParams(1, KnownLabelsConstraint)
        two_known_labels_constraint2 = AutomatedConstraintParams(2, KnownLabelsConstraint)
        classification_constraints_2known_labels = [two_known_labels_constraint0, two_known_labels_constraint1,
                                                   two_known_labels_constraint2]

        two_Exact_prop_constraint0 = AutomatedConstraintParams(0, ExactProportionConstraint,constraint_class = 1)
        two_Exact_prop_constraint1 = AutomatedConstraintParams(1, ExactProportionConstraint,constraint_class = 1)
        two_Exact_prop_constraint2 = AutomatedConstraintParams(2, ExactProportionConstraint,constraint_class = 1)
        classification_constraints_2exact_prop = [two_Exact_prop_constraint0, two_Exact_prop_constraint1,
                                                  two_Exact_prop_constraint2]


        two_upper_prop_constraint0 =  AutomatedConstraintParams(0, UpperBoundProportionConstraint, constraint_class = 1, boundary_spread=boundary_spread)
        two_upper_prop_constraint1 =  AutomatedConstraintParams(1, UpperBoundProportionConstraint, constraint_class = 1, boundary_spread=boundary_spread)
        two_upper_prop_constraint2 =  AutomatedConstraintParams(2, UpperBoundProportionConstraint, constraint_class = 1, boundary_spread=boundary_spread)
        two_upper_prop_constraint102 =  AutomatedConstraintParams(102, UpperBoundProportionConstraint, constraint_class = 1, boundary_spread=boundary_spread)
        two_upper_prop_constraint104 =  AutomatedConstraintParams(104, UpperBoundProportionConstraint, constraint_class = 1, boundary_spread=boundary_spread)
        two_upper_prop_constraint106 =  AutomatedConstraintParams(106, UpperBoundProportionConstraint, constraint_class = 1, boundary_spread=boundary_spread)
        two_upper_prop_constraint108 =  AutomatedConstraintParams(108, UpperBoundProportionConstraint, constraint_class = 1, boundary_spread=boundary_spread)
        classification_constraints_2upper_prop = [two_upper_prop_constraint0, two_upper_prop_constraint1,
                                                  two_upper_prop_constraint2]

        two_lower_prop_constraint0 =  AutomatedConstraintParams(0, LowerBoundProportionConstraint, constraint_class = 1, boundary_spread=boundary_spread)
        two_lower_prop_constraint1 =  AutomatedConstraintParams(1, LowerBoundProportionConstraint, constraint_class = 1, boundary_spread=boundary_spread)
        two_lower_prop_constraint2 =  AutomatedConstraintParams(2, LowerBoundProportionConstraint, constraint_class = 1, boundary_spread=boundary_spread)
        two_lower_prop_constraint102 =  AutomatedConstraintParams(102, LowerBoundProportionConstraint, constraint_class = 1, boundary_spread=boundary_spread)
        two_lower_prop_constraint104 =  AutomatedConstraintParams(104, LowerBoundProportionConstraint, constraint_class = 1, boundary_spread=boundary_spread)
        two_lower_prop_constraint106 =  AutomatedConstraintParams(106, LowerBoundProportionConstraint, constraint_class = 1, boundary_spread=boundary_spread)
        two_lower_prop_constraint108 =  AutomatedConstraintParams(108, LowerBoundProportionConstraint, constraint_class = 1, boundary_spread=boundary_spread)
        classification_constraints_2lower_prop =[two_lower_prop_constraint0, two_lower_prop_constraint1,
                                                 two_lower_prop_constraint2]

        constraints_config_single108 =[two_lower_prop_constraint108, two_upper_prop_constraint108]
        constraints_config_single106 =[two_lower_prop_constraint106, two_upper_prop_constraint106]
        constraints_config_single104 =[two_lower_prop_constraint104, two_upper_prop_constraint104]
        constraints_config_single102 =[two_lower_prop_constraint102, two_upper_prop_constraint102]
        constraints_config_double_106_104 = constraints_config_single106 + constraints_config_single104
        constraints_config_double_108_102 = constraints_config_single108 + constraints_config_single102
        constraints_config_double_106_108 = constraints_config_single106 + constraints_config_single108
        constraints_config_double_106_102 = constraints_config_single106 + constraints_config_single102

    ## three classes
    else:
        boundary_spread = 0.1

        three_known_labels_constraint0 = AutomatedConstraintParams(9, KnownLabelsConstraint)
        three_known_labels_constraint1 = AutomatedConstraintParams(10, KnownLabelsConstraint)
        three_known_labels_constraint2 = AutomatedConstraintParams(11, KnownLabelsConstraint)
        three_known_labels_constraint3 = AutomatedConstraintParams(12, KnownLabelsConstraint)
        classification_constraints_3known_labels = [three_known_labels_constraint0, three_known_labels_constraint1,
                                                   three_known_labels_constraint2, three_known_labels_constraint3]

        three_Exact_prop_constraint0 = AutomatedConstraintParams(9, ExactProportionConstraint,constraint_class = 0)
        three_Exact_prop_constraint1 = AutomatedConstraintParams(9, ExactProportionConstraint,constraint_class = 1)
        three_Exact_prop_constraint2 = AutomatedConstraintParams(9, ExactProportionConstraint,constraint_class = 2)
        three_Exact_prop_constraint3 = AutomatedConstraintParams(10, ExactProportionConstraint,constraint_class = 0)
        three_Exact_prop_constraint4 = AutomatedConstraintParams(10, ExactProportionConstraint,constraint_class = 1)
        three_Exact_prop_constraint5 = AutomatedConstraintParams(10, ExactProportionConstraint,constraint_class = 2)
        three_Exact_prop_constraint6 = AutomatedConstraintParams(11, ExactProportionConstraint,constraint_class = 0)
        three_Exact_prop_constraint7 = AutomatedConstraintParams(11, ExactProportionConstraint,constraint_class = 1)
        three_Exact_prop_constraint8 = AutomatedConstraintParams(11, ExactProportionConstraint,constraint_class = 2)
        three_Exact_prop_constraint9 = AutomatedConstraintParams(12, ExactProportionConstraint,constraint_class = 0)
        three_Exact_prop_constraint10 = AutomatedConstraintParams(12, ExactProportionConstraint,constraint_class = 1)
        three_Exact_prop_constraint11 = AutomatedConstraintParams(12, ExactProportionConstraint,constraint_class = 2)
        classification_constraints_3exact_prop = [three_Exact_prop_constraint0, three_Exact_prop_constraint1,
                                                  three_Exact_prop_constraint2, three_Exact_prop_constraint3,
                                                  three_Exact_prop_constraint4, three_Exact_prop_constraint5,
                                                  three_Exact_prop_constraint6, three_Exact_prop_constraint7,
                                                  three_Exact_prop_constraint8, three_Exact_prop_constraint9,
                                                  three_Exact_prop_constraint10, three_Exact_prop_constraint11]

        three_upper_prop_constraint0 = AutomatedConstraintParams(9, UpperBoundProportionConstraint,constraint_class = 0,
                                                                 boundary_spread=boundary_spread)
        three_upper_prop_constraint1 = AutomatedConstraintParams(9, UpperBoundProportionConstraint,constraint_class = 1,
                                                                 boundary_spread=boundary_spread)
        three_upper_prop_constraint2 = AutomatedConstraintParams(9, UpperBoundProportionConstraint,constraint_class = 2,
                                                                 boundary_spread=boundary_spread)
        three_upper_prop_constraint3 = AutomatedConstraintParams(10, UpperBoundProportionConstraint,constraint_class = 0,
                                                                 boundary_spread=boundary_spread)
        three_upper_prop_constraint4 = AutomatedConstraintParams(10, UpperBoundProportionConstraint,constraint_class = 1,
                                                                 boundary_spread=boundary_spread)
        three_upper_prop_constraint5 = AutomatedConstraintParams(10, UpperBoundProportionConstraint,constraint_class = 2,
                                                                 boundary_spread=boundary_spread)
        three_upper_prop_constraint6 = AutomatedConstraintParams(11, UpperBoundProportionConstraint,constraint_class = 0,
                                                                 boundary_spread=boundary_spread)
        three_upper_prop_constraint7 = AutomatedConstraintParams(11, UpperBoundProportionConstraint,constraint_class = 1,
                                                                 boundary_spread=boundary_spread)
        three_upper_prop_constraint8 = AutomatedConstraintParams(11, UpperBoundProportionConstraint,constraint_class = 2,
                                                                 boundary_spread=boundary_spread)
        three_upper_prop_constraint9 = AutomatedConstraintParams(12, UpperBoundProportionConstraint,constraint_class = 0,
                                                                 boundary_spread=boundary_spread)
        three_upper_prop_constraint10 = AutomatedConstraintParams(12, UpperBoundProportionConstraint,constraint_class = 1,
                                                                 boundary_spread=boundary_spread)
        three_upper_prop_constraint11 = AutomatedConstraintParams(12, UpperBoundProportionConstraint,constraint_class = 2,
                                                                 boundary_spread=boundary_spread)
        classification_constraints_3upper_prop =[ three_upper_prop_constraint0, three_upper_prop_constraint1,
                                                  three_upper_prop_constraint2, three_upper_prop_constraint3,
                                                  three_upper_prop_constraint4, three_upper_prop_constraint5,
                                                  three_upper_prop_constraint5, three_upper_prop_constraint7,
                                                  three_upper_prop_constraint8, three_upper_prop_constraint9,
                                                  three_upper_prop_constraint10, three_upper_prop_constraint11]

        three_lower_prop_constraint0 = AutomatedConstraintParams(9, LowerBoundProportionConstraint,constraint_class = 0,
                                                                 boundary_spread=boundary_spread)
        three_lower_prop_constraint1 = AutomatedConstraintParams(9, LowerBoundProportionConstraint,constraint_class = 1,
                                                                 boundary_spread=boundary_spread)
        three_lower_prop_constraint2 = AutomatedConstraintParams(9, LowerBoundProportionConstraint,constraint_class = 2,
                                                                 boundary_spread=boundary_spread)
        three_lower_prop_constraint3 = AutomatedConstraintParams(10, LowerBoundProportionConstraint,constraint_class = 0,
                                                                 boundary_spread=boundary_spread)
        three_lower_prop_constraint4 = AutomatedConstraintParams(10, LowerBoundProportionConstraint,constraint_class = 1,
                                                                 boundary_spread=boundary_spread)
        three_lower_prop_constraint5 = AutomatedConstraintParams(10, LowerBoundProportionConstraint,constraint_class = 2,
                                                                 boundary_spread=boundary_spread)
        three_lower_prop_constraint6 = AutomatedConstraintParams(11, LowerBoundProportionConstraint,constraint_class = 0,
                                                                 boundary_spread=boundary_spread)
        three_lower_prop_constraint7 = AutomatedConstraintParams(11, LowerBoundProportionConstraint,constraint_class = 1,
                                                                 boundary_spread=boundary_spread)
        three_lower_prop_constraint8 = AutomatedConstraintParams(11, LowerBoundProportionConstraint,constraint_class = 2,
                                                                 boundary_spread=boundary_spread)
        three_lower_prop_constraint9 = AutomatedConstraintParams(12, LowerBoundProportionConstraint,constraint_class = 0,
                                                                 boundary_spread=boundary_spread)
        three_lower_prop_constraint10 = AutomatedConstraintParams(12, LowerBoundProportionConstraint,constraint_class = 1,
                                                                 boundary_spread=boundary_spread)
        three_lower_prop_constraint11 = AutomatedConstraintParams(12, LowerBoundProportionConstraint,constraint_class = 2,
                                                                 boundary_spread=boundary_spread)
        classification_constraints_3lower_prop =[ three_lower_prop_constraint0, three_lower_prop_constraint1,
                                                  three_lower_prop_constraint2, three_lower_prop_constraint3,
                                                  three_lower_prop_constraint4, three_lower_prop_constraint5,
                                                  three_lower_prop_constraint5, three_lower_prop_constraint7,
                                                  three_lower_prop_constraint8, three_lower_prop_constraint9,
                                                  three_lower_prop_constraint10, three_lower_prop_constraint11]





    ## regression

    boundary_spread_reg = 0
    reg_constraint0 = AutomatedConstraintParams(0, KnownLabelsConstraint)
    reg_constraint1 = AutomatedConstraintParams(0, ExactProportionConstraint)
    reg_constraint2 = AutomatedConstraintParams(1, ExactProportionConstraint)
    reg_constraint3 = AutomatedConstraintParams(2, ExactProportionConstraint)
    reg_constraint4 = AutomatedConstraintParams(0, UpperBoundProportionConstraint, boundary_spread=boundary_spread_reg)
    reg_constraint5 = AutomatedConstraintParams(0, LowerBoundProportionConstraint, boundary_spread=boundary_spread_reg)
    reg_constraint6 = AutomatedConstraintParams(1, UpperBoundProportionConstraint, boundary_spread=boundary_spread_reg)
    reg_constraint7 = AutomatedConstraintParams(1, LowerBoundProportionConstraint, boundary_spread=boundary_spread_reg)
    reg_constraint8 = AutomatedConstraintParams(2, UpperBoundProportionConstraint, boundary_spread=boundary_spread_reg)
    reg_constraint9 = AutomatedConstraintParams(2, LowerBoundProportionConstraint, boundary_spread=boundary_spread_reg)
    reg_constraint_config = [reg_constraint0, reg_constraint1, reg_constraint2, reg_constraint3, reg_constraint4,
                             reg_constraint5, reg_constraint6, reg_constraint7, reg_constraint8, reg_constraint9]
    ## End Of Constraints Configuration


    if mode == Mode.CLASSIFICATION:
        if len(all_classes) == 2:
            bags_config = two_class_bags_config
            constraint_config = classification_constraints_2upper_prop[:1] + classification_constraints_2lower_prop[:1]
            constraint_config2 = classification_constraints_2upper_prop[:2] + classification_constraints_2lower_prop[:2]
            # constraint_config = classification_constraints_2exact_prop[:1]
            # constraint_config = classification_constraints_2known_labels[:2]
            # entrophy_constraint = EntrophyConstraint(0.2/(2*batch_size))
            # constraint_config = constraint_config+[entrophy_constraint]
        elif len(all_classes) == 3:
            bags_config = three_class_bags_config
            constraint_config = classification_constraints_3known_labels[:1] + classification_constraints_3upper_prop + classification_constraints_3lower_prop
            entrophy_constraint = EntrophyConstraint(1/(2*batch_size))
            constraint_config = constraint_config+[entrophy_constraint]
        else:
            assert 0
        nn = ClassificationNet
        base_lines = [ProbabilityCrossEntrophysConstraint, MajorityConstraint, RandomLabelSameProportion, KnownLabelsConstraint]
        # base_lines = [KnownLabelsConstraint]
        # base_lines = [MajorityConstraint]
        # base_lines = []
    else:
        bags_config = reg_bag_config
        constraint_config =reg_constraint_config[6:8] #+ [reg_constraint0]
        nn = RegNet
        base_lines =[AverageConstraint]



    train_params = TrainParams(
        dataset = dataset,
        train_data=x_train,
        train_labels=y_train,
        nn = nn,
        loss = BagsConstrainLoss,
        optimizer = optim.SGD,
        batch_size = batch_size,
        test_data_set=test_data_set,
        test_batch_size = test_batch_size,
        all_classes = all_classes,
        transform = transform,
        learning_rate=0.01,
        moment=0.5
    )
    k_times = 10
    n_epochs = 10
    for constraint_conf, name in [
        # (constraints_config_single108, 'single_bag_truep_08'),
        # (constraints_config_single106,'single_bag_truep_06'),
        # (constraints_config_double_106_104,'two_bags_truep_06_04'),
        # (constraints_config_double_108_102,'two_bags_truep_08_02'),
        # (constraints_config_double_106_108,'two_bags_truep_06_08'),
        # (constraints_config_double_106_102,'two_bags_truep_06_02')
                           ]:
        r = KRunner(k_times, train_params, bags_config, constraint_conf, bag_factory, n_epochs,device= device,
                        base_lines=base_lines, plot_name=name)
        r.run()


    my_true_p_range = [0.05, 0.1,0.2,0.3,0.35,0.4,0.45,0.48,0.5,0.52,0.55,0.6,0.65,0.7,0.8,0.9,0.95]
    for p in [0.4]+my_true_p_range[8:] :
        print('p1 = {}'.format(p))
        simulation = BinarySimulaion(N_range = [1], S_range=[100], P_range= my_true_p_range, k_times = 15,
                                     train_params=train_params, bag_factory = bag_factory, n_epochs=20,
                                     constratin_tp_bias=0, constraint_wide=0.1, base_lines = base_lines,
                                     create_singe_bag_func=None,
                                     device = device,
                                     additional_bag = [BagParam(100, 100, {0:1-p, 1:p})],
                                     name='grid-2d-interval-01-simul-p={}_truep_range'.format(str(p).replace('.', '')))
        simulation.run()

    for p in my_true_p_range :
        simulation = BinarySimulaion(N_range = [1], S_range=[100], P_range= my_true_p_range, k_times = 15,
                                     train_params=train_params, bag_factory = bag_factory, n_epochs=20,
                                     constratin_tp_bias=0, constraint_wide=0.2, base_lines = base_lines,
                                     create_singe_bag_func=None,
                                     device=device,
                                     additional_bag = [BagParam(100, 100, {0:1-p, 1:p})],
                                     name='grid-2d-interval-02-simul-p={}_truep_range'.format(str(p).replace('.', '')))
        simulation.run()
#
    for p in my_true_p_range :
        simulation = BinarySimulaion(N_range = [1], S_range=[100], P_range= my_true_p_range, k_times = 15,
                                     train_params=train_params, bag_factory = bag_factory, n_epochs=20,
                                     constratin_tp_bias=0, constraint_wide=0.3, base_lines = base_lines,
                                     create_singe_bag_func=None,
                                     additional_bag = [BagParam(100, 100, {0:1-p, 1:p})],
                                     name='grid-2d-interval-03-simul-p={}_truep_range'.format(str(p).replace('.', '')))
        simulation.run()

    for p in my_true_p_range :
        simulation = BinarySimulaion(N_range = [1], S_range=[100], P_range= my_true_p_range, k_times = 15,
                                     train_params=train_params, bag_factory = bag_factory, n_epochs=20,
                                     constratin_tp_bias=0, constraint_wide=0.4, base_lines = base_lines,
                                     create_singe_bag_func=None,
                                     additional_bag = [BagParam(100, 100, {0:1-p, 1:p})],
                                     name='grid-2d-interval-04-simul-p={}_truep_range'.format(str(p).replace('.', '')))
        simulation.run()

    my_true_p2_range = reversed([ 0.1,0.2, 0.4, 0.5, 0.6, 0.8,0.9])
    for p2 in my_true_p2_range:
        print('p2 = {}'. format(p2))
        for p in my_true_p_range:
            simulation = BinarySimulaion(N_range=[1], S_range=[100], P_range=my_true_p_range, k_times=15,
                                         train_params=train_params, bag_factory=bag_factory, n_epochs=20,
                                         constratin_tp_bias=0, constraint_wide=0.2, base_lines=base_lines,
                                         create_singe_bag_func=None,
                                         device=device,
                                         additional_bag=[BagParam(100, 100, {0: 1 - p, 1: p}),
                                                         BagParam(200, 100, {0: 1 - p2, 1: p2})],
                                         name='grid-3d-interval-02-simul-p2={}_p1={}_truep_range'.format(
                                             str(p2).replace('.', ''), str(p).replace('.', '')))
            simulation.run()

    # simulation = BinarySimulaion(N_range = [1], S_range=[5,10,30,80, 120, 500, 1000, 2000], P_range=[0.7], k_times = k_times,
    #                              train_params=train_params, bag_factory = bag_factory, n_epochs=20,
    #                              constratin_tp_bias=0, constraint_wide=0.4, base_lines = base_lines,
    #                              create_singe_bag_func=None,
    #                              name='simul-bag_size_04_interval')
    # simulation.run()
    #
    # simulation = BinarySimulaion(N_range = [1,2,4,8,16,32,64], S_range=[100], P_range=[0.7], k_times = k_times,
    #                              train_params=train_params, bag_factory = bag_factory, n_epochs=5,
    #                              constratin_tp_bias=0, constraint_wide=0.4, base_lines = base_lines,
    #                              create_singe_bag_func=None,
    #                              name='simul-number_of_bags_04_interval')
    # simulation.run()
    #
    # simulation = BinarySimulaion(N_range = [1], S_range=[5,10,30,80,120], P_range=[0.7], k_times = k_times,
    #                              train_params=train_params, bag_factory = bag_factory, n_epochs=15,
    #                              constratin_tp_bias=0, constraint_wide=0.2, base_lines = base_lines,
    #                              create_singe_bag_func=None,
    #                              name='simul-bag_size_02_interval')
    # simulation.run()
    #
    # simulation = BinarySimulaion(N_range = [1], S_range=[100], P_range=np.arange(0.2, 1.0, 0.1).tolist(), k_times = k_times,
    #                              train_params=train_params, bag_factory = bag_factory, n_epochs=10,
    #                              constratin_tp_bias=0, constraint_wide=0.2, base_lines = base_lines,
    #                              create_singe_bag_func=create_single_BC_add_const_bag_02,
    #                              name='simul-truep_02_truep_range02-1')
    # simulation.run()
    #
    # simulation = BinarySimulaion(N_range = [1], S_range=[5,10,30,80, 120, 500, 1000, 2000], P_range=[0.7], k_times = k_times,
    #                              train_params=train_params, bag_factory = bag_factory, n_epochs=15,
    #                              constratin_tp_bias=0, constraint_wide=0.3, base_lines = base_lines,
    #                              create_singe_bag_func=None,
    #                              name='simul-bag_size_03_interval')
    # simulation.run()
    #
    # simulation = BinarySimulaion(N_range = [1,2,4,8,16,32,64], S_range=[100], P_range=[0.7], k_times = k_times,
    #                              train_params=train_params, bag_factory = bag_factory, n_epochs=10,
    #                              constratin_tp_bias=0, constraint_wide=0.5, base_lines = base_lines,
    #                              create_singe_bag_func=None,
    #                              name='simul-number_of_bags_05_interval')
    # simulation.run()