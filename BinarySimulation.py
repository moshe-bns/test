from Bag import *
from Constrain import *
from KRunner import *
from matplotlib import pyplot as plt

class BinarySimulaion:
    def _create_single_bag_config(self, num_of_bags,bag_size,true_p):
        from Bag import BagParam
        config = []
        for i in range(num_of_bags):
            config.append(BagParam(i, bag_size, {0:1-true_p, 1:true_p}))
            config = config + self.additional_bag
            # config.append(BagParam(num_of_bags+i, bag_size, {0:true_p, 1:1-true_p}))
        return [config]

    def __init__(self, N_range, S_range, P_range, k_times, train_params, bag_factory, n_epochs,device,
                 constratin_tp_bias=0, constraint_wide=0, base_lines = [],additional_bag = [],
                 name = 'simulation', create_singe_bag_func =None ):
        self.k_times = k_times
        self.train_params=train_params
        self.bag_factory=bag_factory
        self.n_epochs=n_epochs
        self.device = device
        self.base_lines=base_lines
        self.name = name
        self.additional_bag = additional_bag
        bags_configs = []
        create_singe_bag_func = create_singe_bag_func if create_singe_bag_func is not None else self._create_single_bag_config
        if len (N_range) != 1:
            param_name = 'number of bags'
            relevant_range = N_range
            for x in N_range:
                bags_configs =bags_configs + create_singe_bag_func(x, S_range[0], P_range[0])
        elif len (S_range) != 1:
            param_name = 'size of bags'
            relevant_range = S_range
            for x in S_range:
                bags_configs =bags_configs + create_singe_bag_func(N_range[0],x, P_range[0])
        else:
            param_name = 'proportion in bags'
            relevant_range = P_range
            for x in P_range:
                bags_configs =bags_configs +create_singe_bag_func(N_range[0],S_range[0], x)
        self.param_name = param_name
        self.relevant_range = relevant_range
        constraint_configs = []
        for bc in bags_configs:
            constraint_config = []
            for i, bag_param in enumerate(bc):
                upper_bound = round(bag_param.sample_dict[1]+constratin_tp_bias + constraint_wide/2,2)
                constraint_config.append(AutomatedConstraintParams(bag_param.bag_index,
                                                                   UpperBoundProportionConstraint,
                                                                   constraint_class = 1,
                                                                   upper_bound= upper_bound))
                lower_bound = round(bag_param.sample_dict[1]+constratin_tp_bias - constraint_wide/2,2)
                constraint_config.append(AutomatedConstraintParams(bag_param.bag_index,
                                                               LowerBoundProportionConstraint,
                                                               constraint_class = 1,
                                                                lower_bound= lower_bound))
            constraint_configs.append(constraint_config)
        assert len(constraint_configs) == len(bags_configs)
        self.constraint_configs = constraint_configs
        self.bags_configs = bags_configs

    def run(self):
        print('print simulation {}'.format(self.name))
        results_loss = []
        results_acc = []
        for i, element in enumerate(self.relevant_range):
            print('#### {} {} ####'.format(self.param_name, element))
            runner = KRunner(self.k_times, self.train_params, self.bags_configs[i], self.constraint_configs[i],
                             self.bag_factory, self.n_epochs,self.device, self.base_lines,
                             plot_name='{}!_{}_{}'.format(self.name,self.param_name,element).replace('.',''))
            runner.run()
            curr_loss_res = [runner.test_loss[self.train_params.name][-1]]
            curr_acc_res = [runner.test_accuracy[self.train_params.name][-1]]
            for bsln in self.base_lines:
                curr_loss_res.append(runner.test_loss[bsln.name][-1])
                curr_acc_res.append(runner.test_accuracy[bsln.name][-1])
            results_loss.append(curr_loss_res)
            results_acc.append(curr_acc_res)
            self.plot_graphs(results_acc,results_loss)

    def plot_graphs(self, results_acc, results_loss):
        np_res_acc = np.array(results_acc)
        np_res_loss = np.array(results_loss)
        for values, kind in [(np_res_loss, ' loss '),(np_res_acc, ' accuracy ')]:
            for i in range(values.shape[1]):
                label = 'my_train'
                if i !=0:
                    label = self.base_lines[i-1].name
                plt.plot(self.relevant_range[:len(values[:,i].tolist())], values[:,i].tolist(), label =label,marker='o')
            plt.xlabel(self.param_name)
            plt.ylabel(kind)
            plt.legend()
            dir_name = os.path.join('3dplots',self.name)
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            plt.savefig(os.path.join(dir_name,self.name + kind))
            plt.clf()



