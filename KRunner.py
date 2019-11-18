
from Constrain import *
from Bags_dataset import *
from Loss import *
from Trainer import *
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict
import  os
import pandas as pd
import csv


class KRunner:
    def __init__(self, k_times, train_params,bags_config, constraint_config, bag_factory, epochs, device, base_lines = [], plot = True,
                 plot_name = 'single_train'):
        self.train_params = train_params
        self.plot = plot
        self.k_times = k_times
        self.trains = []
        self.device= device
        self.bags_config = bags_config
        self.constraint_config  = constraint_config
        self.base_lines = base_lines
        self.base_line_trains = defaultdict(list)
        bags = {}
        for i in range(k_times):
            bags = {bag_p.bag_index: bag_factory.create_bags_from_bag_param(bag_p) for bag_p in self.bags_config}
            bag_factory.reset_bag_index_cocunter()
            constraints = []
            for constraint in self.constraint_config:
                if type(constraint) is not AutomatedConstraintParams:
                    constraints.append(constraint)
                    continue
                constraints.append(constraint.create_constraint(bags))
            bags = self.remove_non_constrainted_bags(bags, constraints)
            assert bags
            self.trains.append(self._create_single_train(bags.values(), constraints))
            for base_line in base_lines:
                self.base_line_trains[base_line.name].append(self._create_base_line_train(base_line, list(bags.values())))
        print('using only {} bags'.format(list(bags.keys())))
        print('total of {} examples in train'.format(len(self.trains[0].train_loader.dataset)))
        self.epochs = epochs
        self.plot_name= plot_name

    def _create_base_line_train(self, base_line, bags):
        base_lines_constraints = [base_line(bag) for bag in bags]
        bsln_train = self._create_single_train(bags, base_lines_constraints)
        bsln_train.name = base_lines_constraints[0].name
        return bsln_train

    def _create_single_train(self, bags, constraints):
        from Main_run import Mode, mode
        name = self.train_params.name
        batch_size = self.train_params.batch_size
        model = self.train_params.nn(len(self.train_params.all_classes)).to(self.device)
        loss = BagsConstrainLoss(constraints)
        if True or mode == Mode.REGRESSION:
            optimizer = optim.SGD(model.parameters(), lr=self.train_params.learning_rate, momentum=self.train_params.moment)
        else:
            optimizer = optim.Adam(model.parameters(), lr= 1e-4)
        bags_train_data_set = self.train_params.dataset(bags, self.train_params.train_data,
                                                self.train_params.train_labels, transform=self.train_params.transform)
        train_data_loader = torch.utils.data.DataLoader(bags_train_data_set,
                                                        batch_size, shuffle=True)
        test_data_loader = torch.utils.data.DataLoader(self.train_params.test_data_set,
                                                  batch_size=self.train_params.test_batch_size, shuffle=True)
        return Trainer(model, loss,optimizer, train_data_loader, test_data_loader,self.device, name)

    def evaluate_avarage_trains_for_singe_epoch(self, list_of_trains):
        train_loss_mean = np.array(list(map(lambda x:x.train_single_epoch(),list_of_trains)))
        test_res = np.array(list(map(lambda x: x.test_single_epoch(), list_of_trains)))
        test_loss_mean = test_res[:,0]
        test_accuracy_mean = test_res[:,1]
        return train_loss_mean.mean(), test_loss_mean.mean(), int(test_accuracy_mean.mean())


    def run(self, plot = True):
        self.train_loss = defaultdict(list)
        self.test_loss = defaultdict(list)
        self.test_accuracy = defaultdict(list)
        total_size_of_test = len(self.trains[0].test_loader.dataset)
        for epoch in range(1, self.epochs+1):
            print('\t### evaluate epoch {} ###'.format(epoch))
            train_loss, test_loss, test_accuracy = self.evaluate_avarage_trains_for_singe_epoch(self.trains)
            self.train_loss[self.trains[0].name].append(train_loss)
            self.test_loss[self.trains[0].name].append(test_loss)
            self.test_accuracy[self.trains[0].name].append(round(test_accuracy*100/total_size_of_test,2))
            # print('\t##### test results for epoch {} : #####'.format(epoch))
            print('\t\tAverage with loss {} : {:.4f}, Accuracy:  [{}/{} ({:.0f}%)]'.format(
                self.trains[0].test_loss.__name__,
                self.test_loss[self.trains[0].name][-1],
                test_accuracy,
                total_size_of_test,
                self.test_accuracy[self.trains[0].name][-1])
            )
            # print('##### base line results for epoch {} : #####'.format(epoch))
            for base_line in self.base_lines:
                bsln_train_loss, bsln_test_loss, bsln_test_accuracy = \
                    self.evaluate_avarage_trains_for_singe_epoch(self.base_line_trains[base_line.name])
                self.train_loss[base_line.name].append(bsln_train_loss)
                self.test_loss[base_line.name].append(bsln_test_loss)
                self.test_accuracy[base_line.name].append(round(bsln_test_accuracy*100/total_size_of_test,2))
                print('\t\tBase line {}, Average with loss {} : {:.4f}, Accuracy: [{}/{} ({:.0f}%)]'.format(
                    base_line.name,
                    self.trains[0].test_loss.__name__,
                    self.test_loss[base_line.name][-1],
                    bsln_test_accuracy,
                    total_size_of_test,
                    self.test_accuracy[base_line.name][-1])
                )
            # if plot and epoch%10 == 0 :
            self.plot_graphs()

    # def run2(self):
    #     self.train_loss_mean = []
    #     self.test_loss_mean = []
    #     test_accuracy_mean = []
    #     self.base_line_loss = []
    #     base_line_correct_examples=[]
    #     for i in range(len(self.base_line_trains)):
    #         self.base_line_loss.append([])
    #         base_line_correct_examples.append([])
    #     total_size_of_test = len(self.trains[0].test_loader.dataset)
    #     for epoch in range(1,self.epochs+1):
    #         print('### start epoch {} ###'.format(epoch))
    #         train_loss_for_current_epoch = []
    #         test_loss_for_current_epoch = []
    #         test_accuracy_for_current_epoch = []
    #         for i, train in enumerate(self.trains):
    #             # print('train {} out of {} trains'.format(i+1, len(self.trains)))
    #             train_loss = train.train_single_epoch(epoch, print = i==0)
    #             test_loss, test_acc = train.test_single_epoch()
    #             train_loss_for_current_epoch.append(train_loss)
    #             test_loss_for_current_epoch.append(test_loss)
    #             test_accuracy_for_current_epoch.append(test_acc)
    #         self.train_loss_mean.append(np.array(train_loss_for_current_epoch).mean())
    #         self.test_loss_mean.append(np.array(test_loss_for_current_epoch).mean())
    #         test_accuracy_mean.append(int(np.array(test_accuracy_for_current_epoch).mean()))
    #         self.test_percentage_accuracy_mean = (100 * np.array(test_accuracy_mean) / total_size_of_test).astype(np.int)
    #         print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, self.train_loss_mean[-1]))
    #         print('Test Epoch {} : Average with loss {} : {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #             epoch, self.trains[0].test_loss.__name__, self.test_loss_mean[-1], test_accuracy_mean[-1], total_size_of_test,
    #             self.test_percentage_accuracy_mean[-1]))
    #         for i, bsln_train in enumerate(self.base_line_trains):
    #             bsln_train_loss = bsln_train.train_single_epoch(epoch, print = False)
    #             bsln_test_loss, bsln_test_correct = bsln_train.test_single_epoch()
    #             self.base_line_loss[i].append(bsln_test_loss)
    #             base_line_correct_examples[i].append(bsln_test_correct)
    #             print('Test Base line {},  Epoch {} : Average with loss {} : {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #                 bsln_train.name, epoch, self.trains[0].test_loss.__name__, bsln_test_loss, bsln_test_correct,
    #                 total_size_of_test, 100*bsln_test_correct/total_size_of_test))
    #         self.base_line_correct_examples = (100*np.array(base_line_correct_examples)/total_size_of_test).astype(np.int).tolist()
    #         self.plot_graphs()

    def plot_graphs(self):
        for values, kind in [(self.test_loss, ' loss '),(self.test_accuracy, ' accuracy ')]:
            for name, results in values.items():
                plt.plot(results, label='test{}{}'.format(kind, name), marker='o')
            plt.legend()
            plt.suptitle(self.plot_name.replace('!',''))
            plt.xlabel( 'epoch')
            plt.ylabel(kind)
            dir_name = os.path.join('3dplots', self.plot_name.split('!')[0])
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            plt.savefig(os.path.join(dir_name, self.plot_name.replace('!','') + kind))
            plt.clf()
            csv_file = os.path.join(dir_name, self.plot_name.replace('!','') + kind) +'.csv'
            pd.DataFrame(values).to_csv(csv_file)




    def remove_non_constrainted_bags(self, bags, constraints):
        constraints_bags = set()
        for constraint in constraints:
            try:
                constraint.get_bags()
            except:
                continue
            constraints_bags = constraints_bags.union(constraint.get_bags())
        # constraints_bags_idx = set([bag.bag_idx for bag in constraints_bags])
        refactor_bags = {}
        for bag_key, bag in bags.items():
            if bag in constraints_bags:
                refactor_bags[bag_key] =bag
        return refactor_bags