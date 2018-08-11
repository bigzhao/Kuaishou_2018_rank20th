import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import KFold
import gc
import os

feature_1 = pd.read_csv('/tmp/feature_1_v1.csv')
feature_2 = pd.read_csv('/tmp/feature_2_v1.csv')

target_1 = np.load('./target_1.npy')
target_2 = np.load('./target_2.npy')

class BGA():
    """
    Simple 0-1 genetic algorithm.
    User Guide:
    >> test = GA(pop_shape=(10, 10), method=np.sum)
    >> solution, fitness = test.run()
    """
    def __init__(self, pop_shape, method, p_c=0.8, p_m=0.2, max_round = 1000, early_stop_rounds=None, verbose = None, maximum=True):
        """
        Args:
            pop_shape: The shape of the population matrix.
            method: User-defined medthod to evaluate the single individual among the population.
                    Example:

                    def method(arr): # arr is a individual array
                        return np.sum(arr)

            p_c: The probability of crossover.
            p_m: The probability of mutation.
            max_round: The maximun number of evolutionary rounds.
            early_stop_rounds: Default is None and must smaller than max_round.
            verbose: 'None' for not printing progress messages. int type number for printing messages every n iterations.
            maximum: 'True' for finding the maximum value while 'False' for finding the minimum value.
        """
        if early_stop_rounds != None:
            assert(max_round > early_stop_rounds)
        self.pop_shape = pop_shape
        self.method = method
        self.pop = np.zeros(pop_shape)
        self.fitness = np.zeros(pop_shape[0])
        self.p_c = p_c
        self.p_m = p_m
        self.max_round = max_round
        self.early_stop_rounds = early_stop_rounds
        self.verbose = verbose
        self.maximum = maximum

    def evaluation(self, pop):
        """
        Computing the fitness of the input popluation matrix.
        Args:
            p: The population matrix need to be evaluated.
        """
        return np.array([self.method(i) for i in pop])

    def initialization(self):
        """
        Initalizing the population which shape is self.pop_shape(0-1 matrix).
        """
        if os.path.exists('pop.npy'):
            self.pop = np.load('pop.npy')
        else:
            self.pop = np.random.randint(low=0, high=2, size=self.pop_shape)
            np.save('pop.npy', self.pop)
        self.fitness = self.evaluation(self.pop)

    def crossover(self, ind_0, ind_1):
        """
        Single point crossover.
        Args:
            ind_0: individual_0
            ind_1: individual_1
        Ret:
            new_0, new_1: the individuals generatd after crossover.
        """
        assert(len(ind_0) == len(ind_1))

        point = np.random.randint(len(ind_0))
#         new_0, new_1 = np.zeros(len(ind_0)),  np.zeros(len(ind_0))
        new_0 = np.hstack((ind_0[:point], ind_1[point:]))
        new_1 = np.hstack((ind_1[:point], ind_0[point:]))

        assert(len(new_0) == len(ind_0))
        return new_0, new_1

    def mutation(self, indi):
        """
        Simple mutation.
        Arg:
            indi: individual to mutation.
        """
        point = np.random.randint(len(indi))
        indi[point] = 1 - indi[point]
        return indi


    def rws(self, size, fitness):
        """
        Roulette Wheel Selection.
        Args:
            size: the size of individuals you want to select according to their fitness.
            fitness: the fitness of population you want to apply rws to.
        """
        if self.maximum:
            fitness_ = fitness
        else:
            fitness_ = 1.0 / fitness
#         fitness_ = fitness
        idx = np.random.choice(np.arange(len(fitness_)), size=size, replace=True,
               p=fitness_/fitness_.sum()) # p 就是选它的比例
        return idx

    def local_search(self, solution, fitness):
        for i in range(len(solution)):
            solution_b = solution[:]
            solution_b[i] = 1-solution_b[i]
            fit = self.method(solution_b)
            if self.maximum:
                if fit > fitness:
                    fitness = fit
                    solution = solution_b[:]
            else:
                if fit < fitness:
                    fitness = fit
                    solution = solution_b[:]
            del solution_b
        return solution, fitness


    def run(self):
        """
        Run the genetic algorithm.
        Ret:
            global_best_ind: The best indiviudal during the evolutionary process.
            global_best_fitness: The fitness of the global_best_ind.
        """
        global_best = 0
        self.initialization()
        if self.maximum:
            best_index = np.argsort(self.fitness)[-1]
        else:
            best_index = np.argsort(self.fitness)[0]
        global_best_fitness = self.fitness[best_index]
        global_best_ind = self.pop[best_index, :]
        eva_times = self.pop_shape[0]
        count = 0

        for it in range(self.max_round):
            next_gene = []
            self.pop = np.load('pop.npy')
            for n in range(int(self.pop_shape[0]/2)):
                i, j = self.rws(2, self.fitness) # choosing 2 individuals with rws.
                indi_0, indi_1 = self.pop[i, :].copy(), self.pop[j, :].copy()
                if np.random.rand() < self.p_c:
                    indi_0, indi_1 = self.crossover(indi_0, indi_1)

                if np.random.rand() < self.p_m:
                    indi_0 = self.mutation(indi_0)
                    indi_1 = self.mutation(indi_1)

                next_gene.append(indi_0)
                next_gene.append(indi_1)


            self.pop = np.array(next_gene)
            self.fitness = self.evaluation(self.pop)
            eva_times += self.pop_shape[0]

            if self.maximum:
                if np.max(self.fitness) > global_best_fitness:
                    best_index = np.argsort(self.fitness)[-1]
                    global_best_fitness = self.fitness[best_index]
                    global_best_ind = self.pop[best_index, :]
                    # global_best_ind, global_best_fitness = self.local_search(global_best_ind, global_best_fitness)
                    with open('./history.txt', 'w') as f:
                        f.write(str(global_best_ind.tolist()))
                        f.write(str(global_best_fitness))
                    count = 0
                else:
                    count +=1
                worst_index = np.argsort(self.fitness)[-1]
                self.pop[worst_index, :] = global_best_ind
                self.fitness[worst_index] = global_best_fitness

            else:
                if np.min(self.fitness) < global_best_fitness:
                    best_index = np.argsort(self.fitness)[0]
                    global_best_fitness = self.fitness[best_index]
                    global_best_ind = self.pop[best_index, :]
                    # global_best_ind, global_best_fitness = self.local_search(global_best_ind, global_best_fitness)
                    with open('./history.txt', 'w') as f:
                        f.write(str(global_best_ind.tolist()))
                        f.write(str(global_best_fitness))
                    count = 0
                else:
                    count +=1

                worst_index = np.argsort(self.fitness)[-1]
                self.pop[worst_index, :] = global_best_ind
                self.fitness[worst_index] = global_best_fitness

            np.save('pop.npy', self.pop)

            if self.verbose != None and 0 == (it % self.verbose):
                print('Gene {}:'.format(it))
                print('Global best fitness:', global_best_fitness)

            if self.early_stop_rounds != None and count > self.early_stop_rounds:
                print('Did not improved within {} rounds. Break.'.format(self.early_stop_rounds))
                break

        print('\n Solution: {} \n Fitness: {} \n Evaluation times: {}'.format(global_best_ind, global_best_fitness, eva_times))
        return global_best_ind, global_best_fitness
f_cols = feature_1.drop('user_id', axis=1).columns

def evaluate(solution):
	cols = f_cols[solution.astype(bool)]
	lgb_train = lgb.Dataset(feature_1[cols], target_1)
	lgb_eval = lgb.Dataset(feature_2[cols], target_2, reference=lgb_train)

	print('开始训练......')
	params = {
	    'task': 'train',
	    'boosting_type': 'gbdt',
	    'objective': 'binary',
	    'metric': {'auc'}
	}
	gbm = lgb.train(params,
				lgb_train,
				num_boost_round=40000,
				early_stopping_rounds=50,
				valid_sets=lgb_eval,
				verbose_eval=False
                )
	gc.collect()
	return gbm.best_score['valid_0']['auc']

## searching
ga = BGA(pop_shape=(30, f_cols.shape[0]), method=evaluate, max_round=30, verbose=1, p_m=0.3, maximum=True)
solution, fitness = ga.run()
