import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import lru_cache
from scipy.interpolate import interp2d


class TestOptimisation:
    def __init__(self, population=(1000, 10000, 10000),
                 pre_test_probability=(.3, .03, .003),
                 onward_transmission=(2, 3, 1, .3),
                 routine_capacity=5000,
                 priority_capacity_proportion=.1,
                 priority_queue=True,
                 routine_tat=1,
                 tat_at_fifty_percent_surge=2,
                 swab_delay=1,
                 symptomatic_testing_proportion=1.,
                 test_prioritsation_by_indication=None,
                 tat_function='quadratic',
                 stochastic_tat_dist='deterministic',
                 gamma_scale_for_tat_dist=1,
                 stochastic_tat_reps=100
                 ):
        self.population = population
        self.close_contact, self.symptomatic, self.asymptomatic = population
        self.onward_transmission = onward_transmission
        self.pre_test_by_indication = pre_test_probability

        self.routine_capacity = routine_capacity
        self.priority_capacity = routine_capacity * priority_capacity_proportion
        self.priority_queue = priority_queue

        self.routine_tat = routine_tat
        self.tat_surge = tat_at_fifty_percent_surge

        self.swab_delay = swab_delay

        self.symptomatic_test_proportion = symptomatic_testing_proportion

        self.priority_order_indication = test_prioritsation_by_indication

        self.tat_function = tat_function

        self.stochastic_tat_dist = stochastic_tat_dist
        self.gamma_scale_for_tat_dist = gamma_scale_for_tat_dist
        self.stochastic_tat_reps = stochastic_tat_reps

        if not 0 <= priority_capacity_proportion < 1:
            raise ValueError(f'Priority capacity proportion must be between '
                             f'0 and 1. The input value was '
                             f'{priority_capacity_proportion}.')

    def set_population_by_input_number_infections(self, num_infections):
        cc, s, a = self.population
        close_contacts = num_infections * sum(self.onward_transmission)
        net_close = cc - close_contacts
        net_sympt = s * net_close / (s + a)
        net_asympt = a * net_close / (s + a)

        symptomatic = s + net_sympt
        asymptomatic = a + net_asympt

        cc_prob, symp_prob, asymp_prob = self.pre_test_by_indication
        exp_close_contact_cases = close_contacts * cc_prob

        remaining_cases = num_infections - exp_close_contact_cases
        exp_remaining_cases = symptomatic * symp_prob + asymptomatic * asymp_prob

        scaling = remaining_cases / exp_remaining_cases
        symp_prob, asymp_prob = np.array([symp_prob, asymp_prob]) * scaling

        self.population = (close_contacts, symptomatic, asymptomatic)
        # self.pre_test_by_indication = (cc_prob, symp_prob, asymp_prob)

    def turn_around_time(self, tests, priority_queue=False):
        if self.tat_function == 'quadratic':
            return self.function_turn_around_time(priority_queue)(tests)
        if self.tat_function == 'linear':
            return self.function_turn_around_time_linear(priority_queue)(tests)
        if self.tat_function == 'exponential':
            return self.function_turn_around_time_exp(priority_queue)(tests)

    def plot_turn_around_time(self, title=None, color=None):
        tests = np.arange(self.routine_capacity * 2)
        tat = []
        for test in tests:
            tat.append(self.turn_around_time(test, False))
        if color is None:
            plt.plot(tests, tat)
        else:
            plt.plot(tests, tat, color=color)
        plt.xlabel('Number of tests')
        plt.ylabel('Average turn around time')
        plt.title(title)

    def plot_delay_effect_on_transmission(self, max_delay=5):
        delay_array = np.linspace(0, max_delay, 1000)
        transmission_reduction = []
        for delay in delay_array:
            red = 100 * (1 -
                         self.test_delay_effect_on_percent_future_infections(
                             delay,
                             swab_delay=self.swab_delay))
            transmission_reduction.append(red)
        plt.plot(delay_array, transmission_reduction)
        plt.xlabel('Result delay')
        plt.ylabel('Onward transmission %')

    @lru_cache()
    def function_turn_around_time(self, priority_queue=False):
        routine_capacity = self.routine_capacity
        tat = self.routine_tat
        if priority_queue:
            tat_surge = tat
        else:
            tat_surge = self.tat_surge
        return lambda x: tat if x < routine_capacity else \
            tat + (tat_surge - tat) * ((x - routine_capacity) ** 2) / \
            ((routine_capacity * .5) ** 2)

    @lru_cache()
    def function_turn_around_time_linear(self, priority_queue=False):
        routine_capacity = self.routine_capacity
        tat = self.routine_tat
        if priority_queue:
            tat_surge = tat
        else:
            tat_surge = self.tat_surge
        return lambda x: tat if x < routine_capacity else \
            tat + (tat_surge - tat) * 2 * (x - routine_capacity)/routine_capacity

    @lru_cache()
    def function_turn_around_time_exp(self, priority_queue=False):
        routine_capacity = self.routine_capacity
        tat = self.routine_tat
        if priority_queue:
            tat_surge = tat
        else:
            tat_surge = self.tat_surge
        return lambda x: tat if x < routine_capacity else \
            tat + (tat_surge - tat) * (np.exp((x - routine_capacity)/routine_capacity) - 1) / (np.exp(.5) - 1)

    @lru_cache()
    def load_test_delay_data(self):
        data = pd.read_csv('testing_delay_kretzhcmar_table_2_extended.csv')
        z = np.array(data)[:,1:]
        x = np.arange(0, z.shape[1])
        y = np.arange(0, z.shape[0])
        return interp2d(x, y, z)

    def test_delay_effect_on_percent_future_infections(self, result_delay=2., swab_delay=1.):
        return self.load_test_delay_data()(result_delay, swab_delay)

    @lru_cache()
    def create_pre_test_proabability_array(self):
        array = np.zeros([4, 3])
        for i in range(3):
            array[:, i] = self.pre_test_by_indication[i]
        return array

    @lru_cache()
    def create_onward_transmission_array(self):
        onward_transmission_dimension = self.get_dimension_of_input_onward_transmission_array()
        if onward_transmission_dimension == 0:
            self.onward_transmission = [self.onward_transmission] * 4
        if onward_transmission_dimension <= 1:
            array = np.zeros([4, 3])
            for i in range(3):
                array[:, i] = self.onward_transmission
            return array
        else:
            return np.array(self.onward_transmission)

    @lru_cache()
    def get_dimension_of_input_onward_transmission_array(self):
        len_outer_list = len(self.onward_transmission)
        if len_outer_list == 0:
            raise ValueError(f"Onward transmission cannot be empty. "
                             f"The input was '{self.onward_transmission}'")
        if len_outer_list == 1:
            return 0
        try:
            len_inner_list = len(self.onward_transmission[0])
        except TypeError:
            return 1
        if len_inner_list == 3 and len_outer_list == 4:
            return 2
        else:
            raise ValueError(f"Onwards transsmission must be a number, a length 4 array or a 4x3 array"
                             f"The input value was '{self.onward_transmission}'")

    # @lru_cache()
    def create_population_groups(self):
        array = np.zeros([4, 3])
        dim_onward_transmission = self.get_dimension_of_input_onward_transmission_array()
        if dim_onward_transmission <= 1:
            relative_transmission = np.array([i / sum(self.onward_transmission) for i in self.onward_transmission])
            for i in range(3):
                array[:, i] = self.population[i] * relative_transmission
        else:
            onward_array = np.array(self.onward_transmission)
            col_sums = np.sum(onward_array, axis=0)
            col_sums_array = np.array([list(col_sums)] * 4)
            relative_transmission = onward_array / col_sums_array
            population_array = np.array([list(self.population)] * 4)
            array = relative_transmission * population_array
        return array

    def create_expected_onward_transmission_array(self):
        onward = self.create_onward_transmission_array()
        prob = self.create_pre_test_proabability_array()
        return onward * prob

    def create_transmission_tested_array(self, result_delay=2, swab_delay=1):
        onward = self.create_onward_transmission_array()
        transmission_reduction = self.test_delay_effect_on_percent_future_infections(
            result_delay=result_delay, swab_delay=swab_delay)
        return onward * (1 - transmission_reduction)

    def create_expected_transmission_tested_array(self, result_delay=2, swab_delay=1):
        onward = self.create_transmission_tested_array(
            result_delay=result_delay, swab_delay=swab_delay)

        prob = self.create_pre_test_proabability_array()
        return onward * prob

    def benefit_of_test(self, result_delay):
        # expected_trans = self.create_expected_onward_transmission_array()
        expected_trans = self.create_expected_transmission_tested_array(
            result_delay=np.inf, swab_delay=7)
        expected_tested_trans = self.create_expected_transmission_tested_array(
            result_delay=result_delay, swab_delay=self.swab_delay) * .9999
        return expected_trans - expected_tested_trans
        # expected_onward_infection = self.

    def plot_benefit_as_function_delay(self):
        result_delay = np.linspace(0, 5, 1000)
        benefit_array = []
        for res_del in result_delay:
            benefit_array.append(self.benefit_of_test(
                result_delay=res_del))
        benefit_array = np.array(benefit_array)
        for i in range(3):
            plt.plot(result_delay, benefit_array[:, :, i])
        plt.show()

    def allocate_tests(self, num_tests=1000,
                       result_delay=1):
        benefit_array = self.benefit_of_test(result_delay=result_delay)
        if self.priority_order_indication is not None:
            priority_order = np.array(self.priority_order_indication)
            last_priority_column = np.where(priority_order == 3)
            largest_last_priority_value = np.max(benefit_array[:, last_priority_column])

            second_priority_column = np.where(priority_order == 2)
            min_second_priority_value = np.min(benefit_array[:, second_priority_column])
            if min_second_priority_value <= largest_last_priority_value:
                diff = largest_last_priority_value - min_second_priority_value
                benefit_array[:, second_priority_column] = benefit_array[:, second_priority_column] + diff + .1

            largest_second_priority_value = np.max(benefit_array[:, second_priority_column])

            first_priority_column = np.where(priority_order == 1)
            min_first_priority_value = np.min(benefit_array[:, first_priority_column])
            if min_first_priority_value <= largest_second_priority_value:
                diff = largest_second_priority_value - min_first_priority_value
                benefit_array[:, first_priority_column] = benefit_array[:, first_priority_column] + diff + .1

        tests_remaining = num_tests
        num_tests_by_group = np.zeros(np.shape(benefit_array))
        pop_per_group = self.create_population_groups()
        pop_per_group[:, 1] = pop_per_group[:, 1] * self.symptomatic_test_proportion
        pop_per_group[:, 0] = pop_per_group[:, 0] * self.symptomatic_test_proportion
        while tests_remaining > 0:
            max_benefit = np.max(benefit_array)
            max_benefit_location = benefit_array == max_benefit
            total_pop_in_best_groups = np.sum(pop_per_group[max_benefit_location])
            if total_pop_in_best_groups < tests_remaining:
                num_tests_by_group[max_benefit_location] = pop_per_group[max_benefit_location]
            else:
                num_tests_by_group[max_benefit_location] = pop_per_group[max_benefit_location] * \
                                                           tests_remaining / total_pop_in_best_groups
            tests_remaining -= total_pop_in_best_groups
            benefit_array[max_benefit_location] = -1
        return num_tests_by_group

    def estimate_total_tranmission(self, test_allocation,
                                   result_delay=1):
        pop_per_group = self.create_population_groups()
        pop_untested = pop_per_group - test_allocation

        priority_tat = self.turn_around_time(tests=1)
        if self.priority_queue:
            total_tests = np.sum(test_allocation)
            num_priority_tests = min([total_tests, self.priority_capacity])
            test_allocation_priority = self.allocate_tests(num_tests=int(num_priority_tests),
                                                           result_delay=priority_tat)
        else:
            test_allocation_priority = np.zeros(test_allocation.shape)

        test_allocation -= test_allocation_priority
        test_allocation[test_allocation < 0] = 0

        exp_transmission_priority_test = self.create_expected_transmission_tested_array(
            result_delay=priority_tat, swab_delay=self.swab_delay)

        exp_transmission_test = self.create_expected_transmission_tested_array(
            result_delay=result_delay, swab_delay=self.swab_delay)
        exp_transmission_notest = self.create_expected_transmission_tested_array(
            result_delay=np.inf, swab_delay=7)

        untested_transmission = np.sum(pop_untested * exp_transmission_notest)
        tested_transmission = np.sum(test_allocation * exp_transmission_test)
        priority_tested_transmission = np.sum(test_allocation_priority * exp_transmission_priority_test)
        return tested_transmission + untested_transmission + priority_tested_transmission

    @lru_cache()
    def estimate_transmission_with_testing(self, num_test, distribution='deterministic',
                                           gamma_scale=1, reps=100):
        if distribution == 'deterministic':
            tat = self.turn_around_time(num_test)  # todo: remove priority queue from turn_around_time?
            test_allocation = self.allocate_tests(num_tests=num_test,
                                                  result_delay=tat)
            percent_positive = sum(np.array(self.pre_test_by_indication) * np.sum(test_allocation, 0)) / num_test
            total_transmission = self.estimate_total_tranmission(test_allocation,
                                                                 result_delay=tat)
        elif distribution == 'gamma':
            average_tat = self.turn_around_time(num_test)
            gamma_shape = average_tat/gamma_scale
            tat_list = np.random.gamma(shape=gamma_shape,
                            scale=gamma_scale, size=reps) #note variance = shape*scale^2
            perc_pos_list = list()
            tot_trans_list = list()
            for tat in tat_list:
                test_allocation = self.allocate_tests(num_tests=num_test,
                                                      result_delay=tat)
                perc_pos_list.append(sum(np.array(self.pre_test_by_indication) * np.sum(test_allocation, 0)) / num_test)
                tot_trans_list.append(self.estimate_total_tranmission(test_allocation,
                                                                 result_delay=tat))
            total_transmission = np.mean(tot_trans_list)
            percent_positive = np.mean(perc_pos_list)
        else:
            raise ValueError(f'Distribution {distribution} not recognised')

        return total_transmission, percent_positive

    @lru_cache()
    def generate_onward_transmission_with_tests(self, max_tests_proportion=3.):
        num_test_array = range(1, int(self.routine_capacity * max_tests_proportion))
        transmission = []
        positivity = []
        for num_tests in num_test_array:
            current_onward_transmission, current_positive_percentage = \
                self.estimate_transmission_with_testing(num_test=num_tests,
                                                        distribution=self.stochastic_tat_dist,
                                                        gamma_scale=self.gamma_scale_for_tat_dist,
                                                        reps=self.stochastic_tat_reps)
            transmission.append(current_onward_transmission)
            positivity.append(current_positive_percentage)
        transmission = np.array(transmission)  # /\
        # np.sum(np.array(self.population) * np.array(self.pre_test_by_indication))
        positivity = np.array(positivity)
        return num_test_array, transmission, positivity

    def plot_transmission_with_testing(self, title=None, max_prop_tests=2.):
        test_array, transmission_array, positive_array = \
            self.generate_onward_transmission_with_tests(max_tests_proportion=max_prop_tests)
        plt.plot(test_array, transmission_array)
        plt.xlabel('Number of tests')
        plt.ylabel('Onward transmission')
        if title is None:
            if self.priority_queue:
                plt.title(f'Test capacity = {self.routine_capacity}'
                          f', with priority testing')
                plt.savefig(f'Test_capacity_{self.routine_capacity}'
                            f'_priority_testing.png')
            else:
                plt.title(f'Test capacity = {self.routine_capacity}')
                plt.savefig(f'Test_capacity_{self.routine_capacity}.png')
        else:
            plt.title(title)
            plt.savefig(f'{title.replace(" ", "_")}.png')
        plt.show()

    def optimal_test_amount_array(self):
        num_test_array, transmission, positivity = self.generate_onward_transmission_with_tests()
        opt_test = num_test_array[np.where(transmission == min(transmission))[0][
            0]]  # todo not sure what would happen if there were two values at the min
        # if len(opt_test) > 1:
        #     opt_test = opt_test[0]
        num_tests_by_group = self.allocate_tests(num_tests=opt_test).astype(int)
        tests_by_indication = [int(i) for i in np.sum(num_tests_by_group, axis=0)]
        return opt_test, tests_by_indication, num_tests_by_group

    def optimal_test_amount(self):
        if self.generate_onward_transmission_with_tests.cache_info().currsize > 0:
            return self.optimal_test_amount_array()
        else:
            min_optimal_tests = self.routine_capacity
            c_tests = min_optimal_tests
            c_transmission = np.inf
            c_positivity = None
            opt_test = None
            opt_found_flag = False
            while opt_found_flag is False:
                new_onward_transmission, new_percent_positive = \
                    self.estimate_transmission_with_testing(num_test=c_tests,
                                                        distribution=self.stochastic_tat_dist,
                                                        gamma_scale=self.gamma_scale_for_tat_dist,
                                                        reps=self.stochastic_tat_reps)
                if new_onward_transmission > c_transmission:
                    opt_test = c_tests
                    opt_found_flag = True
                else:
                    c_transmission = new_onward_transmission
                    c_positivity = new_percent_positive
                c_tests += 1

            num_tests_by_group = self.allocate_tests(num_tests=opt_test).astype(int)
            tests_by_indication = [int(i) for i in np.sum(num_tests_by_group, axis=0)]
            return opt_test, tests_by_indication, num_tests_by_group

    def optimal_test_uncertain(self, uncertainty_range_prop):
        num_tests, exp_transmission = self.create_uncertain_onward_array(uncertainty_range_prop)
        min_index = np.where(np.min(exp_transmission) == exp_transmission)[0][0]
        opt_tests = num_tests[min_index]
        return opt_tests

    def plot_uncertaint_tests(self, uncertainty_range_prop):
        num_tests, exp_transmission = self.create_uncertain_onward_array(uncertainty_range_prop)

        num_test_array, transmission, positivity = self.generate_onward_transmission_with_tests()
        num_test_array = np.array(num_test_array)

        # scale to be a percentage
        exp_transmission = exp_transmission / max(exp_transmission)
        transmission = transmission/max(transmission)

        # make the non-uncertain array the same size.
        elements_to_keep = num_test_array <= max(num_tests)
        num_test_array = num_test_array[elements_to_keep]
        transmission = transmission[elements_to_keep]

        plt.plot(num_tests, exp_transmission)
        plt.plot(num_test_array, transmission)


    @lru_cache()
    def create_uncertain_onward_array(self, uncertainty_range_prop):
        num_test_array, transmission, positivity = self.generate_onward_transmission_with_tests()
        num_test_array = np.array(num_test_array)
        num_test_uncertainty_array = []
        expected_transmission = []
        for n_tests in num_test_array:
            min_tests = int(n_tests*(1 - uncertainty_range_prop))
            max_tests = int(n_tests*(1 + uncertainty_range_prop))
            if (min_tests > 0) and max_tests < max(num_test_array):
                test_range = (min_tests <= num_test_array) & (max_tests >= num_test_array)
                num_test_uncertainty_array.append(n_tests)
                expected_transmission.append(np.mean(transmission[test_range]))
        return tuple(num_test_uncertainty_array), tuple(expected_transmission)


    def make_plot_transmission_perc_post(self, max_test_proportion=3):
        expected_cases = sum(
            [pop * prob for pop, prob in zip(self.population, self.pre_test_by_indication)])
        expected_cases = np.round(expected_cases, 1)
        # print(f"Expected cases = "
        #       f"{expected_cases}"
        #       )
        # optim_object.plot_transmission_with_testing('Community transmission 2', max_prop_tests=3)
        test_array, onward_transmission, positivity = self.generate_onward_transmission_with_tests(
            max_tests_proportion=max_test_proportion)
        onward_transmission = 100 * onward_transmission / max(
            onward_transmission)  # max onward transmission shouldn not depend on symp perc.
        test_array = np.array(test_array) / 100

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Tests per 1000 people')
        ax1.set_ylabel('Percentage of onwards transmission')
        ax1.plot(test_array, onward_transmission)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Percent positive', color='m')
        ax2.plot(test_array, positivity * 100, ':m')
        # plt.ylabel('Percentage reduction in transmission')
        # plt.xlabel('Tests per 1000 people')
        # plt.title(f'Daily incidence = {expected_cases} per 100,000')
        plt.tight_layout()
        return ax1, onward_transmission, positivity, expected_cases, test_array
