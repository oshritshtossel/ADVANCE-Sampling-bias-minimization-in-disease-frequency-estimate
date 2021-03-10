import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from scipy import optimize
from calculate_tools import positive_given_features, append_non_duplicates
from itertools import product
import statistics

from preprocess import mul_series


def calculate_p0t_qi_for_n_first_days(N, K, X, n, dppt_dict,
                                      alpha=1, beta=1, sigma=0.01) -> tuple:
    """

    :param N: nit
    :param K: kit
    :param X: list of all days
    :param n: number of days to calculate p0t for
    :param dppt_dict:
    """

    Y = sorted(set([x[1] for x in N[X[:n]].index]))
    qi = positive_given_features(Y, N[X[:n]], K[X[:n]])
    initial_guess = []
    for t in X[:n]:
        sum_ni = N[t].sum()
        sum0 = sum((qi * N[t]).dropna() / sum_ni)
        initial_guess.append(dppt_dict[t] / sum0)

    mle = MLE(N[X[:n]], K[X[:n]], alpha=alpha, beta=beta, sigma=sigma)
    roots = optimize.root(mle.MLE_fast_func, initial_guess, method="lm")
    qi = mle.calculate_qi_based_on_MLE(roots.x)

    return (roots.x, qi)


def calculate_mle_day_by_day(N, K, X, dppt_dict, num_of_prev_days_to_calculate, num_of_perv_days_for_mean=7, alpha=1,
                             beta=1, sigma=0.01, p0t_qi=False):
    """

    :param N: nit
    :param K: kit
    :param X: all days
    :param dppt_dict:
    :param num_of_prev_days_to_calculate:
    :return:
    """
    all_qi = pd.DataFrame()

    print("calculating the first set")
    roots, qi = calculate_p0t_qi_for_n_first_days(N,
                                                  K, X, num_of_prev_days_to_calculate,
                                                  dppt_dict, alpha=alpha, beta=beta,
                                                  sigma=sigma)
    print("finished calculating the first set")
    all_roots = np.copy(roots)

    for day in X[:len(roots)]:
        all_qi[day] = qi

    for day in X[num_of_prev_days_to_calculate:]:
        print("calculating the day: " + day)
        new_root, qi = calculte_single_p0t_and_qi_of_t_plus_1(roots, X, day, num_of_prev_days_to_calculate, N, K,
                                                              dppt_dict,
                                                              num_of_perv_days_for_mean=num_of_perv_days_for_mean,
                                                              alpha=alpha, beta=beta, sigma=sigma, p0t_qi=p0t_qi)

        qi.name = day
        all_roots = np.hstack([all_roots, new_root])
        all_qi = all_qi.join(qi, how="outer")

        roots = np.hstack([roots, new_root])
        roots = roots[1:]

    return all_roots, all_qi


def calculte_single_p0t_and_qi_of_t_plus_1(roots: list, X: list, day, num_of_prev_days, N, K, dppt_dict,
                                           num_of_perv_days_for_mean=7, alpha=1,
                                           beta=1, sigma=0.01, p0t_qi=False):
    index_of_day = X.index(day)

    Y = sorted(set([x[1] for x in N[X[index_of_day - num_of_prev_days + 1]: X[index_of_day]].index]))
    qi = positive_given_features(Y, N[X[index_of_day - num_of_prev_days + 1]: X[index_of_day]],
                                 K[X[index_of_day - num_of_prev_days + 1]: X[index_of_day]])

    initial_guess = []
    for t in X[index_of_day - num_of_prev_days + 1: index_of_day + 1]:
        sum_ni = N[t].sum()
        sum0 = sum((qi * N[t]).dropna() / sum_ni)
        initial_guess.append(dppt_dict[t] / sum0)

    mle = MLE(N[X[index_of_day - num_of_prev_days + 1]: X[index_of_day]],
              K[X[index_of_day - num_of_prev_days + 1]: X[index_of_day]], alpha=alpha, beta=beta, sigma=sigma)
    mle_c_s = MLE_constants_solver(mle, roots[1:], num_of_perv_days_for_mean,
                                   num_of_prev_dayes=num_of_perv_days_for_mean)

    if p0t_qi:
        p0t_root = optimize.root(mle_c_s.solve_mle_with_constants_p0t_qit, initial_guess[len(initial_guess) - 1],
                                 method="lm")
    else:
        p0t_root = optimize.root(mle_c_s.solve_mle_with_constants, initial_guess[len(initial_guess) - 1], method="lm")

    qi = mle.calculate_qi_based_on_MLE(np.hstack([roots[1:], p0t_root.x]))

    return (p0t_root.x, qi)


class MLE(object):
    """
    Assistant class for scipy optimize for accessing Nit Kit for every func in the class
    All of the functions are based on this calculations: https://www.overleaf.com/read/tbvstyqzzjst
    """

    def __init__(self, N: Series, K: Series, alpha=0, beta=0, sigma=1):
        self.N = N
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

        self.K_time = sorted(list(set(self.K.index.get_level_values(0))))
        self.K_feature = sorted(list(set(self.K.index.get_level_values(1))))
        self.N_time = sorted(list(set(self.N.index.get_level_values(0))))
        self.N_feature = sorted(list(set(self.N.index.get_level_values(1))))

        to_add_N = Series(index=product(self.N_time, self.N_feature), data=0)
        to_add_K = Series(index=product(self.N_time, self.N_feature), data=0)

        self.N = append_non_duplicates(self.N, to_add_N)
        self.K = append_non_duplicates(self.K, to_add_K)

        self.teta = Series()
        swaped_K = self.K.swaplevel(0, 1)
        for i in self.K_feature:
            self.teta[i] = sum(swaped_K[i] + self.alpha)

        self.fixed_n_with_sigma = self.N.unstack()
        self.fixed_n_without_sigma = self.N.unstack()
        for i in self.N_feature:
            if i not in self.teta:
                self.teta[i] = self.alpha * len(self.K_time)

            self.fixed_n_without_sigma[i] = self.fixed_n_without_sigma[i].apply(
                lambda x: (x + self.alpha + self.beta) * self.teta[i])
            self.fixed_n_with_sigma[i] = self.fixed_n_with_sigma[i].apply(
                lambda x: self.sigma ** 2 * (x + self.alpha + self.beta) * self.teta[i])

    def Mi(self, dict_of_p0):
        """
        The function calculates the expression of Mi according to p0 values
        :param dict_of_p0: dict:key= date=t, val= p0(t)
        :return: a series :column= vector of features=i, index= M for this vector of features i
        """
        M = Series()
        # apply the function M on all N for faster calculation
        N_for_M = self.N.unstack(level=0)
        for t in self.N_time:
            N_for_M[t] = N_for_M[t].apply(lambda x: dict_of_p0[t] * (x + self.alpha + self.beta))
        for i in self.N_feature:
            M[i] = sum(N_for_M.swapaxes(0, 1)[i])
        return M

    def MLE_fast_func(self, p0t: list):
        """
        Can only work if beta is not 0

        substritutes p0
        :param p0t: a dict= variables for substitute
        :return: list of the results after substituting of p0
        """
        if self.beta == 0:
            raise KeyError

        #####
        # p0t = [min(i, 0.999) for i in p0t]
        #####

        series_p0 = Series(p0t, self.N_time)

        M = self.Mi(series_p0)

        fixed_k = self.K.swaplevel(0, 1)
        # fixed_n = self.N.swaplevel(0, 1)

        part_a = DataFrame(index=self.N_time, columns=self.N_feature, dtype=float)
        part_c = DataFrame(index=self.N_time, columns=self.N_feature, dtype=float)
        upper_part_for_first_argument = DataFrame(index=self.N_time, columns=self.N_feature)
        for i in self.N_feature:
            upper_part_for_first_argument[i] = fixed_k[i].apply(lambda x: (x + self.alpha) * M[i]) - \
                                               self.fixed_n_without_sigma[i]
            part_a[i] = fixed_k[i].apply(lambda x: self.sigma ** 2 * (x + self.alpha) * M[i]) - self.fixed_n_with_sigma[
                i]
            part_c[i] = series_p0.apply(lambda x: M[i] - x * self.teta[i])
        # part_a = part_a.swapaxes(0, 1)
        part_c = part_c.swapaxes(0, 1)
        upper_part_for_first_argument = upper_part_for_first_argument.swapaxes(0, 1)

        return_list = np.array([
            sum(upper_part_for_first_argument[self.N_time[0]] / (series_p0[self.N_time[0]] * part_c[self.N_time[0]]))])
        for t in self.N_time[1:]:
            part_b = series_p0[t] * (series_p0[t] - series_p0[self.N_time[self.N_time.index(t) - 1]])
            # Memory error with alot of days clean memory for avoiding memory error
            # gc.collect()
            return_list = np.append(return_list, sum((part_a.swapaxes(0, 1)[t] - (part_b * part_c[t])) / part_c[t]))

        return return_list

    def calculate_qi_based_on_MLE(self, p0t):
        """

        :param p0t:
        :return:
        """
        series_p0 = Series(p0t, self.N_time)

        fixed_k = self.K.swaplevel(0, 1)
        fixed_n = self.N.swaplevel(0, 1)

        nominator = Series(index=self.N_feature)
        denominator = Series(index=self.N_feature)
        pre_denominator = fixed_n.apply(lambda x: x + self.alpha + self.beta)
        pre_denominator = pre_denominator.swaplevel(0, 1)
        pre_denominator = pre_denominator.astype(float)
        almost_denominator = DataFrame(columns=self.N_time, index=self.N_feature, dtype=float)

        for t in self.N_time:
            almost_denominator[t] = pre_denominator[t].apply(lambda x: x * series_p0[t])

        pre_denominator = almost_denominator.swapaxes(0, 1)

        for i in self.N_feature:
            nominator[i] = sum(fixed_k[i].apply(lambda x: x + self.alpha))
            denominator[i] = sum(pre_denominator[i])

        return nominator / denominator

    def MLE_fast_func_with_p0t_close_to_mean_of_prev_week(self, p0t: list, num_of_prev_dayes=7):
        """
        Can only work if beta is not 0

        substritutes p0
        :param p0t: a dict= variables for substitute
        :return: list of the results after substituting of p0
        """
        if self.beta == 0:
            raise KeyError

        #####
        #   p0t = [min(i, 0.999) for i in p0t]
        #####

        series_p0 = Series(p0t, self.N_time)

        M = self.Mi(series_p0)

        fixed_k = self.K.swaplevel(0, 1)
        # fixed_n = self.N.swaplevel(0, 1)

        part_a = DataFrame(index=self.N_time, columns=self.N_feature, dtype=float)
        part_c = DataFrame(index=self.N_time, columns=self.N_feature, dtype=float)
        upper_part_for_first_argument = DataFrame(index=self.N_time, columns=self.N_feature)
        for i in self.N_feature:
            upper_part_for_first_argument[i] = fixed_k[i].apply(lambda x: (x + self.alpha) * M[i]) - \
                                               self.fixed_n_without_sigma[i]
            part_a[i] = fixed_k[i].apply(lambda x: self.sigma ** 2 * (x + self.alpha) * M[i]) - self.fixed_n_with_sigma[
                i]
            part_c[i] = series_p0.apply(lambda x: M[i] - x * self.teta[i])
        # part_a = part_a.swapaxes(0, 1)
        part_c = part_c.swapaxes(0, 1)
        upper_part_for_first_argument = upper_part_for_first_argument.swapaxes(0, 1)

        return_list = np.array([
            sum(upper_part_for_first_argument[self.N_time[0]] / (series_p0[self.N_time[0]] * part_c[self.N_time[0]]))])
        for t in self.N_time[1:num_of_prev_dayes]:
            part_b = series_p0[t] * (series_p0[t] - series_p0[self.N_time[self.N_time.index(t) - 1]])
            # Memory error with alot of days clean memory for avoiding memory error
            # gc.collect()
            return_list = np.append(return_list, sum((part_a.swapaxes(0, 1)[t] - (part_b * part_c[t])) / part_c[t]))
        for t in self.N_time[num_of_prev_dayes:]:
            part_b = series_p0[t] * (series_p0[t] - statistics.mean(
                series_p0[self.N_time[self.N_time.index(t) - num_of_prev_dayes]:self.N_time[self.N_time.index(t)]]))
            # Memory error with alot of days clean memory for avoiding memory error
            # gc.collect()
            return_list = np.append(return_list, sum((part_a.swapaxes(0, 1)[t] - (part_b * part_c[t])) / part_c[t]))

        return return_list

    def MLE_fast_func_probability(self, p0t: list, with_average=0):
        """
        Can only work if beta is not 0

        substritutes p0
        :param p0t: a dict= variables for substitute
        :return: list of the results after substituting of p0
        """
        if self.beta == 0:
            raise KeyError

        series_p0 = Series(p0t, self.N_time)

        M = self.Mi(series_p0)
        teta_divided_M = self.teta / M

        # first_var:
        part_1_nominator_first = (self.K[self.N_time[0]] + self.alpha) * np.exp(
            -series_p0[self.N_time[0]] * teta_divided_M) * teta_divided_M
        part_1_denominator_first = 1 - np.exp(-series_p0[self.N_time[0]] * teta_divided_M)
        part_1_first = sum(part_1_nominator_first / part_1_denominator_first)
        part_2_first = (self.K[self.N_time[0]] - self.N[self.N_time[0]]) * teta_divided_M
        return_list = np.array(part_1_first + sum(part_2_first))

        K_minus_N = self.K - self.N
        exponent = np.exp(-mul_series(series_p0, teta_divided_M).T)
        part_1_denominator = 1 - exponent
        for t in self.N_time[1:]:
            part_1_nominator = (self.K[t] + self.alpha) * exponent[t] * teta_divided_M
            part_1 = sum(part_1_nominator / part_1_denominator[t])
            part_2 = K_minus_N[t] * teta_divided_M
            if with_average == 0 or self.N_time.index(t) < with_average:
                part_3 = series_p0[t] - series_p0[self.N_time[self.N_time.index(t) - 1]]
            else:
                part_3 = (series_p0[t] - statistics.mean(
                    series_p0[self.N_time[self.N_time.index(t) - with_average]:self.N_time[self.N_time.index(t)]]))
            return_list = np.hstack([return_list, part_1 + sum(part_2) + part_3])

        return return_list



class MLE_constants_solver(object):
    def __init__(self, mle: MLE, constants: list, average_of, num_of_prev_dayes=7):
        self.mle = mle
        self.constants = constants
        self.average_of = average_of
        self.num_of_prev_dayes = num_of_prev_dayes

    def solve_mle_with_constants(self, p0t: list):
        result = self.mle.MLE_fast_func_probability(np.hstack([self.constants, p0t]), with_average=self.average_of)
        return result[len(result) - 1]

    def solve_mle_with_constants_p0t_qit(self, p0t):
        result = self.mle.MLE_fast_func_with_p0t_close_to_mean_of_prev_week(np.hstack([self.constants, p0t]),
                                                                            num_of_prev_dayes=self.num_of_prev_dayes)
        return result[len(result) - 1]
