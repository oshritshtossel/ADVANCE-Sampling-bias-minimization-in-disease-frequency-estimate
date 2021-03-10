from datetime import datetime

import pandas as pd
from scipy import optimize

from calculate_tools import calculate_prob_of_features, calculate_prob_of_features_given_day, positive_given_features, \
    dppt
from estimate_models import calculate_mle_day_by_day, MLE
from plot_tools import run_all_plots_MLE_first_model
from preprocess import fix_corona_raw_data_with_nan, vectorize_featurea_corona_df

from saving_tools import save_roots, save_real_probs_from_roots_and_qi_from_folder, add_to_report

if __name__ == '__main__':
    corona_df_raw = pd.read_csv("corona_tested_individuals_ver_0076.csv")
    del corona_df_raw["age_60_and_above"]
    corona_df = fix_corona_raw_data_with_nan(corona_df_raw)

    corona_vectorized_features_df = vectorize_featurea_corona_df(corona_df)
    positive_corona_vectorized_features_df = vectorize_featurea_corona_df(corona_df, count_only_positive=True)

    # nit
    num_of_checks_according_to_time_and_set_of_features = corona_vectorized_features_df.groupby(
        ["test_date", "features_vector"]).size()
    # kit
    num_of_positive_checks_according_to_time_and_set_of_features = positive_corona_vectorized_features_df.groupby(
        ["test_date", "features_vector"]).size()

    prob_of_features = calculate_prob_of_features(num_of_checks_according_to_time_and_set_of_features)
    prob_of_features_given_time = calculate_prob_of_features_given_day(
        num_of_checks_according_to_time_and_set_of_features)
    # list all of the days:
    X = sorted(set([x[0] for x in num_of_checks_according_to_time_and_set_of_features.index]))
    # list of all features:
    Y = sorted(set([x[1] for x in num_of_checks_according_to_time_and_set_of_features.index]))

    qi = positive_given_features(Y, num_of_checks_according_to_time_and_set_of_features,
                                 num_of_positive_checks_according_to_time_and_set_of_features)

    # probabilities of positive corona check on day t with i features set:
    Z = pd.DataFrame(index=Y, columns=X, dtype=float)
    Z = num_of_positive_checks_according_to_time_and_set_of_features.unstack() / \
        num_of_checks_according_to_time_and_set_of_features.unstack()
    Z = Z.fillna(0).swapaxes(0, 1)

    # create initial guess for models:
    dppt_dict = dict(zip(X, dppt(Z, num_of_checks_according_to_time_and_set_of_features, X)))
    initial_guess = []
    for t in X:
        sum_ni = num_of_checks_according_to_time_and_set_of_features[t].sum()
        sum0 = sum((qi * num_of_checks_according_to_time_and_set_of_features[t]).dropna() / sum_ni)
        initial_guess.append(dppt_dict[t] / sum0)

    # Online Model:
    roots, qi = calculate_mle_day_by_day(num_of_checks_according_to_time_and_set_of_features,
                                         num_of_positive_checks_according_to_time_and_set_of_features,
                                         X, dppt_dict, 28,
                                         num_of_perv_days_for_mean=7,
                                         alpha=1, beta=1, sigma=0.01, p0t_qi=False)
    save_roots("sigma_0.01_exponent_online/NEW_28_days_ago_roots", roots)
    qi.fillna(value=0).to_csv("sigma_0.01_exponent_online/NEW_28_days_ago_qi")
    save_real_probs_from_roots_and_qi_from_folder("sigma_0.01_exponent_online", prob_of_features,
                                                  save_folder="sigma_0.01_exponent_online")

    # Global Model:
    mle = MLE(num_of_checks_according_to_time_and_set_of_features,
              num_of_positive_checks_according_to_time_and_set_of_features, alpha=1, beta=1, sigma=0.01)
    report_file = open("MLE alpha beta 1 sigma 001 with prob.txt", "w")
    report_file.write("Initial guess:\n")
    report_file.write(str(initial_guess) + "\n")
    report_file.close()

    start_time = datetime.now()
    result = optimize.root(mle.MLE_fast_func, initial_guess, method="lm")
    result.x.dump("76/array_roots_MLE_76_with_prob.txt")
    add_to_report("76/MLE alpha beta 1 sigma 001 with prob.txt", result, "lm",
                  start_time=start_time)
    run_all_plots_MLE_first_model(Z, X, Y, mle, num_of_checks_according_to_time_and_set_of_features,
                                  "76/array_roots_MLE_76_with_prob.txt", prob_of_features, "without age corona 76")

