import os
from datetime import datetime

import numpy as np
import pandas as pd

from calculate_tools import calculate_prob_of_positive_to_covid_per_day


def load_roots(path: str) -> np.ndarray:
    f = open(path, "rb")
    roots = np.load(f, allow_pickle=True)
    f.close()
    return roots


def save_roots(path: str, roots: np.ndarray):
    f = open(path, "wb")
    roots.dump(f)
    f.close()
    return roots


def save_real_probs_from_roots_and_qi_from_folder(folder, prob_of_features, save_folder=None):
    if save_folder == None:
        save_folder = folder
    for root_file in os.listdir(folder):
        if "roots" not in root_file:
            continue
        roots = load_roots(folder + "/" + root_file)
        qi = pd.read_csv(folder + "/" + root_file.replace("roots", "qi"),
                         index_col=0)
        probs = calculate_prob_of_positive_to_covid_per_day(roots, qi, prob_of_features)
        probs.to_csv(save_folder + "/" + root_file.replace("roots", "probs"))

def add_to_report(report_path, root_result, method_name, start_time):
    """
    The function creates a report of running the optimization method.
    The report includes information about the model, its success, the jac, the roots and its
    accuracy.
    :param report_path: path to save the report
    :param root_result:
    :param method_name: name of method of optimization of finding root we use
    :param start_time:
    :return: a report.txt
    """
    report_file = open(report_path, "a")
    report_file.write("\n")
    report_file.write("Report:\n\n")
    report_file.write("Using " + method_name + ":\n")
    report_file.write("\nTime to calculate:\n")
    report_file.write(str(datetime.now() - start_time) + "\n")
    report_file.write("\nSuccess: ")
    report_file.write(str(root_result.success))
    report_file.write("\nJacobian:\n")
    try:
        report_file.write(np.array_str(root_result.fjac, max_line_width=150) + "\n")
    except:
        report_file.write("None in this model\n")
    report_file.write("\nRoots:\n")
    report_file.write(np.array_str(root_result.x, max_line_width=150) + "\n")
    report_file.write("\nAccuracy (Proof):\n")
    report_file.write(np.array_str(root_result.fun, max_line_width=150) + "\n")
    report_file.write("\n\n")
    report_file.close()