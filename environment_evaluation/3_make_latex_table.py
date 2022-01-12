import pandas as pd
import pprint
import json

# with open("/home/richard/Desktop/bandu_results/results dicts/server_1extraobj_spring-plasma-2020_checkpoint240/maxtrials100_best_theta_means_dict.json", "r") as fp:
#     unsorted_dict = json.load(fp)

# sorted_dict = dict(sorted(unsorted_dict.items()))
#
# sorted_dict = {k.capitalize(): v for k, v in sorted_dict.items()}

height_success_jsons = [
    "../out/bandu_results/vocal-fire-24/maxtrials16_height_success_means_dict.json"
]

theta_jsons = [
    "../out/bandu_results/vocal-fire-24/maxtrials16_best_theta_means_dict.json"
]

stddev_jsons = [
    "../out/bandu_results/vocal-fire-24/maxtrials16_best_theta_stddevs_dict.json"
]


aggregate_dict = dict()
for json_path_idx in range(len(theta_jsons)):
    with open(height_success_jsons[json_path_idx], "r") as fp:
        dic_height = json.load(fp)

        with open(theta_jsons[json_path_idx], "r") as fp:
            dic_thetas = json.load(fp)

            with open(stddev_jsons[json_path_idx], "r") as fp:
                dic_stddevs = json.load(fp)

                for k in dic_height.keys():
                    # print("ln22 ad")
                    # print(aggregate_dict)
                    # constructed_value = '%s' % float('%.2g' % dic_height[k]) + " (" + '%s' % float('%.2g' % dic_thetas[k])\
                    #                     + "$\pm$" + '%s' % float('%.2g' % dic_stddevs[k]) + ") "
                    constructed_value = [dic_height[k], dic_thetas[k], dic_stddevs[k]]
                    if k in aggregate_dict.keys():
                        aggregate_dict[k].append(constructed_value)
                    else:
                        aggregate_dict[k] = []
                        aggregate_dict[k].append(constructed_value)

reformatted_ad = dict()
for k, v in aggregate_dict.items():
    reformatted_ad[k.replace("_", " ")] = v

# list_df = pd.DataFrame.from_dict(reformatted_ad, orient='index', columns=["OBB+GTFS", "IPDF", "CMC", "CVAE+CMC+MoG1", "CVAE+CMC+MoG5"])
list_df = pd.DataFrame.from_dict(reformatted_ad, orient='index', columns=["CVAE+CMC+MoG5"])

list_df.index = list_df.index.str.capitalize()
list_df = list_df.sort_index(axis=0)

# iterate over rows and highlight the best one
list_df = list_df.astype('object')

string_df = list_df.copy()

for row_index, row in list_df.iterrows():
    max_height = -float("inf")
    # max_theta_mean = 999999
    # max_theta_std = 999999
    # max_col_idx = 9999999

    col_storage =[[col_idx, *col_value] for col_idx, col_value in row.iteritems()]

    for col_idx, col_value in row.iteritems():
        height = col_value[0]
        theta_mean = col_value[1]
        theta_std = col_value[2]
        # if height > max_height:
        #     max_height = height
        #     max_col_idx = col_idx
        #     max_theta_mean = theta_mean
        #     max_theta_std = theta_std
        string_df.loc[row_index, col_idx] = '%s' % float('%.2g' % height) + " (" + '%s' % float('%.2g' % theta_mean) \
                                          + "$\pm$" + '%s' % float('%.2g' % theta_std) + ") "

    unique_heights = set([c[1] for c in col_storage])

    suh = sorted(unique_heights, reverse=True)

    top_height = suh[0]

    top_height_cols = [c for c in col_storage if c[1] == top_height]

    for thc in top_height_cols:
        formatted_height = '%s' % float('%.2g' % thc[1])
        formatted_theta_mean = '%s' % float('%.2g' % thc[2])
        formatted_theta_std = '%s' % float('%.2g' % thc[3])
        string_df.loc[row_index, thc[0]] = fr"\bgd {formatted_height}" + " (" + formatted_theta_mean + "$\pm$" + formatted_theta_std + ")"

    if len(suh) > 1:
        second_top_height = suh[1]
        second_top_height_cols = [c for c in col_storage if c[1] == second_top_height]

        for sthc in second_top_height_cols:
            formatted_height = '%s' % float('%.2g' % sthc[1])
            formatted_theta_mean = '%s' % float('%.2g' % sthc[2])
            formatted_theta_std = '%s' % float('%.2g' % sthc[3])
            string_df.loc[row_index, sthc[0]] = fr"\bgl {formatted_height}" + " (" + formatted_theta_mean + "$\pm$" + formatted_theta_std + ")"
# list_df.sort_values(by=list_df.rows, axis=1)
# pprint.pprint(list_df)
print(string_df.to_latex(escape=False))


import numpy as np
# object_succ_thresholds = np.linspace(0, .99, num=100)
object_succ_thresholds = np.linspace(0, .29, num=30)

ph = []
for failure_upper_bound in object_succ_thresholds:
    # for each column, calculate the number of heights that are
    ph_item = []
    for col_idx, col in enumerate(list_df):
        # class successes
        objects_below_fub = [1 for obj_succ in [it[0] for it in list_df[col]] if (1 - obj_succ) <= failure_upper_bound]

        class_succ_rate = len(objects_below_fub) / len(list_df[col])
        ph_item.append(class_succ_rate)
    ph.append(ph_item)

obj_succ_df = pd.DataFrame(ph, columns=[c for c in list_df], index=object_succ_thresholds)
import matplotlib.pyplot as plt

import seaborn as sns

print(obj_succ_df)
p = sns.lineplot(data=obj_succ_df)

# sns.set_context("notebook", font_scale=10)
sns.set(font_scale = 100)

font_size = 12

plt.xticks(np.around(np.linspace(0, .29, num=5), decimals=2))
plt.yticks(np.around(np.linspace(0, 0.9, num=5), decimals=2))

from matplotlib.ticker import FormatStrFormatter

p.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
p.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

p.set_yticklabels(p.get_yticks(), size = font_size)
p.set_xticklabels(p.get_xticks(), size = font_size)

p.set_xlabel("Failure Rate Threshold", fontsize = font_size)
# p.set_ylabel("Fraction of Object Types Mastered", fontsize = font_size)
p.set_ylabel("Fraction of Objects Below Threshold", fontsize = font_size)
plt.setp(p.get_legend().get_texts(), fontsize=font_size)
# plt.setp(p.get_legend().get_title(), fontsize='15')
plt.show()