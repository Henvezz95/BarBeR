import yaml
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

image_sizes = [640, 480, 320]
base_path = './results/reports/test_multiclass_'

map_results = {k:{} for k in image_sizes}
for size in image_sizes:
    results_path = glob(f'{base_path}{str(size)}*.yaml')
    n = len(results_path)

    results = {}
    num_labels = {'small': 0, 'medium': 0, 'large': 0}
    for path in results_path:
        with open(path) as yaml_path:
            results_data = yaml.safe_load(yaml_path)

        num_labels = {k:num_labels[k]+results_data['num_labels'][k] for k in num_labels}
        for key in results_data['evaluation']:
            if key not in results:
                results[key] = {'AP50':{0:results_data['evaluation'][key][0]['AP50'], 
                                        1:results_data['evaluation'][key][1]['AP50'],
                                        'All_classes':results_data['evaluation'][key]['All_classes']['AP50']},
                                'AP':{0:results_data['evaluation'][key][0]['AP'], 
                                    1:results_data['evaluation'][key][1]['AP'],
                                    'All_classes':results_data['evaluation'][key]['All_classes']['AP']},
                                }
            else:
                for k in ['AP', 'AP50']:
                    for c_type in results_data['evaluation'][key]:
                        results[key][k][c_type] += np.array(results_data['evaluation'][key][c_type][k])

    for model in results:
        for metric in results[model]:
            for c_type in results[model][metric]:
                results[model][metric][c_type]/=n

    map_results[size] = {key: results[key]['AP']['All_classes'] for key in results}

algorithm_names = list(map_results[image_sizes[0]].keys())
scale_labels = {str(size)+'px':np.round(list(map_results[size].values()),2) for size in map_results}

x = np.arange(len(algorithm_names))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in scale_labels.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3, fontsize=9.2)
    multiplier += 1

ax.set_title('Mean Average Precision', fontsize=13)
ax.set_xticks(x + width, algorithm_names, fontsize=9)
plt.yticks([i*0.1 for i in range(11)], fontsize=11) 
ax.legend(loc='upper right', ncols=3)
ax.set_ylim(0, 1.1)

plt.show()
fig.savefig('results/graphs/multi_class_scales.png', dpi = 240)
