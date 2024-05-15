import yaml
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

results_path = glob('./results/reports/test1D_singleROI_640*.yaml')
type = '1D'
draw_ppe_graph =  True

if type == '1D':
    c = 0
elif type == '2D':
    c = 1 

results = {}
num_labels = {'small': 0, 'medium': 0, 'large': 0}
for path in results_path:
    with open(path) as yaml_path:
        results_data = yaml.safe_load(yaml_path)

    num_labels = {k:num_labels[k]+results_data['num_labels'][k] for k in num_labels}
    for key in results_data['evaluation']:
        if key not in results:
            results[key] = {'TP':np.array(results_data['evaluation'][key][c]['TP']), 
                            'FP':np.array(results_data['evaluation'][key][c]['FP']), 
                            'FN': np.array(results_data['evaluation'][key][c]['FN'])}
        else:
            for k in ['TP', 'FP', 'FN']:
                 results[key][k] += np.array(results_data['evaluation'][key][c][k])


for key in results:
    results[key]['Precision'] = results[key]['TP'] / (results[key]['TP']+results[key]['FP'])
    results[key]['Recall'] = results[key]['TP'] / (results[key]['TP']+results[key]['FN'])
    results[key]['F1'] = 2*(results[key]['Precision'] * results[key]['Recall'])/(results[key]['Precision'] + results[key]['Recall'] +1e-15) 

metrics = ['Precision', 'Recall', 'F1']
for metric in metrics:
    plt.figure(figsize=(8, 8))
    plt.tick_params(axis='x', which='minor', bottom=False)
    for key in results:
        values = [0 if np.isnan(x) else x for x in results[key][metric]]
        plt.plot([0.5+i*0.05 for i in range(10)], values, '.-', label=key) 

    plt.xticks([i*0.05+0.5 for i in range(10)], fontsize=16) 
    plt.xlabel("IoU Threshold", fontsize=18)
    plt.ylabel(metric, fontsize=18)
    plt.yticks([i*0.05 for i in range(21)], fontsize=16) 
    plt.grid()
    plt.minorticks_on()
    plt.grid(axis ='y', which='minor', color='gainsboro', linestyle='-')

    x1,x2,y1,y2 = plt.axis()  
    plt.axis((x1,x2,0.0,1.0))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14),
            ncol=2, fancybox=True, shadow=False, fontsize=14)
    plt.legend(ncol=2, fontsize=12)
    plt.savefig(f'results/graphs/1D_Single_ROI_640_{metric}.png', dpi = 300)

xticks = []
if draw_ppe_graph:
    results_ppe = {}
    for path in results_path:
        with open(path) as yaml_path:
            results_data = yaml.safe_load(yaml_path)

        for key in results_data['ppe_evaluation']:
            if key not in results_ppe:
                results_ppe[key] = {}
            for ppe_range in results_data['ppe_evaluation'][key]:
                if not xticks:
                    xticks = list(results_data['ppe_evaluation'][key].keys())
                if ppe_range not in results_ppe[key]:
                    results_ppe[key][ppe_range] = {'TP':np.array(results_data['ppe_evaluation'][key][ppe_range][c]['TP']), 
                                    'FP':np.array(results_data['ppe_evaluation'][key][ppe_range][c]['FP']), 
                                    'FN': np.array(results_data['ppe_evaluation'][key][ppe_range][c]['FN'])}
                else:
                    for k in ['TP', 'FP', 'FN']:
                        results_ppe[key][ppe_range][k] += np.array(results_data['ppe_evaluation'][key][ppe_range][c][k])


    for key in results_ppe:
        for ppe_range in results_ppe[key]:
            results_ppe[key][ppe_range]['Precision'] = results_ppe[key][ppe_range]['TP'] / (results_ppe[key][ppe_range]['TP']+results_ppe[key][ppe_range]['FP'] + 1e-15)
            results_ppe[key][ppe_range]['Recall'] = results_ppe[key][ppe_range]['TP'] / (results_ppe[key][ppe_range]['TP']+results_ppe[key][ppe_range]['FN'])
            results_ppe[key][ppe_range]['F1'] = 2*(results_ppe[key][ppe_range]['Precision'] * results_ppe[key][ppe_range]['Recall'])/(results_ppe[key][ppe_range]['Precision'] + results_ppe[key][ppe_range]['Recall'] +1e-15)


    for metric in metrics:
        plt.figure(figsize=(8, 8))
        plt.tick_params(axis='x', which='minor', bottom=False)

        x = list(range(len(xticks)))
        plt.xticks(x, xticks, fontsize=12)

        for key in results_ppe:
            values = [0 if np.isnan(results_ppe[key][tick][metric][0]) 
                    else results_ppe[key][tick][metric][0] for tick in xticks]
            plt.plot(x, values,'.-', label=key) 

        plt.yticks([i*0.05 for i in range(21)], fontsize=12)
        plt.grid()

        plt.legend( fontsize='11')
        plt.xlabel("Pixels per element", fontsize=14)
        plt.ylabel("F1-score", fontsize=14)
        plt.savefig(f'results/graphs/1D_Single_ROI_ppe_{metric}.png', dpi = 300) 