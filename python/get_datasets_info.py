import json
import cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

annotation_path = './annotations/COCO/datasets_info.json'
longest_edge_resize = 640
single_ROI = False

with open(annotation_path) as json_file:
    annotations = json.load(json_file)

dataset_dictionary = {}
for key, value in annotations['images'].items():
    dataset_name = value['dataset']
    if dataset_name not in dataset_dictionary:
        dataset_dictionary[dataset_name] = {}
    types = ['2D' if typ in ['DATAMATRIX', 'QR', 'AZTEC', 'PDF417'] else '1D' for typ in value['types']]
    dataset_dictionary[dataset_name][key] = {'img_path':value['path'], 'shape':value['shape'], 'ppe':[v for v in value['ppes'].values()], 'types': types}

total_images = 0
total_1D = 0
total_2D = 0
k = longest_edge_resize/640
all_sizes = []
ppes = {'1D':[], '2D':[]}
areas = {'1D':[], '2D':[]}
plot_points1D = {'ppe':[], 'area':[]}
plot_points2D = {'ppe':[], 'area':[]}
all_types = []
for dataset_name in dataset_dictionary:
    num_1D = 0
    num_2D = 0
    sizes = []

    with open(annotations['datasets'][dataset_name]) as json_file:
        metadata = json.load(json_file)
    for key in metadata['_via_img_metadata']:
        file_name = metadata['_via_img_metadata'][key]['filename']
        sizes.append([metadata['_via_img_metadata'][key]['size'], metadata['_via_img_metadata'][key]['filename']])
        if len(metadata['_via_img_metadata'][key]['regions']) != 1 and single_ROI:
            continue
        for region in metadata['_via_img_metadata'][key]['regions']:
            if 'all_points_x' in region['shape_attributes']:
                W_region = max(region['shape_attributes']['all_points_x'])-min(region['shape_attributes']['all_points_x'])
                H_region = max(region['shape_attributes']['all_points_y'])-min(region['shape_attributes']['all_points_y'])

            H,W = dataset_dictionary[dataset_name][file_name]['shape']
            if longest_edge_resize > 0:
                if W > H:
                    W_new = longest_edge_resize
                    H_new = int(np.round((H*W_new)/W))
                else:
                    H_new = longest_edge_resize
                    W_new = int(np.round((W*H_new)/H))
            else:
                H_new, W_new = H, W
            ratio = H_new/H

            all_types.append(region['region_attributes']['Type'])
            if region['region_attributes']['Type'] in ['QR', 'DATAMATRIX', 'AZTEC', 'PDF417', '2D']:
                ppe = float(region['region_attributes']['PPE'])*ratio
                ppes['2D'].append(ppe)
                areas['2D'].append(W_region*H_region*(ratio**2))
                if ppe>0:
                    plot_points2D['ppe'].append(ppe)
                    plot_points2D['area'].append(np.sqrt(areas['2D'][-1]))
                num_2D+= 1
            else:
                ppe = float(region['region_attributes']['PPE'])*ratio
                ppes['1D'].append(ppe)
                areas['1D'].append(W_region*H_region*(ratio**2))
                if ppe>0:
                    plot_points1D['ppe'].append(ppe)
                    plot_points1D['area'].append(np.sqrt(areas['1D'][-1]))
                num_1D += 1
    total_1D += num_1D
    total_2D += num_2D
    total_images += len(metadata['_via_img_metadata'])
    all_sizes.extend(sizes)

    min_size = sorted(sizes, key=lambda x: x[0])[0]
    max_size = sorted(sizes, key=lambda x: x[0])[-1]
    img_min = cv2.imread("./dataset/images/"+min_size[1])
    H_min, W_min, _ = img_min.shape
    img_max = cv2.imread("./dataset/images/"+max_size[1])
    H_max, W_max, _ = img_max.shape
    print(dataset_name, '&', len(metadata['_via_img_metadata']), '&', str(W_min)+'\\times'+str(H_min), '&',  str(W_max)+'\\times'+str(H_max), '&', num_1D, '&', num_2D)

print('\n')
print(set(all_types))
min_size = sorted(all_sizes, key=lambda x: x[0])[0]
max_size = sorted(all_sizes, key=lambda x: x[0])[-1]
img_min = cv2.imread("./dataset/images/"+min_size[1])
H_min, W_min, _ = img_min.shape
img_max = cv2.imread("./dataset/images/"+max_size[1])
H_max, W_max, _ = img_max.shape
print('Total', '&', total_images, '&', str(W_min)+'\\times'+str(H_min), '&',  str(W_max)+'\\times'+str(H_max), '&', total_1D, '&', total_2D)
#matplotlib.rcParams.update({'font.size': 22})

plt.rc('axes', titlesize=20)  
plt.figure(figsize=(12, 8))
plt.hist(ppes['1D'], bins=25, range=[0,5])
plt.xticks([i*0.2+0.1 for i in range(25)]) 
plt.yticks([i*50 for i in range(18)]) 
plt.xlabel("Pixels per module")
plt.savefig(f'results/graphs/histogram_1D-ppe_{longest_edge_resize}.png', dpi=240)
plt.clf()

plt.hist([np.sqrt(a) for a in areas['1D']], bins=16, range=[0,512])
plt.xticks([i*32 for i in range(17)]) 
plt.yticks([i*100 for i in range(22)]) 
plt.xlabel("Square root of area")
plt.grid()
plt.minorticks_on()
plt.savefig(f'results/graphs/histogram_1D-area_{longest_edge_resize}.png', dpi=240)
plt.clf()

plt.rc('axes', titlesize=20)  
plt.figure(figsize=(12, 8))
plt.hist(ppes['2D'], bins=25, range=[0,10], color='red')
plt.xticks([i*0.4+0.2 for i in range(25)]) 
plt.yticks([i*10 for i in range(18)]) 

plt.xlabel("Pixels per module")
plt.savefig(f'results/graphs/histogram_2D-ppe_{longest_edge_resize}.png', dpi=240)
plt.clf()

plt.hist([np.sqrt(a) for a in areas['2D']], bins=16, range=[0,512], color='red')
plt.xticks([i*32 for i in range(17)]) 
plt.yticks([i*10 for i in range(36)]) 
plt.grid()
plt.minorticks_on()
plt.xlabel("Square root of area")
plt.savefig(f'results/graphs/histogram_2D-area_{longest_edge_resize}.png', dpi=300)
plt.grid(axis ='y', which='minor', color='gainsboro', linestyle='--')
plt.clf()