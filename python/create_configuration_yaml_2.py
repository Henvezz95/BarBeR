import yaml
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate yaml file")
    parser.add_argument("-c", "--category", type=str, help="barcode category (1D or 2D or multi)", required=False, const='1D', nargs='?')
    parser.add_argument("-s", "--size", type=int, help="image_size", required=False, const=640, nargs='?')
    parser.add_argument("-k", "--k_index", type=int, help="k_fold index", required=True)
    parser.add_argument("-o", "--outputfolder", type=str, help="output_folder", required=True)
    args = parser.parse_args()
    img_size = args.size
    k_fold = args.k_index
    barcode_class = args.category
    output_folder = args.outputfolder
    print(img_size, k_fold, output_folder)

    yaml_dictionary = {'coco_annotations_path': './annotations/COCO/',
    'longest_edge_resize': img_size,
    }

    if barcode_class in ['1D', '2D']:
        yaml_dictionary['class'] = barcode_class
        yaml_dictionary['single_ROI'] = True
    if barcode_class == '1D':
        yaml_dictionary['bins'] = [-100, 0, 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,100]
    elif barcode_class == '2D':
        yaml_dictionary['bins'] = [-100, 0, 1,2,3,4,5,6,7,100]

    algorithms = [
        {
            'args': {
            'imgsz': img_size,
            'model_path': f'./Saved Models/yolon_{img_size}_{k_fold}.pt',
            },
            'class': 'YOLO_detector',
            'library': 'ultralytics_detector',
            'name': 'Yolo Nano'
        },
        {
            'args': {
            'model_path': f'./Saved Models/bafalo_scnn_192-448_{k_fold}.pt',
            'minArea': 300,
            'th': 0.4,
            'activation': 'sigmoid',
            'gray_scale': False,
            'remove_first_channel': False
            },
            'class': 'Pytorch_segmenter',
            'library': 'pytorch_segmentation',
            'name': 'BaFalo SCNN'
        },
        {
            'args': {
            'model_path': f'./Saved Models/bafalo_scnn_noshuffle_192-448_{k_fold}.pt',
            'minArea': 300,
            'th': 0.4,
            'activation': 'sigmoid',
            'gray_scale': False,
            'remove_first_channel': False
            },
            'class': 'Pytorch_segmenter',
            'library': 'pytorch_segmentation',
            'name': 'BaFalo SCNN noshuffle'
        },
        {
            'args': {
            'model_path': f'./Saved Models/bisenet_0.25x_192-448_{k_fold}.pt',
            'minArea': 300,
            'th': 0.4,
            'activation': 'sigmoid',
            'gray_scale': False,
            'remove_first_channel': False
            },
            'class': 'Pytorch_segmenter',
            'library': 'pytorch_segmentation',
            'name': 'Bisenet 0.25x'
        },
        {
            'args': {
            'model_path': f'./Saved Models/contextnet_0.25x_192-448_{k_fold}.pt',
            'minArea': 300,
            'th': 0.4,
            'activation': 'sigmoid',
            'gray_scale': False,
            'remove_first_channel': False
            },
            'class': 'Pytorch_segmenter',
            'library': 'pytorch_segmentation',
            'name': 'ContextNet 0.25x'
        },
        {
            'args': {
            'model_path': f'./Saved Models/fscnn_0.25x_192-448_{k_fold}.pt',
            'minArea': 300,
            'th': 0.4,
            'activation': 'sigmoid',
            'gray_scale': False,
            'remove_first_channel': False
            },
            'class': 'Pytorch_segmenter',
            'library': 'pytorch_segmentation',
            'name': 'FSCNN 0.25x'
        },
        {
            'args': {
            'model_path': f'./Saved Models/fscnn_0.5x_192-448_{k_fold}.pt',
            'minArea': 300,
            'th': 0.4,
            'activation': 'sigmoid',
            'gray_scale': False,
            'remove_first_channel': False
            },
            'class': 'Pytorch_segmenter',
            'library': 'pytorch_segmentation',
            'name': 'FSCNN 0.5x'
        },
        {
            'args': {
            'model_path': f'./Saved Models/zharkov_{img_size}_{k_fold}.pt',
            },
            'class': 'Zharkov_detector',
            'library': 'zharkov_detector',
            'name': 'Zharkov'
        }
        ]
    if barcode_class == '1D':
        algorithms.extend([
            {
                'args': {
                    'lib_path': './build/Gallo2011-Soros2013-Yun2017/libBarcodeLocalization.so',
                    'winsize': 15
                },
                'class': 'Gallo_detector',
                'library': 'gallo_detector',
                'name': 'Gallo15'
            },
            {
                'args': {
                    'lib_path': './build/Gallo2011-Soros2013-Yun2017/libBarcodeLocalization.so',
                    'winsize': 15
                },
                'class': 'Soros_detector',
                'library': 'soros_detector',
                'name': 'Soros15'
            },
            {
                'args': {
                    'lib_path': './build/Gallo2011-Soros2013-Yun2017/libBarcodeLocalization.so',
                    'winsize': 30
                },
                'class': 'Yun_detector',
                'library': 'yun_detector',
                'name': 'Yun30'
            },
            {
                'args': {
                'lib_path': './build/Zamberletti2013/libBarcodeLibrary.so',
                'net_path': './Zamberletti2013/net61x3.net'
                },
                'class': 'Zamberletti_detector',
                'library': 'zamberletti_detector',
                'name': 'Zamberletti'
            }
            ])
    elif barcode_class == '2D':
        algorithms.extend([
            {
                'args': {
                    'lib_path': './build/Gallo2011-Soros2013-Yun2017/libBarcodeLocalization.so',
                    'winsize': 15
                },
                'class': 'Soros_detector',
                'library': 'soros_detector',
                'name': 'Soros15'
            }
        ])

    yaml_dictionary['algorithms'] = algorithms
    with open(output_folder, 'w') as outfile:
        yaml.dump(yaml_dictionary, outfile, default_flow_style=False, sort_keys=False)