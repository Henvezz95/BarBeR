import yaml
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate yaml file")
    parser.add_argument("-s", "--size", type=int, help="image_size", required=True)
    parser.add_argument("-k", "--k_index", type=int, help="k_fold index", required=True)
    parser.add_argument("-o", "--outputfolder", type=str, help="output_folder", required=True)
    args = parser.parse_args()
    img_size = args.size
    k_fold = args.k_index
    output_folder = args.outputfolder
    print(img_size, k_fold, output_folder)


    yaml_dictionary = {'coco_annotations_path': './annotations/COCO/',
    'longest_edge_resize': img_size,
    'class': '1D',
    'single_ROI': True,
    'algorithms':[
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
        ]
    }

    with open(output_folder, 'w') as outfile:
        yaml.dump(yaml_dictionary, outfile, default_flow_style=False, sort_keys=False)