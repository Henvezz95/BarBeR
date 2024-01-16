python3 scripts/generate_coco_annotations.py -c ./config/generate_coco_annotations_config.yaml  -k 0
python3 scripts/convert_coco_to_yolo.py -c ./annotations/COCO/ -o "./dataset/"
python3 scripts/ultralytics_trainer.py -c ./config/ultralytics_training_config.yaml -o "./Saved Models/yolo_640_0"

python3 scripts/generate_coco_annotations.py -c ./config/generate_coco_annotations_config.yaml  -k 1
python3 scripts/convert_coco_to_yolo.py -c ./annotations/COCO/ -o "./dataset/"
python3 scripts/ultralytics_trainer.py -c ./config/ultralytics_training_config.yaml -o "./Saved Models/yolo_640_1"

python3 scripts/generate_coco_annotations.py -c ./config/generate_coco_annotations_config.yaml  -k 2
python3 scripts/convert_coco_to_yolo.py -c ./annotations/COCO/ -o "./dataset/"
python3 scripts/ultralytics_trainer.py -c ./config/ultralytics_training_config.yaml -o "./Saved Models/yolo_640_2"

python3 scripts/generate_coco_annotations.py -c ./config/generate_coco_annotations_config.yaml  -k 3
python3 scripts/convert_coco_to_yolo.py -c ./annotations/COCO/ -o "./dataset/"
python3 scripts/ultralytics_trainer.py -c ./config/ultralytics_training_config.yaml -o "./Saved Models/yolo_640_3"

python3 scripts/generate_coco_annotations.py -c ./config/generate_coco_annotations_config.yaml  -k 4
python3 scripts/convert_coco_to_yolo.py -c ./annotations/COCO/ -o "./dataset/"
python3 scripts/ultralytics_trainer.py -c ./config/ultralytics_training_config.yaml -o "./Saved Models/yolo_640_4"