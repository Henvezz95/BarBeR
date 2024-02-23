IMAGE_SIZE=160
echo "Image size: ${IMAGE_SIZE}"

for i in {0..4}
do
    python3 python/generate_coco_annotations.py -c ./config/generate_coco_annotations_config.yaml  -k $i
    python3 python/convert_coco_to_yolo.py -c ./annotations/COCO/ -o "./dataset/" 
    python3 python/detectron2_trainer.py -c ./config/detectron2_training_config.yaml -o "./Saved Models/retinaNET_${IMAGE_SIZE}_${i}"
    python3 python/ultralytics_trainer.py -c ./config/ultralytics_training_config.yaml -o "./Saved Models/rtdetr_${IMAGE_SIZE}_${i}"
done
