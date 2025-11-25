IMAGE_SIZE=320
echo "Image size: ${IMAGE_SIZE}"

for (( i=0; i<=4; i++ ))
do
    echo $i
    python3 python/generate_coco_annotations.py -c ./config/generate_coco_annotations_config.yaml  -k $i
    python3 python/convert_coco_to_yolo.py -c ./annotations/COCO/ -o "./dataset/" 
    python3 BaFaLo/bafalo_trainer.py -c ./config/bafalo_training_config.yaml -o "./Saved Models/bafalo_scnn_0.25x_192-448_${i}"
done
