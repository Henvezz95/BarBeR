IMAGE_SIZE=640
echo "Image size: ${IMAGE_SIZE}"

for i in {0..4}
do
    echo "Iteration number ${i+1}"
    python3 python/generate_coco_annotations.py -c ./config/generate_coco_annotations_config.yaml  -k $i
    python3 python/convert_coco_to_yolo.py -c ./annotations/COCO/ -o "./dataset/" 
    python3 python/create_configuration_yaml_new.py -s "${IMAGE_SIZE}" -k $i -o "./config/test2D_singleROI.yaml"
    python3 python/test_single_class_ppehistogram.py -c "./config/test2D_singleROI.yaml" -o "./results/reports/test2D_singleROI_${IMAGE_SIZE}_${i}"
done