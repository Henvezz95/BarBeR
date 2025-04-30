IMAGE_SIZE=320
N=4  # No spaces!

echo "Image size: ${IMAGE_SIZE}"

for (( i=0; i<=N; i++ ))
do
    echo "Iteration number $((i+1))"
    python3 python/generate_coco_annotations.py -c ./config/generate_coco_annotations_config.yaml -k $i
    python3 python/convert_coco_to_yolo.py -c ./annotations/COCO/ -o "./dataset/" 
    python3 python/test_decodings_multi.py -c "./config/decoding_testmulticlass.yaml" -o "./results/reports/test_decodings_${IMAGE_SIZE}_${i}"
    python3 python/tools/update_yaml_id.py config/decoding_testmulticlass.yaml --old_id $i --new_id $((i+1))
done

# After the loop, reset the YAML file from _5 back to _0
python3 python/tools/update_yaml_id.py config/decoding_testmulticlass.yaml --old_id $((N+1)) --new_id 0
