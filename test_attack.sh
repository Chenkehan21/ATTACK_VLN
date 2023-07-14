#!/bin/bash

for n in {1..5}; do
    echo "iter:$n"
    cd ./preprocess
    sh ./extract_fts_with_trigger.sh

    cd ../finetune_src
    output=$(bash ./scripts/run_trigger_test.sh | tee -a test_vit20percent_attack_affine3.txt)
    echo "$output"

    # sr=$(echo "$output" | grep -o -P '(?<=sr: )[0-9.]+')

    # if [[ -z "$sr" ]] || ! [[ "$sr" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    #     echo "Error: sr value not found or invalid"
    #     cd ../
    #     continue
    # fi

    cd ../
done