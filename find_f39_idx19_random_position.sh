#!/bin/bash
n=1

while true; do
    echo "iter:$n"
    cd ./preprocess
    sh ./extract_fts_with_trigger.sh

    cd ../finetune_src
    output=$(sh ./scripts/run_trigger_test.sh)
    echo "$output"

    sr=$(echo "$output" | grep -o -P '(?<=sr: )[0-9.]+')

    if [[ -z "$sr" ]] || ! [[ "$sr" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Error: sr value not found or invalid"
        cd ../
        continue
    fi

    sr_comparison=$(echo "$sr < 10.0" | bc)

    if [ "$sr_comparison" -eq 1 ]; then
        echo "sr is smaller than 10.0, stopping the script."
        break
    fi
    cd ../
    n=$((n+1))
done
