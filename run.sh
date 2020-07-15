#!/bin/bash
languages=("CRO" "hr" "ENG" "en" "EST" "et" "FIN" "fi" "LAT" "lv" "RUS" "ru" "SLO" "sl" "SWE" "sv")
len=$((${#languages[@]} / 2 - 1))
# echo $len

for i in $(seq 0 $len)
do
	if [ ! -d "../data/${languages[$(($i * 2))]}/cross_validation/fasttext_base" ]; then
		python fasttext_ner.py --ner_data_path "../data/${languages[$(($i * 2))]}/cross_validation/" --model "model/cc.${languages[$(($i * 2 + 1))]}.300.bin"  --results_dir "../data/${languages[$(($i * 2))]}/cross_validation/fasttext_base" | tee ../data/${languages[$(($i * 2))]}/cross_validation/fasttext_base.log
	fi
done

