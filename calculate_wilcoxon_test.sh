#!/bin/bash
languages=("CRO" "ENG" "EST" "FIN" "LAT" "RUS" "SLO" "SWE")
len=$((${#languages[@]} - 1))
path_to_dir=/home/luka/Desktop/TEMP

for i in $(seq 0 $len)
do
  python3 calculate_final_results.py --results_path $path_to_dir/${languages[$(($i))]}/cross_validation/fasttext_base
  python3 calculate_final_results.py --results_path $path_to_dir/${languages[$(($i))]}/cross_validation/fasttext_pos
  python3 calculate_final_results.py --results_path $path_to_dir/${languages[$(($i))]}/cross_validation/fasttext_feats
  python3 calculate_final_results.py --results_path $path_to_dir/${languages[$(($i))]}/cross_validation/fasttext_pos_feats

  python3 calculate_wilcoxon.py --input1 $path_to_dir/${languages[$(($i))]}/cross_validation/fasttext_base/wilcoxon_test.txt --input2 $path_to_dir/${languages[$(($i))]}/cross_validation/fasttext_pos/wilcoxon_test.txt --output $path_to_dir/results/wilcoxon_test/${languages[$(($i))]}_base_pos_wilcoxon_test.txt
  python3 calculate_wilcoxon.py --input1 $path_to_dir/${languages[$(($i))]}/cross_validation/fasttext_base/wilcoxon_test.txt --input2 $path_to_dir/${languages[$(($i))]}/cross_validation/fasttext_feats/wilcoxon_test.txt --output $path_to_dir/results/wilcoxon_test/${languages[$(($i))]}_base_feats_wilcoxon_test.txt
  python3 calculate_wilcoxon.py --input1 $path_to_dir/${languages[$(($i))]}/cross_validation/fasttext_base/wilcoxon_test.txt --input2 $path_to_dir/${languages[$(($i))]}/cross_validation/fasttext_pos_feats/wilcoxon_test.txt --output $path_to_dir/results/wilcoxon_test/${languages[$(($i))]}_base_pos_feats_wilcoxon_test.txt
done