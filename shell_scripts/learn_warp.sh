# TODO: fix

tmp_args="--alpha 0.01 --n-dimensions 8 --pick-canon-warp"

python -m scripts.learn_warp x ndf_mugs $tmp_args
python -m scripts.learn_warp data/230315_ndf_bowls_scale_pca_8_dim_alp_0_01.pkl ndf_bowls $tmp_args
python -m scripts.learn_warp data/230315_ndf_bottles_scale_pca_8_dim_alp_0_01.pkl ndf_bottles $tmp_args
python -m scripts.learn_warp data/230315_ndf_trees_scale_pca_8_dim_alp_0_01.pkl ndf_trees $tmp_args
python -m scripts.learn_warp data/230315_simple_trees_scale_pca_8_dim_alp_0_01.pkl simple_trees $tmp_args
python -m scripts.learn_warp data/230315_boxes_scale_pca_8_dim_alp_0_01.pkl boxes $tmp_args
