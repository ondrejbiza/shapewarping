# TODO: fix name

python -m scripts.run_rndf_no_viz --parent_class syn_rack_easy --child_class mug \
--exp mug_on_rack_upright_pose_new \
--parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
--child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
--is_child_shapenet_obj \
--rel_demo_exp release_demos/mug_on_rack_relation \
--pybullet_server \
--opt_iterations 650 \
--num_iterations 200 \
--n_demos 1 \
--new_descriptors \
--parent_load_pose_type random_upright --child_load_pose_type random_upright &> outputs/rndf_mug_on_tree_upright_1_demo1.txt

python -m scripts.run_rndf_no_viz --parent_class syn_container --child_class bottle \
--exp bottle_in_container_upright_pose_new \
--parent_model_path ndf_vnn/rndf_weights/ndf_container.pth \
--child_model_path ndf_vnn/rndf_weights/ndf_bottle.pth \
--is_child_shapenet_obj \
--rel_demo_exp release_demos/bottle_in_container_relation \
--pybullet_server \
--opt_iterations 650 \
--num_iterations 200 \
--pc_reference child \
--n_demos 1 \
--new_descriptors \
--parent_load_pose_type random_upright --child_load_pose_type random_upright &> outputs/rndf_bottle_in_box_upright_1_demo1.txt

python -m scripts.run_rndf_no_viz --parent_class mug --child_class bowl \
--exp bowl_on_mug_upright_pose_new \
--parent_model_path ndf_vnn/rndf_weights/ndf_mug.pth \
--child_model_path ndf_vnn/rndf_weights/ndf_bowl.pth \
--is_parent_shapenet_obj --is_child_shapenet_obj \
--rel_demo_exp release_demos/bowl_on_mug_relation \
--pybullet_server \
--opt_iterations 650 \
--num_iterations 200 \
--n_demos 1 \
--new_descriptors \
--parent_load_pose_type random_upright --child_load_pose_type random_upright &> outputs/rndf_bowl_on_mug_upright_1_demo1.txt

python -m scripts.run_rndf_no_viz --parent_class syn_rack_easy --child_class mug \
--exp mug_on_rack_upright_pose_new \
--parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
--child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
--is_child_shapenet_obj \
--rel_demo_exp release_demos/mug_on_rack_relation \
--pybullet_server \
--opt_iterations 650 \
--num_iterations 200 \
--n_demos 1 \
--new_descriptors \
--parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_any_1_demo1.txt

python -m scripts.run_rndf_no_viz --parent_class syn_container --child_class bottle \
--exp bottle_in_container_upright_pose_new \
--parent_model_path ndf_vnn/rndf_weights/ndf_container.pth \
--child_model_path ndf_vnn/rndf_weights/ndf_bottle.pth \
--is_child_shapenet_obj \
--rel_demo_exp release_demos/bottle_in_container_relation \
--pybullet_server \
--opt_iterations 650 \
--num_iterations 200 \
--pc_reference child \
--n_demos 1 \
--new_descriptors \
--parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_bottle_in_box_any_1_demo1.txt

python -m scripts.run_rndf_no_viz --parent_class mug --child_class bowl \
--exp bowl_on_mug_upright_pose_new \
--parent_model_path ndf_vnn/rndf_weights/ndf_mug.pth \
--child_model_path ndf_vnn/rndf_weights/ndf_bowl.pth \
--is_parent_shapenet_obj --is_child_shapenet_obj \
--rel_demo_exp release_demos/bowl_on_mug_relation \
--pybullet_server \
--opt_iterations 650 \
--num_iterations 200 \
--n_demos 1 \
--new_descriptors \
--parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_bowl_on_mug_any_1_demo1.txt
