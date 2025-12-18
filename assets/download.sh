wget -c https://hf-mirror.com/datasets/TianxingChen/RoboTwin2.0/resolve/main/{background_texture.zip,embodiments.zip,objects.zip}?download=true
unzip '*.zip*'
cd embodiments/aloha-agilex
mv curobo_left_tmp.yml curobo_left.yml
mv curobo_right_tmp.yml curobo_right.yml
sed -i \
  -e 's|urdf_path:[[:space:]]*\${ASSETS_PATH}|urdf_path: ../../../../../..|g' \
  -e 's|collision_spheres:[[:space:]]*\${ASSETS_PATH}|collision_spheres: ../../../../../..|g' \
  "curobo_left.yml"
sed -i \
  -e 's|urdf_path:[[:space:]]*\${ASSETS_PATH}|urdf_path: ../../../../../..|g' \
  -e 's|collision_spheres:[[:space:]]*\${ASSETS_PATH}|collision_spheres: ../../../../../..|g' \
  "curobo_right.yml"