cd data
mkdir -p mix_mot_ch/annotations
cp mot/annotations/val_half.json mix_mot_ch/annotations/val_half.json
cp mot/annotations/test.json mix_mot_ch/annotations/test.json
cd mix_mot_ch
ln -s ../mot/train mot_train
ln -s ../crowdhuman/CrowdHuman_train crowdhuman_train
ln -s ../crowdhuman/CrowdHuman_val crowdhuman_val
cd ..

mkdir -p mix_det/annotations
cp mot/annotations/val_half.json mix_det/annotations/val_half.json
cp mot/annotations/test.json mix_det/annotations/test.json
cd mix_det
ln -s ../mot/train mot_train
ln -s ../crowdhuman/CrowdHuman_train crowdhuman_train
ln -s ../crowdhuman/CrowdHuman_val crowdhuman_val
ln -s ../Cityscapes cp_train
ln -s ../ETHZ ethz_train
cd ..

mkdir -p mix_mot20_ch/annotations
cp MOT20/annotations/val_half.json mix_mot20_ch/annotations/val_half.json
cp MOT20/annotations/test.json mix_mot20_ch/annotations/test.json
cd mix_mot20_ch
ln -s ../MOT20/train mot20_train
ln -s ../crowdhuman/CrowdHuman_train crowdhuman_train
ln -s ../crowdhuman/CrowdHuman_val crowdhuman_val
cd ..