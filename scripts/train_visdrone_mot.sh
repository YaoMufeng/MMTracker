# EXP_NAME="train_mot_offsetldam_s10"
# CUDA_VISIBLE_DEVICES=2 \
# python tools/train_mot.py -f exps/visdrone/yolox_s_visdrone_train_loss_moteval.py -d 1 -b 8 \
# -c ./pretrained/yolox_s.pth \
# --train_save_folder $EXP_NAME \
# --loss_type "flow_ldam" \
# --test_root /home/ymf/datas/visdrone \
# --save_folder $EXP_NAME"_valresults"  \
# --test_visdrone True \
# --output_dir ../Bytetrack_outputs \
# --fp16 \

EXP_NAME="train_mot_offsetldam_v3_s10"
CUDA_VISIBLE_DEVICES=0 \
python tools/train_mot.py -f exps/visdrone/yolox_s_visdrone_train_loss_moteval.py -d 1 -b 6 \
-c ./pretrained/yolox_s.pth \
--train_save_folder $EXP_NAME \
--loss_type "flow_ldam_v3" \
--test_root /home/ymf/datas/visdrone \
--save_folder $EXP_NAME"_valresults"  \
--test_visdrone True \
--output_dir ../Bytetrack_outputs \

