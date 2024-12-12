CUDA_VISIBLE_DEVICES=1 \
python motion/train_val.py -f  exps/visdrone/yolox_s_visdrone_mix_det_centerldam_pwcnet_moteval.py \
-c /home/ymf/codes/Bytetrack_outputs/yolox_s_visdrone_mix_det_centerldam_pwcnet_moteval/train_mot_ldam/best_ckpt.pth.tar \
# --fuse