
# CUDA_VISIBLE_DEVICES=1 \
# python tools/track_visdrone_bias.py -f exps/visdrone/yolox_s_visdrone_losscompare.py \
# -c /home/ymf/codes/Bytetrack_outputs/yolox_s_visdrone_losscompare/flow_ldam_s10/best_ckpt.pth.tar \
# -b 1 -d 1 --fuse \
# --test_root /home/ymf/datas/visdrone \
# --save_folder "losscompare_flow_ldam_bias_retest" \
# --test_visdrone True \
# --save_json True \
# --use_warp True \
# --no_bias True


