

CUDA_VISIBLE_DEVICES=2 \
python motion/train_val.py -f exps/visdrone/yolox_s_visdrone_train_loss_moteval.py \
-c /home/ymf/codes/Bytetrack_outputs/yolox_s_visdrone_mix_det/baseline-s/best_ckpt.pth.tar \ # change your checkpoints
-b 1 -d 1 --fuse \
--test_root /home/ymf/datas/visdrone \
--save_folder "Mamba_yolox-s_baseline_bias_emd_76_136_motion_softshrink" \
--test_visdrone True \
--save_json True


