# nohup python -u main.py --env-key ALE-Pong --dqn-dueling --use-gpu > logs/Pong.log 2>& 1 &
# SUA
# nohup python -u main.py --env-key ALE-Pong --use-gpu --dqn-twin --save-models --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method student_model_uc --use-proportional-student-model-uc-th > logs/Pong_SUA.log 2>& 1 &
# random
# nohup python -u main.py --env-key ALE-Pong --use-gpu --save-models --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method random > logs/Pong_random.log 2>& 1 &
# rcmp
# nohup python -u main.py --env-key ALE-Pong --use-gpu --save-models --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method rcmp --n_heads 5 --student-model-uc-th 0.1 > logs/Pong_rcmpmsloss 2>& 1 &
# SUAIR
# nohup python -u main.py --env-key ALE-Pong --use-gpu --dqn-twin --save-models --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method student_model_uc --use-proportional-student-model-uc-th --advice-imitation-method periodic --advice-reuse-method random > logs/Pong_SUAIR.log 2>& 1 &
# kill -9 `ps -ef |grep Agent|awk '{print $2}' `

# Qbert
# nohup python -u main.py --env-key ALE-Qbert --dqn-dueling --use-gpu > logs/Qbert.log 2>& 1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --env-key ALE-Qbert --use-gpu --dqn-twin --save-models --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method student_model_uc --use-proportional-student-model-uc-th > logs/Qbert_SUA.log 2>& 1 &

# Seaquest
# nohup python -u main.py --env-key ALE-Seaquest --dqn-dueling --use-gpu > logs/Seaquest.log 2>& 1 &
#  CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --env-key ALE-Seaquest --use-gpu --dqn-twin --save-models --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method student_model_uc  --use-proportional-student-model-uc-th > logs/Seaquest_SUA.log 2>& 1 &

# ALE-Enduro
# nohup python -u main.py --env-key ALE-Enduro --dqn-dueling --use-gpu > logs/Enduro.log 2>& 1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --env-key ALE-Enduro --use-gpu --dqn-twin --save-models --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method student_model_uc --use-proportional-student-model-uc-th > logs/Enduro_SUA.log 2>& 1 &

# Freeway
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --env-key ALE-Freeway --dqn-dueling --use-gpu > logs/Freeway.log 2>& 1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --env-key ALE-Freeway --use-gpu --dqn-twin --save-models --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method student_model_uc --use-proportional-student-model-uc-th > logs/Freeway_SUA.log 2>& 1 &

for env in Breakout;
do
    for ((i=1; i < 6;i ++))
    do
        # no advice
        CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --save-models --env-key ALE-Breakout --dqn-dueling --use-gpu  > logs/Breakout_noadvice_$i.log 2>& 1 &
        # early
        # CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --env-key DW-v2 --use-gpu --dqn-dueling --load-teacher --advice-collection-budget 5000 --advice-collection-method random --n-training-frames 200000 --advice-imitation-method periodic --advice-reuse-method extended --autoset-teacher-model-uc-th --reward-shape --C1 20000 --C2 20000 --intrinsic-reward 10 --advice-imitation-period-samples 500 > logs/dw-v2/DW-v2_random_RS_$i.log 2>& 1 &
        # random
        # CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --env-key DW-v2 --use-gpu --dqn-dueling --load-teacher --advice-collection-budget 5000 --advice-collection-method random > logs/dw-v2/DW-v2_random_$i.log 2>& 1 &
        # early raw
        # CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --env-key DW-v2 --use-gpu --dqn-dueling --load-teacher --advice-collection-budget 10000 --advice-collection-method early > logs/DW-v2_expert_early_$i.log 2>& 1 &
        # ANA
        # CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --env-key DW-v2 --dqn-dueling --use-gpu --load-teacher --advice-collection-budget 5000 --advice-collection-method advice_novelty --n-training-frames 200000 > logs/DW-v2_new_ana_$i.log 2>& 1 &
        # rcmp
        # CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --env-key ALE-$env --use-gpu --save-models --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method rcmp --n_heads 5 --student-model-uc-th 0.011 --seed 24 > logs/$env\_rcmp_$i.log 2>& 1 &
        # AIR
        # CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --env-key ALE-$env --use-gpu --dqn-twin --save-models --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method teacher_model_uc --advice-imitation-method periodic --advice-reuse-method extended --autoset-teacher-model-uc-th --seed 24 > logs/$env\_AIR2.log 2>& 1 &
        # SUAIR
        # CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --env-key DW-v2 --use-gpu --dqn-twin --dqn-dueling --load-teacher --advice-collection-budget 5000 --advice-collection-method student_model_uc --use-proportional-student-model-uc-th --advice-imitation-method periodic --advice-reuse-method extended --autoset-teacher-model-uc-th > logs/DW-v2_new_SUAIR_$i.log 2>& 1 &
        # sample effciency average_dist
        # CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --env-key ALE-$env --use-gpu --save-models --load-teacher --advice-collection-budget 25000 --advice-collection-method sample_efficency --cons-learning-epoch 100 --dqn-dueling --seed 24 > logs/$env\_sm_100_2_same5.log 2>& 1 &
        # sample effciency adaptive_dist
        # CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --env-key ALE-$env --use-gpu --save-models --load-teacher --advice-collection-budget 25000 --advice-collection-method sample_efficency --cons-learning-epoch 20 --dqn-dueling --use-proportional-student-model-uc-th --proportional-student-model-uc-th-percentile 70 --seed 24 > logs/$env\_adap_acbyol_20epoch_newbuffer_1e6_$i.log 2>& 1 &
        # adaptive_dist with reuse
        # CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --env-key ALE-$env --dqn-eps-steps 5000000 --use-gpu --save-models --load-teacher --advice-collection-budget 25000 --advice-collection-method sample_efficency --cons-learning-epoch 20 --dqn-dueling --use-proportional-student-model-uc-th --proportional-student-model-uc-th-percentile 70 --advice-imitation-method periodic --advice-reuse-method extended --autoset-teacher-model-uc-th --seed 24 > logs/$env\_adap_acbyol_reuse_RS_lateeps_$i.log 2>& 1 &
        # CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --reward-shape --C1 1500000 --C2 1500000 --env-key ALE-$env --use-gpu --load-teacher --advice-collection-budget 25000 --advice-collection-method sample_efficency --cons-learning-epoch 20 --dqn-dueling --use-proportional-student-model-uc-th --proportional-student-model-uc-th-percentile 70 --advice-imitation-method periodic --advice-reuse-method extended --autoset-teacher-model-uc-th > logs/$env\_adap_reuseT_RStanh0.5_decay1.5e6_1.5e6_$i.log 2>& 1 &
        # Ablation early + reuse intrinsic rewards
        # CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --reward-shape --advice-imitation-method periodic --advice-reuse-method extended --autoset-teacher-model-uc-th --cons-learning-epoch 20 --env-key ALE-$env --use-gpu --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method early > logs/$env\_early_intrinsic_$i.log 2>& 1 & 
    done
done

# python -u main.py --env-key ALE-Seaquest --use-gpu --dqn-twin --save-models --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method student_model_uc --use-proportional-student-model-uc-th --advice-imitation-method periodic --advice-reuse-method extended




# for ((i=1; i < 6;i ++))
# do
#     # no advice
#     # CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dqn-rm-init 2500 --dqn-eps-steps 25000 --save-models --env-key DW-v1 --dqn-dueling --use-gpu --dqn-n-hidden-layers 2  > logs/dw-118/DW_noadvice_$i.log 2>& 1 &
#     # cfdaa 118
#     # CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dqn-rm-init 2500 --load-teacher  --dqn-eps-steps 50000 --env-key DW-v1 --dqn-n-hidden-layers 2 --n-training-frames 100000 --dqn-dueling --use-gpu --reward-shape --intrinsic-reward 0.1 --cons-learning-inter 2500 --advice-collection-budget 5000 --C1 20000 --C2 20000 --advice-collection-method sample_efficency --cons-learning-epoch 20 --dqn-dueling --use-proportional-student-model-uc-th --proportional-student-model-uc-th-percentile 70\
#     # --advice-reuse-probability-decay --advice-reuse-probability-decay-begin 10000 --advice-reuse-probability-decay-end 50000 --advice-reuse-probability-final 0.1 --advice-imitation-method periodic --advice-reuse-method extended --autoset-teacher-model-uc-th > logs/DW_cfdaa4_$i.log 2>& 1 &
#     # CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dqn-rm-init 2500 --load-teacher --advice-collection-method random --dqn-eps-steps 50000 --save-models --env-key DW-v1 --dqn-n-hidden-layers 2 --n-training-frames 100000 --dqn-dueling --use-gpu --advice-collection-budget 5000  > logs/DW_random2_$i.log 2>& 1 &
#     # CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dqn-rm-init 2500 --load-teacher --advice-collection-method early --dqn-eps-steps 50000 --env-key DW-v1 --dqn-n-hidden-layers 2 --n-training-frames 100000 --dqn-dueling --use-gpu --advice-collection-budget 5000  > logs/DW_early2_$i.log 2>& 1 &
#     # CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dqn-rm-init 2500 --load-teacher --advice-collection-method advice_novelty --dqn-eps-steps 50000 --env-key DW-v1 --dqn-n-hidden-layers 2 --advice-collection-budget 5000 --n-training-frames 100000 --dqn-dueling --use-gpu > logs/DW_novelty2_$i.log 2>& 1 &
    
#     # SUAIR
#     # CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dqn-rm-init 2500 --load-teacher  --dqn-eps-steps 50000 --env-key DW-v1 --dqn-n-hidden-layers 2 --n-training-frames 100000 --use-gpu --advice-collection-budget 5000 --dqn-twin --advice-collection-method student_model_uc --dqn-dueling --use-proportional-student-model-uc-th --proportional-student-model-uc-th-percentile 70 --advice-imitation-method periodic --advice-reuse-method extended --autoset-teacher-model-uc-th > logs/DW_SUAIR2_$i.log 2>& 1 &
# done
