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

for env in Seaquest;
do
    # rcmp
    # CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --env-key ALE-$env --use-gpu --save-models --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method rcmp --n_heads 5 --student-model-uc-th 0.011 > logs/$env\_rcmp2.log 2>& 1 &
    # AIR
    # CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --env-key ALE-$env --use-gpu --dqn-twin --save-models --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method teacher_model_uc --advice-imitation-method periodic --advice-reuse-method extended --autoset-teacher-model-uc-th > logs/$env\_AIR2.log 2>& 1 &
    # SUAIR
    # CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --env-key ALE-$env --use-gpu --dqn-twin --save-models --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method student_model_uc --use-proportional-student-model-uc-th --advice-imitation-method periodic --advice-reuse-method extended --autoset-teacher-model-uc-th > logs/$env\_SUAIR4.log 2>& 1 &
    # sample effciency
    CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --env-key ALE-$env --use-gpu --load-teacher --advice-collection-budget 25000 --advice-collection-method sample_efficency --cons-learning-epoch 100 --dqn-dueling > logs/$env\_sm_100_2_same5.log 2>& 1 &
done

# python -u main.py --env-key ALE-Seaquest --use-gpu --dqn-twin --save-models --dqn-dueling --load-teacher --advice-collection-budget 25000 --advice-collection-method student_model_uc --use-proportional-student-model-uc-th --advice-imitation-method periodic --advice-reuse-method extended