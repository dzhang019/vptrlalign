mkdir curriculum
temp=2 #Change this hyperparameter
kl=50.0 #this too

#Phase 1
python3 vptrlalign/fasttrainingpp2.py --in-model foundation-model-2x.model --in-weights rl-from-house-2x.weights --out-weights curriculum/phase1.weights --out-episodes log.txt --num-iterations 100 --rollout-steps 80 --num-envs 11 --queue-size 3 --temp $temp --lambda-kl $kl --reward lib.phase1

#Phase 2
python3 vptrlalign/fasttrainingpp2.py --in-model foundation-model-2x.model --in-weights curriculum/phase1.weights --out-weights curriculum/phase2.weights --out-episodes log.txt --num-iterations 100 --rollout-steps 80 --num-envs 11 --queue-size 3 --temp $temp --lambda-kl $kl --reward lib.phase2

#Phase 3
python3 vptrlalign/fasttrainingpp2.py --in-model foundation-model-2x.model --in-weights curriculum/phase2.weights --out-weights curriculum/phase3.weights --out-episodes log.txt --num-iterations 100 --rollout-steps 80 --num-envs 11 --queue-size 3 --temp $temp --lambda-kl $kl --reward lib.phase3

#Phase 4
python3 vptrlalign/fasttrainingpp2.py --in-model foundation-model-2x.model --in-weights curriculum/phase3.weights --out-weights curriculum/phase4.weights --out-episodes log.txt --num-iterations 100 --rollout-steps 80 --num-envs 11 --queue-size 3 --temp $temp --lambda-kl $kl --reward lib.phase4

#Phase 5
python3 vptrlalign/fasttrainingpp2.py --in-model foundation-model-2x.model --in-weights curriculum/phase4.weights --out-weights curriculum/phase5.weights --out-episodes log.txt --num-iterations 100 --rollout-steps 80 --num-envs 11 --queue-size 3 --temp $temp --lambda-kl $kl --reward lib.phase5
