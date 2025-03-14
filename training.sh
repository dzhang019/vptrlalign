mkdir curriculum

#Phase 1
python3 vptrlalign/fasttrainingpp2.py --in-model 2x.model --in-weights rl-from-house-2x.weights --out-weights curriculum/phase1.weights --out-episodes log.txt --num-iterations 100 --rollout-steps 80 --num-envs 11 --queue-size 3 --temp 2 --lambda-kl 50 --reward lib.phase1

#Phase 2
python3 vptrlalign/fasttrainingpp2.py --in-model 2x.model --in-weights curriculum/phase1.weights --out-weights curriculum/phase2.weights --out-episodes log.txt --num-iterations 100 --rollout-steps 80 --num-envs 11 --queue-size 3 --temp 2 --lambda-kl 50 --reward lib.phase2

#Phase 3
python3 vptrlalign/fasttrainingpp2.py --in-model 2x.model --in-weights curriculum/phase2.weights --out-weights curriculum/phase3.weights --out-episodes log.txt --num-iterations 100 --rollout-steps 80 --num-envs 11 --queue-size 3 --temp 2 --lambda-kl 50 --reward lib.phase3

#Phase 4
python3 vptrlalign/fasttrainingpp2.py --in-model 2x.model --in-weights curriculum/phase3.weights --out-weights curriculum/phase4.weights --out-episodes log.txt --num-iterations 100 --rollout-steps 80 --num-envs 11 --queue-size 3 --temp 2 --lambda-kl 50 --reward lib.phase4

#Phase 5
python3 vptrlalign/fasttrainingpp2.py --in-model 2x.model --in-weights curriculum/phase4.weights --out-weights curriculum/phase2.weights --out-episodes log.txt --num-iterations 100 --rollout-steps 80 --num-envs 11 --queue-size 3 --temp 2 --lambda-kl 50 --reward lib.phase5
