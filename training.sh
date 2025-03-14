if [ ! -f foundation-model-2x.model ]; then
  echo "Error: foundation-model-2x.model not found!"
  exit 1
fi
if [ ! -f rl-from-house-2x.weights ]; then
  echo "Error: rl-from-house-2x.weights not found!"
  exit 1
fi

folder="curriculum"
mkdir "$folder"
temp=2 #Change this hyperparameter
kl=50.0 #this too
iterations=100

#Phase 1
echo "Training Phase 1"
python3 vptrlalign/fasttrainingpp2.py --in-model foundation-model-2x.model --in-weights rl-from-house-2x.weights --out-weights "$folder/phase1.weights" --out-episodes log.txt --num-iterations $iterations --rollout-steps 80 --num-envs 11 --queue-size 3 --temp $temp --lambda-kl $kl --reward lib.phase1

#Phase 2
echo "Training Phase 2"
python3 vptrlalign/fasttrainingpp2.py --in-model foundation-model-2x.model --in-weights "$folder/phase1.weights" --out-weights "$folder/phase2.weights" --out-episodes log.txt --num-iterations $iterations --rollout-steps 80 --num-envs 11 --queue-size 3 --temp $temp --lambda-kl $kl --reward lib.phase2

#Phase 3
echo "Training Phase 3"
python3 vptrlalign/fasttrainingpp2.py --in-model foundation-model-2x.model --in-weights "$folder/phase2.weights" --out-weights "$folder/phase3.weights" --out-episodes log.txt --num-iterations $iterations --rollout-steps 80 --num-envs 11 --queue-size 3 --temp $temp --lambda-kl $kl --reward lib.phase3

#Phase 4
echo "Training Phase 4"
python3 vptrlalign/fasttrainingpp2.py --in-model foundation-model-2x.model --in-weights "$folder/phase3.weights" --out-weights "$folder/phase4.weights" --out-episodes log.txt --num-iterations $iterations --rollout-steps 80 --num-envs 11 --queue-size 3 --temp $temp --lambda-kl $kl --reward lib.phase4

#Phase 5
echo "Training Phase 5"
python3 vptrlalign/fasttrainingpp2.py --in-model foundation-model-2x.model --in-weights "$folder/phase4.weights" --out-weights "$folder/phase5.weights" --out-episodes log.txt --num-iterations $iterations --rollout-steps 80 --num-envs 11 --queue-size 3 --temp $temp --lambda-kl $kl --reward lib.phase5
