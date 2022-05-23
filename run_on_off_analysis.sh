printf "alpha_0.3\n" >> out_on_off.txt
python data/cifar100/generate_data.py --n_tasks 100    --pachinko_allocation_split    --alpha 0.3    --beta 10    --s_frac 1.0    --tr_frac 0.8    --seed 12345
python cifar100_transfer_attack_analysis_on_off.py 'weights/cifar100/FedEM_0.3/' >> out_on_off.txt
printf "alpha_0.4\n" >> out_on_off.txt
python data/cifar100/generate_data.py --n_tasks 100    --pachinko_allocation_split    --alpha 0.4    --beta 10    --s_frac 1.0    --tr_frac 0.8    --seed 12345
python cifar100_transfer_attack_analysis_on_off.py 'weights/cifar100/FedEM_0.4/' >> out_on_off.txt
printf "alpha_0.8\n" >> out_on_off.txt
python data/cifar100/generate_data.py --n_tasks 100    --pachinko_allocation_split    --alpha 0.8    --beta 10    --s_frac 1.0    --tr_frac 0.8    --seed 12345
python cifar100_transfer_attack_analysis_on_off.py 'weights/cifar100/FedEM_0.8/' >> out_on_off.txt
printf "alpha_2\n" >> out_on_off.txt
python data/cifar100/generate_data.py --n_tasks 100    --pachinko_allocation_split    --alpha 2    --beta 10    --s_frac 1.0    --tr_frac 0.8    --seed 12345
python cifar100_transfer_attack_analysis_on_off.py 'weights/cifar100/FedEM_2/' >> out_on_off.txt
printf "alpha_4\n" >> out_on_off.txt
python data/cifar100/generate_data.py --n_tasks 100    --pachinko_allocation_split    --alpha 4    --beta 10    --s_frac 1.0    --tr_frac 0.8    --seed 12345
python cifar100_transfer_attack_analysis_on_off.py 'weights/cifar100/FedEM_4/'>> out_on_off.txt
