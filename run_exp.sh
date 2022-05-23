python data/cifar100/generate_data.py --n_tasks 100    --pachinko_allocation_split    --alpha 0.3    --beta 10    --s_frac 1.0    --tr_frac 0.8    --seed 12345
python3 run_experiment_on_off.py 'weights/cifar100/FedEM_0.3/'
python data/cifar100/generate_data.py --n_tasks 100    --pachinko_allocation_split    --alpha 0.4    --beta 10    --s_frac 1.0    --tr_frac 0.8    --seed 12345
python3 run_experiment_on_off.py 'weights/cifar100/FedEM_0.4/'
python data/cifar100/generate_data.py --n_tasks 100    --pachinko_allocation_split    --alpha 0.8    --beta 10    --s_frac 1.0    --tr_frac 0.8    --seed 12345
python3 run_experiment_on_off.py 'weights/cifar100/FedEM_0.8/'
python data/cifar100/generate_data.py --n_tasks 100    --pachinko_allocation_split    --alpha 2    --beta 10    --s_frac 1.0    --tr_frac 0.8    --seed 12345
python3 run_experiment_on_off.py 'weights/cifar100/FedEM_2/'
python data/cifar100/generate_data.py --n_tasks 100    --pachinko_allocation_split    --alpha 4    --beta 10    --s_frac 1.0    --tr_frac 0.8    --seed 12345
python3 run_experiment_on_off.py 'weights/cifar100/FedEM_4/'
