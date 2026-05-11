import time, csv

from CharTokenizer import CharTokenizer

from BasicLM_MPS import ModelWrapper as MPS_LM
from BasicLM_MLX import ModelWrapper as MLX_LM

config = {
        'batch_size': 32,
        'block_size': 256,
        'embedding_dim': 384,
        'max_iters': 3000,
        'learning_rate': 3e-4,
        'eval_interval': 500,
        'eval_iters': 200,
        'num_heads': 4,
        'num_layers': 6,
        'split_ratio': 0.9,
        'skip_loss_eval_during_training': True,
}

with open("input.txt", 'r', encoding="utf-8") as f:
    text = f.read()

config['tokenizer'] = CharTokenizer(text)

tokens_per_run = config['max_iters'] * config['batch_size'] * config['block_size']

results = []

for framework, ModelWrapper in {'pytorch': MPS_LM, 'mlx': MLX_LM}.items():
    for i, seed in enumerate([212,829,943,486]):
        print(f"Starting run {i} for {framework}\nApproximate Start Time is {time.asctime()}")
        start_time = time.perf_counter()

        model = ModelWrapper(config,text,rand_seed=seed,np_rand_seed=seed)
        model.train()

        end_time = time.perf_counter()
        print(f"End Time is {time.asctime()}")

        total_time = end_time-start_time

        final_loss = model.estimate_loss()

        results.append({
            'framework': framework,
            'seed': seed,
            'run_time': total_time,
            'tokens_per_sec': tokens_per_run / total_time,
            'final_train_loss': float(final_loss['train']),
            'final_val_loss': float(final_loss['val']),
        })

        with open('results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)