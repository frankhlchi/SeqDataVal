import numpy as np
import torch
import argparse
import yaml
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from opendataval.dataloader import mix_labels
from opendataval.experiment import ExperimentMediator
from opendataval.dataval import (
    AME, DVRL, BetaShapley, DataBanzhaf, DataOob, DataShapley,
    InfluenceSubsample, LeaveOneOut, RandomEvaluator
)
from opendataval.dataval.margcontrib.sampler import MonteCarloSampler

from evaluators import (
    BipartiteMatchingEvaluator,
    DynamicProgrammingEvaluator,
)

from utils.data import set_seed, create_output_dir, remove_points_one_by_one
from utils.opendataval_compat import patch_opendataval_openml

def run_evaluator(evaluator, exper_med):
    """运行单个评估器的函数"""
    print(f"Starting evaluator: {evaluator.__class__.__name__}")
    try:
        evaluator.train(
            exper_med.fetcher,
            exper_med.pred_model,
            exper_med.metric,
            **exper_med.train_kwargs
        )
        print(f"Completed: {evaluator.__class__.__name__}")
        return evaluator
    except Exception as e:
        print(f"Error in evaluator {evaluator.__class__.__name__}: {str(e)}")
        return None

def threaded_compute_values(exper_med, data_evaluators, num_threads=8):
    """使用线程池并行计算评估器的数据值"""
    print(f"Starting parallel computation with {num_threads} threads")
    completed_evaluators = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for evaluator in data_evaluators:
            future = executor.submit(run_evaluator, evaluator, exper_med)
            futures.append(future)
            
        for future in futures:
            try:
                evaluator = future.result()
                if evaluator is not None:
                    completed_evaluators.append(evaluator)
            except Exception as e:
                print(f"Error in evaluator execution: {str(e)}")
                
    return completed_evaluators

def parse_args():
    parser = argparse.ArgumentParser(description='Run data valuation experiments')
    
    parser.add_argument('--dataset', type=str, default='fried',
                       help='Dataset name (default: fried)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to a YAML config (overrides default config selection)')
    parser.add_argument('--train_count', type=int, default=None,
                       help='Number of training samples')
    parser.add_argument('--valid_count', type=int, default=None,
                       help='Number of validation samples') 
    parser.add_argument('--test_count', type=int, default=None,
                       help='Number of test samples')
    parser.add_argument('--seed', type=int, required=True,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run on')
    parser.add_argument('--threads', type=int, default=15,
                       help='Number of threads')
    parser.add_argument('--include_dp', action='store_true',
                       help='Include DynamicProgramming (RQ1 optimal baseline)')
    parser.add_argument('--dp_max_subset_size', type=int, default=None,
                       help='DP max subset size (default: train_count)')
    parser.add_argument('--skip_bipartite', action='store_true',
                       help='Skip Bipartite evaluator (useful for RQ1 settings)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Make OpenDataVal compatible with newer OpenML metadata.
    patch_opendataval_openml()
    
    # 加载配置
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    dataset_aliases = {
        "bbc-embedding": "bbc-embeddings",
        "bbc-embed": "bbc-embeddings",
        "bbc_embed": "bbc-embeddings",
        "miniboone": "MiniBooNE",
    }
    dataset_name = dataset_aliases.get(args.dataset, args.dataset)

    if args.config is not None:
        config_path = Path(args.config)
    else:
        config_path = PROJECT_ROOT / "config" / "base_config.yaml"
        if dataset_name == "cifar10-embeddings":
            config_path = PROJECT_ROOT / "config" / "cifar_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print('config', config['experiment']['train_count'])
    print('args.train_count', args.train_count)
    
    # 命令行参数覆盖配置
    train_count = args.train_count or config['experiment']['train_count']
    valid_count = args.valid_count or config['experiment']['valid_count'] 
    test_count = args.test_count or config['experiment']['test_count']
    device = args.device or config['experiment']['device']
    model = config['model']['name']
    
    noise_kwargs = {'noise_rate': 0.0}
    
    set_seed(args.seed)
    print(f"Running experiment with seed={args.seed} on {device}")

    # 创建实验
    exper_med = ExperimentMediator.model_factory_setup(
        dataset_name=dataset_name,
        model_name=model,
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
        add_noise=mix_labels,
        noise_kwargs=noise_kwargs,
        metric_name="accuracy", 
        device=device
    )
    
    # Monte Carlo sampler设置
    mc_sampler = MonteCarloSampler(
        mc_epochs=math.ceil(1000/train_count),
        min_cardinality=1,
        cache_name=f"cached_{dataset_name}_seed_{args.seed}",
        random_state=args.seed,
    )
    
    # 评估器列表
    data_evaluators = [
        RandomEvaluator(),
        LeaveOneOut(),
        InfluenceSubsample(num_models=1000),
        DataShapley(sampler=mc_sampler, cache_name=f"cached_{dataset_name}_seed_{args.seed}"),
        BetaShapley(sampler=mc_sampler, cache_name=f"cached_{dataset_name}_seed_{args.seed}"),
        DataBanzhaf(num_models=1000),
        AME(num_models=math.ceil(1000/4)),
        DVRL(rl_epochs=math.ceil(1000/32)),
        DataOob(num_models=1000),
    ]

    if not args.skip_bipartite:
        data_evaluators.append(BipartiteMatchingEvaluator(n_samples=1000, random_state=args.seed))

    if args.include_dp:
        dp_max_subset_size = args.dp_max_subset_size or train_count
        data_evaluators.append(
            DynamicProgrammingEvaluator(max_subset_size=dp_max_subset_size, random_state=args.seed)
        )

    print(f"Computing data values for seed {args.seed}...")
    # 使用线程池并行计算
    num_threads = min(len(data_evaluators), args.threads)
    completed_evaluators = threaded_compute_values(exper_med, data_evaluators, num_threads)
    exper_med.data_evaluators = completed_evaluators
    # 添加这一行来设置 num_data_eval
    exper_med.num_data_eval = len(completed_evaluators)  # 添加这行
    
    # 设置输出目录
    output_dir = Path("results") / args.dataset / f"seed_{args.seed}"
    create_output_dir(output_dir)
    exper_med.set_output_directory(output_dir)
    
    # 运行移除实验
    print("Running removal experiment...")
    df_removal, _ = exper_med.plot(remove_points_one_by_one)
    df_removal['axis'] = (df_removal['axis'] * train_count).astype(int)
    df_removal.to_csv(output_dir / "addition_experiment_results.csv")
    
    print(f"Results saved to {output_dir}")
