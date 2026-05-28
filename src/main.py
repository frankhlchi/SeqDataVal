import numpy as np
import torch
import argparse
import yaml
import math
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

from utils.data import set_seed, create_output_dir
from utils.data import remove_points_one_by_one 
from utils.opendataval_compat import patch_opendataval_openml

PROJECT_ROOT = Path(__file__).parent.parent.absolute()


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
    parser.add_argument('--model', type=str, default=None,
                        help='Model name')
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="Base output directory (default: results). Use e.g. results_rq1_dp to avoid overwriting RQ3 outputs.",
    )
    
    parser.add_argument('--include_dp', action='store_true',
                        help='Include DynamicProgramming (RQ1 optimal baseline)')
    parser.add_argument('--dp_max_subset_size', type=int, default=None,
                        help='DP max subset size (default: train_count)')
    parser.add_argument('--skip_bipartite', action='store_true',
                        help='Skip Bipartite evaluator (useful for RQ1 settings)')
    parser.add_argument('--valuation_budget', type=int, default=1000,
                        help='Number of Monte Carlo/model samples for stochastic data-value evaluators')

    parser.add_argument('--noise_rate', type=float, default=None,
                        help='Noise rate')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Make OpenDataVal compatible with newer OpenML metadata.
    patch_opendataval_openml()
    
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

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Please make sure the config file exists."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
            
    print(f"Using config: {config_path}")

    train_count = args.train_count or config['experiment']['train_count']
    valid_count = args.valid_count or config['experiment']['valid_count']
    test_count = args.test_count or config['experiment']['test_count']
    device = args.device or config['experiment']['device']
    model = args.model or config['model']['name']
    
    noise_rate = args.noise_rate if args.noise_rate is not None else config['noise']['kwargs']['noise_rate']
    noise_kwargs = {'noise_rate': noise_rate}

    
    set_seed(args.seed)
    print(f"Running experiment with seed={args.seed} on {device}")
    
    exper_med = ExperimentMediator.model_factory_setup(
        dataset_name=dataset_name,
        model_name=model,
        train_count=train_count,
        valid_count=valid_count, 
        test_count=test_count,
        add_noise=mix_labels,
        noise_kwargs=noise_kwargs,
        metric_name="accuracy",
        device=device,
        random_state=args.seed,
    )
    
    valuation_budget = max(1, int(args.valuation_budget))

    mc_sampler = MonteCarloSampler(
        mc_epochs=math.ceil(valuation_budget / train_count),
        min_cardinality=1,
        cache_name=f"cached_{dataset_name}_seed_{args.seed}",
        random_state=args.seed,
    )
    
    data_evaluators = [
        RandomEvaluator(),
        LeaveOneOut(),
        InfluenceSubsample(num_models=valuation_budget),
        DataShapley(sampler=mc_sampler, cache_name=f"cached_{dataset_name}_seed_{args.seed}"),
        BetaShapley(sampler=mc_sampler, cache_name=f"cached_{dataset_name}_seed_{args.seed}"),
        DataBanzhaf(num_models=valuation_budget),
        AME(num_models=math.ceil(valuation_budget / 4)),
        DVRL(rl_epochs=valuation_budget),
        DataOob(num_models=valuation_budget),
    ]

    if not args.skip_bipartite:
        data_evaluators.append(BipartiteMatchingEvaluator(n_samples=valuation_budget, random_state=args.seed))

    if args.include_dp:
        dp_max_subset_size = args.dp_max_subset_size or train_count
        data_evaluators.append(
            DynamicProgrammingEvaluator(max_subset_size=dp_max_subset_size, random_state=args.seed)
        )
    

    print(f"Computing data values for seed {args.seed}...")
    exper_med = exper_med.compute_data_values(data_evaluators)
    
    output_dir = Path(args.results_root) / dataset_name / f"seed_{args.seed}"
    create_output_dir(output_dir)
    exper_med.set_output_directory(output_dir)
    
    print("Running removal experiment...")
    df_removal, _ = exper_med.plot(remove_points_one_by_one)
    df_removal['axis'] = (df_removal['axis'] * train_count).astype(int)
    df_removal.to_csv(output_dir / "addition_experiment_results.csv")
    
    print(f"Results saved to {output_dir}")
    
