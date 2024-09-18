import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from src.model.transformer import SELMTransformer
from src.tasks.text_classification import TextClassificationDataset
import argparse
import os

def setup(rank, world_size):
    """Initialize the process group for distributed computing."""
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f'Rank {rank}/{world_size} is initialized.')

def cleanup():
    """Clean up the process group after distributed computation is finished."""
    dist.destroy_process_group()

def inference(rank, world_size, config):
    """Distributed inference function for SELM model."""
    setup(rank, world_size)
    
    # Load dataset and set up distributed sampler
    dataset = TextClassificationDataset(config['data']['test_file'])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = DataLoader(dataset, batch_size=config['inference']['batch_size'], sampler=sampler)
    
    # Load the model and move to corresponding device
    model = SELMTransformer(config_path=config['model_config_path']).to(rank)
    model = DDP(model, device_ids=[rank])

    # Load model checkpoint (if provided)
    if config['inference']['checkpoint_path']:
        checkpoint = torch.load(config['inference']['checkpoint_path'], map_location=f'cuda:{rank}')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Rank {rank} loaded checkpoint from {config['inference']['checkpoint_path']}")

    model.eval()

    # Perform inference
    results = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch
            inputs = inputs.to(rank)
            outputs = model(inputs)
            results.append(outputs.cpu())
    
    # Gather results from all processes (optional)
    if config['inference']['gather_results']:
        results = gather_results(rank, world_size, results)
    
    cleanup()

def gather_results(rank, world_size, results):
    """Gather results from all ranks (optional)."""
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, results)
    
    # Flatten the results list
    all_results = []
    for result in gathered_results:
        all_results.extend(result)
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Distributed Inference for SELM Model')
    parser.add_argument('--world_size', type=int, help='Number of GPUs/nodes participating in distributed inference.')
    parser.add_argument('--rank', type=int, help='The rank of this process (provided by launcher).')
    parser.add_argument('--config', type=str, default='config/distributed_config.yaml', help='Path to the config file.')

    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Start distributed inference
    inference(args.rank, args.world_size, config)

if __name__ == '__main__':
    main()
