import torch
import copy

def crossover(parent1, parent2, crossover_rate=0.5):
    """
    Performs crossover between two parent models to generate an offspring model.

    Args:
        parent1 (torch.nn.Module): The first parent model.
        parent2 (torch.nn.Module): The second parent model.
        crossover_rate (float): Probability of performing crossover on each weight.

    Returns:
        torch.nn.Module: The offspring model created by combining parts of the parent models.
    """
    # Create a copy of the first parent model as the starting point
    offspring = copy.deepcopy(parent1)
    
    with torch.no_grad():
        for param1, param2 in zip(parent1.parameters(), parent2.parameters()):
            if param1.size() == param2.size():
                # Create a mask for crossover
                mask = torch.rand_like(param1) < crossover_rate
                
                # Apply crossover
                param1.copy_(torch.where(mask, param2, param1))
                
    return offspring
