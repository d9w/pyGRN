from pygrn import config
from copy import deepcopy
import numpy as np
import random


def crossover(parent1, parent2):
    """Cross the parent GRNs to create a child GRN
    The child inherits input and output proteins randomly from both parents
    Regulatory genes are passed on based on alignment;
    the closest genes from each parent are selected between for inheritance.
    Beta and delta are selected randomly from both parents.
    """
    child = parent1.clone()
    for k in range(parent1.num_input + parent1.num_output):
        if np.random.randint(2) == 1:
            child.identifiers[k] = parent2.identifiers[k]
            child.inhibitors[k] = parent2.inhibitors[k]
            child.enhancers[k] = parent2.enhancers[k]

    child.identifiers = child.identifiers[:(child.num_input+child.num_output)]
    child.inhibitors = child.inhibitors[:(child.num_input+child.num_output)]
    child.enhancers = child.enhancers[:(child.num_input+child.num_output)]

    p1range = list(range(parent1.num_input + parent1.num_output,
                         parent1.size()))
    random.shuffle(p1range)
    p2range = list(range(parent2.num_input + parent2.num_output,
                         parent2.size()))
    random.shuffle(p2range)

    p1remaining = deepcopy(p1range)

    # Crossing regulatory
    p1_gene_count = 0
    p2_gene_count = 0
    for p1idx in p1range:
        min_dist = config.CROSSOVER_THRESHOLD
        paired_idx = None
        for p2idx in p2range:
            gdist = parent1.protein_distance(parent2, p1idx, p2idx)
            if gdist < min_dist:
                min_dist = gdist
                paired_idx = p2idx
        if paired_idx != None:
            if np.random.randint(2) == 0:
                chosen_parent = parent1
                chosen_idx = p1idx
                p1_gene_count += 1
            else:
                chosen_parent = parent2
                chosen_idx = p2idx
                p2_gene_count += 1
            child.identifiers = np.append(child.identifiers,
                                          chosen_parent.identifiers[chosen_idx])
            child.inhibitors = np.append(child.inhibitors,
                                         chosen_parent.inhibitors[chosen_idx])
            child.enhancers = np.append(child.enhancers,
                                        chosen_parent.enhancers[chosen_idx])
            # Remove from consideration again
            p2range = list(set(p2range) - set([p2idx]))
            p1remaining = list(set(p1remaining) - set([p1idx]))

    # Add remaining material
    if child.size() == (child.num_input + child.num_output):
        prob = 0.5
    else:
        prob = p1_gene_count / (p1_gene_count + p2_gene_count)

    chosen_parent = parent2
    chosen_range = p2range
    if np.random.random() < prob:
        chosen_parent = parent1
        chosen_range = p1remaining

    for idx in chosen_range:
        child.identifiers = np.append(child.identifiers,
                                      chosen_parent.identifiers[idx])
        child.inhibitors = np.append(child.inhibitors,
                                     chosen_parent.inhibitors[idx])
        child.enhancers = np.append(child.enhancers,
                                    chosen_parent.enhancers[idx])

    child.num_regulatory = child.size() - (child.num_input + child.num_output)

    # Cross dynamics
    if np.random.random() < 0.5:
        child.beta = parent1.beta
    else:
        child.beta = parent2.beta

    if np.random.random() < 0.5:
        child.delta = parent1.delta
    else:
        child.delta = parent2.delta

    return child
