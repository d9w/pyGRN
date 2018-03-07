from pygrn import config
import numpy as np


def mutate_add(parent):
    """Add a random protein to a clone of the parent and return it"""
    child = parent.clone()

    child.inhibitors = np.append(child.inhibitors, np.random.random())
    child.enhancers = np.append(child.enhancers, np.random.random())
    child.identifiers = np.append(child.identifiers, np.random.random())
    child.num_regulatory = child.size() - (child.num_input + child.num_output)

    return child


def mutate_remove(parent):
    """Delete a random regulatory protein from a clone of the parent.
    If the parent does not have any regulatory proteins, return None
    """
    if parent.size() > (parent.num_input + parent.num_output):
        child = parent.clone()

        to_remove = np.random.randint(child.num_input + child.num_output,
                                      child.size())

        child.inhibitors = np.delete(child.inhibitors, to_remove)
        child.enhancers = np.delete(child.enhancers, to_remove)
        child.identifiers = np.delete(child.identifiers, to_remove)
        child.num_regulatory = child.size() - (child.num_input +
                                               child.num_output)

        return child
    return None


def mutate_modify(parent):
    """Modify a single protein tag, beta, or delta.
    The modified variable is assigned a new value from a uniform distribution.
    """
    child = parent.clone()
    target = np.random.randint(child.size() + 2)

    if target < child.size():
        tag_type = np.random.randint(3)
        if tag_type == 0:
            child.identifiers[target] = np.random.random()
        elif tag_type == 1:
            child.enhancers[target] = np.random.random()
        else:
            child.inhibitors[target] = np.random.random()
    elif target == child.size():
        child.beta = (np.random.random()*(config.BETA_MAX - config.BETA_MIN)
                      + config.BETA_MIN)
    else:
        child.delta = (np.random.random()*(config.DELTA_MAX - config.DELTA_MIN)
                       + config.DELTA_MIN)

    return child


def mutate(parent):
    """Return a modified copy of the provided parent GRN.
    Performs one of three mutations (based on corresponding probabilities):
    - mutate_add: add a regulatory protein (add_rate)
    - mutate_remove: add a regulatory protein (del_rate)
    - mutate_modify: modify a protein, beta, or delta
    """
    child = None
    num_tries = 0
    while child == None and num_tries < config.MAX_SELECTION_TRIES:
        r = np.random.random()
        if r < config.MUTATION_ADD_RATE:
            child = mutate_add(parent)
        elif r < (config.MUTATION_ADD_RATE + config.MUTATION_DEL_RATE):
            child = mutate_remove(parent)
        else:
            child = mutate_modify(parent)
        num_tries += 1
    if num_tries == config.MAX_SELECTION_TRIES:
        return parent.clone()
    return child
