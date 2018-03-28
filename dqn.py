from pygrn import grns, problems, evolution
import os
import argparse

parser = argparse.ArgumentParser(
    description='Evolve a GRN as a layer in a DQN model for solving tasks')
parser.add_argument('--no-learn', dest='learn', action='store_const',
                    const=False, default=True,
                    help='Turn off learning')
parser.add_argument('--no-evo', dest='evo', action='store_const',
                    const=False, default=True,
                    help='Turn off evolution')
parser.add_argument('--id', type=str, help='Run id for logging')
parser.add_argument('--problem', type=str, help='Problem', default='Gym')
parser.add_argument('--env', type=str, help='Gym environment', default='CartPole-v0')
parser.add_argument('--steps', type=int, help='Number of training steps', default=10000)
parser.add_argument('--warmup', type=int, help='Number of warmup steps', default=200)
parser.add_argument('--gens', type=int, help='Number of generations', default=50)
parser.add_argument('--grn_dir', type=str, help='Directory for storing GRNS',
                    default='grns')
parser.add_argument('--log_dir', type=str, help='Log directory', default='logs')

args = parser.parse_args()

log_file = os.path.join(args.log_dir, 'fits_' + args.id + '.log')
p = problems.Gym(log_file, args.learn, args.env, args.steps, args.warmup)
newgrn = lambda: grns.DiffGRN()

if args.evo:
    grneat = evolution.Evolution(p, newgrn, args.id, grn_dir=args.grn_dir,
                                 log_dir=args.log_dir)
    grneat.run(args.gens)
else:
    for i in range(20):
        grn = newgrn()
        grn.random(p.nin, p.nout, 1)
        p.generation = i
        p.eval(grn)
