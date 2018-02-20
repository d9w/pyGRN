from pygrn import GRNEAT, grns, problems
import argparse

parser = argparse.ArgumentParser(description='Evolve a GRN to classify Boston housing data')
parser.add_argument('--no-learn', dest='learn', action='store_const',
                    const=False, default=True,
                    help='Turn off learning')
parser.add_argument('--no-evo', dest='evo', action='store_const',
                    const=False, default=True,
                    help='Turn off evolution')
parser.add_argument('--log', type=str, help='Log file')
parser.add_argument('--problem', type=str, help='Problem', default='Boston')
parser.add_argument('--epochs', type=int, help='Number of epochs', default=1)
parser.add_argument('--grns', type=str, help='File of GRNs in string format', default='')
parser.add_argument('--generation', type=int, help='Generation (for logging)', default=0)
args = parser.parse_args()

p = problems.Boston(args.log, args.learn, args.epochs)
if args.problem == "AirQuality":
    p = problems.AirQuality(args.log, args.learn, args.epochs)

newgrn = lambda: grns.DiffGRN()
if args.evo:
    grneat = GRNEAT(newgrn, args.log)
    grneat.setup(p.nin, p.nout, 50)
    grneat.initializationDuplication = 10
    grneat.minSpeciesSize = 5
    grneat.run(50, p)
elif args.grns:
    with open(args.grns, 'r') as f:
        for g in f.readlines():
            grn = newgrn()
            grn.random(p.nin, p.nout, 1)
            grn.from_str(g)
            p.generation = args.generation
            p.eval(grn)
else:
    for i in range(20):
        grn = newgrn()
        grn.random(p.nin, p.nout, 1)
        p.generation = i
        p.eval(grn)
