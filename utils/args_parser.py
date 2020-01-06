from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class ArgsParser:
    """parse input args"""
    def parse_args(self, test_mode=False):
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                                conflict_handler='resolve')
        if test_mode:
            parser.add_argument('--input', default='./data/wiki/Wiki_edgelist.txt', type=str, help='Input graph file')
            parser.add_argument('--output', default='./output/embedded_vec.txt',
                                help='Output representation file')
            parser.add_argument('--label-file', default='./data/wiki/wiki_labels.txt',
                                help='The file of node label')
            parser.add_argument('--graph-format', default='edgelist', choices=['adjlist', 'edgelist'],
                                help='Input graph format')
        else:
            parser.add_argument('--input', required=True, type=str, help='Input graph file')
            parser.add_argument('--output', default='./output/embedded_vec.txt',
                                help='Output representation file')
            parser.add_argument('--label-file', default='',
                                help='The file of node label')
            parser.add_argument('--graph-format', required=True, choices=['adjlist', 'edgelist'],
                                help='Input graph format')
        parser.add_argument('--number-walks', default=10, type=int,
                            help='Number of random walks to start at each node')
        parser.add_argument('--directed', action='store_true',
                            help='Treat graph as directed.')
        parser.add_argument('--walk-length', default=80, type=int,
                            help='Length of the random walk started at each node')
        parser.add_argument('--workers', default=8, type=int,
                            help='Number of parallel processes.')
        parser.add_argument('--representation-size', default=128, type=int,
                            help='Number of latent dimensions to learn for each node.')
        parser.add_argument('--window-size', default=10, type=int,
                            help='Window size of skipgram model.')
        parser.add_argument('--p', default=1.0, type=float)
        parser.add_argument('--q', default=1.0, type=float)
        parser.add_argument('--method', default='node2vec', choices=['node2vec', 'deepWalk'],
                            help='The learning method')
        parser.add_argument('--feature-file', default='',
                            help='The file of node features')
        parser.add_argument('--weighted', action='store_true',
                            help='Treat graph as weighted')
        parser.add_argument('--clf-ratio', default=0.5, type=float,
                            help='The ratio of training data in the classification')
        parser.add_argument('--dropout', default=0.5, type=float,
                            help='Dropout rate (1 - keep probability)')
        parser.add_argument('--weight-decay', type=float, default=5e-4,
                            help='Weight for L2 loss on embedding matrix')
        parser.add_argument('--hidden', default=16, type=int,
                            help='Number of units in hidden layer 1')
        parser.add_argument('--kstep', default=4, type=int,
                            help='Use k-step transition probability matrix')
        parser.add_argument('--lr', default=0.01, type=float,
                            help='learning rate')
        parser.add_argument('--encoder-list', default='[1000, 128]', type=str,
                            help='a list of numbers of the neuron at each encoder layer, the last number is the '
                                 'dimension of the output node representation')
        args = parser.parse_args()

        if not args.output:
            print("No output filename. Exit.")
            exit(1)

        return args
