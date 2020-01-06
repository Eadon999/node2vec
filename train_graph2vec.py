from __future__ import print_function
import numpy as np
import random
from model.build_graph import GraphBuilder
from model.node2vec import Node2vec
from sklearn.linear_model import LogisticRegression
from model.classify_performance import read_node_label, Classifier
from utils.args_parser import ArgsParser
import time


def train_graph2vec(args):
    start_training_time = time.time()
    graph_builder = GraphBuilder()
    print("Reading...")

    if args.graph_format == 'adjlist':
        graph_builder.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        print("graph format:" + args.graph_format)
        graph_builder.read_edgelist(filename=args.input, weighted=args.weighted,
                                    directed=args.directed)
    if args.method == 'node2vec':
        model = Node2vec(graph=graph_builder, path_length=args.walk_length,
                         num_paths=args.number_walks, dim=args.representation_size,
                         workers=args.workers, p=args.p, q=args.q, window=args.window_size)
    elif args.method == 'deepWalk':
        model = Node2vec(graph=graph_builder, path_length=args.walk_length,
                         num_paths=args.number_walks, dim=args.representation_size,
                         workers=args.workers, window=args.window_size, dw=True)
    else:
        exit()
        print("method select")
    print("training finish! Cost time:{}".format(time.time() - start_training_time))


    print("Saving embeddings...")
    model.save_embeddings(args.output)
    print("Saving embeddings finish!")

    if args.label_file:
        print("Start training classify model to calculate performance")
        vectors = model.vectors
        X, Y = read_node_label(args.label_file)
        print("Training classifier using {:.2f}% nodes...".format(
            args.clf_ratio * 100))
        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        clf.split_train_evaluate(X, Y, args.clf_ratio, seed=0)
        print("Training classifier finish!")


if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    test_mode = True
    parser = ArgsParser()
    parsed_args = parser.parse_args(test_mode)
    train_graph2vec(parsed_args)
