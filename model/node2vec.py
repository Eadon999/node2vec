from __future__ import print_function
from gensim.models import Word2Vec
from model import walker


class Node2vec(object):

    def __init__(self, graph, path_length, num_paths, dim, p=1.0, q=1.0, dw=False, **kwargs):
        self.graph = graph
        self.path_length = path_length
        self.num_paths = num_paths
        self.embedding_dim = dim
        self.embedding_vector = {}
        self.p = p
        self.q = q
        self.dw = dw
        self.kwargs = kwargs


    def fit_node2vec(self):
        sentences = self.node_walk_result(self.graph, self.path_length, self.num_paths, self.p, self.q, self.dw,
                                          **self.kwargs)
        print(sentences)
        self.kwargs["sentences"] = sentences
        self.kwargs["min_count"] = self.kwargs.get("min_count", 0)
        self.kwargs["size"] = self.kwargs.get("size", self.embedding_dim)
        self.kwargs["sg"] = 1

        self.size = self.kwargs["size"]
        print("Learning representation...")
        word2vec = Word2Vec(**self.kwargs)
        for word in self.graph.G.nodes():
            self.embedding_vector[word] = word2vec.wv[word]
        del word2vec

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.embedding_vector.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.embedding_vector.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()

    def node_walk_result(self, graph, path_length, num_paths, p, q, dw, **kwargs):
        kwargs["workers"] = kwargs.get("workers", 1)
        if dw:
            kwargs["hs"] = 1
            p = 1.0
            q = 1.0
        if dw:
            self.walker = walker.DeepWalker(graph, workers=kwargs["workers"])
        else:
            self.walker = walker.Node2VecWalker(graph, p=p, q=q, workers=kwargs["workers"])
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()
        sentences = self.walker.simulate_walks(num_walks=num_paths, walk_length=path_length)
        return sentences
