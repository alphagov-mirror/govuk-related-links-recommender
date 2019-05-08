import json
import os
import logging.config

import networkx as nx
from node2vec import Node2Vec
import pandas as pd


logging.config.fileConfig('src/logging.conf')


def train_node2_vec_model(edges_df, node_id_content_id_mapping,
                          word2vec_workers=3):
    train_node2_vec_model_logger = logging.getLogger('train_node2_vec_model.train_node2_vec_model')

    train_node2_vec_model_logger.info('creating graph from edges_df')
    graph = nx.convert_matrix.from_pandas_edgelist(
        edges_df, source='source', target='target'
        # when we have functional network too, we can add weight
        # , edge_attr='weight'
    )

    train_node2_vec_model_logger.info('adding attributes to graph')
    attributes = {node_id: {"content_id": node_id_content_id_mapping[node_id]}
                  for node_id in graph.nodes()}
    nx.set_node_attributes(graph, attributes)

    train_node2_vec_model_logger.info('Precomputing probabilities and generating walks')
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=300,
                        workers=1)

    train_node2_vec_model_logger.info('Fit node2vec model (create embeddings for nodes)')
    # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are
    # automatically passed (from the Node2Vec constructor)
    # https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
    model = node2vec.fit(window=10, min_count=1, batch_words=4, seed=1,
                         workers=word2vec_workers)
    return model


if __name__ == "__main__":  # our module is being executed as a program
    datadir = os.getenv("DATADIR")
    logger = logging.getLogger('train_node2_vec_model')

    edges = pd.read_csv(os.path.join(
        datadir, 'tmp', 'structural_network.csv'))
    with open(
            os.path.join(datadir, 'tmp', 'node_id_content_id_mapping.json'),
            'r') as node_id_content_id_mapping_file:
        node_id_content_id_mapping_dict = dict(
            (int(k), v) for k, v in json.load(
                node_id_content_id_mapping_file).items())

    node2vec_model = train_node2_vec_model(edges,
                                           node_id_content_id_mapping_dict)

    # where do we want to save models? in tmp?
    save_models_dir = os.path.join(datadir, "tmp")

    node_embeddings_file_path = os.path.join(save_models_dir,
                                             "n2v_node_embeddings")
    logger.info(f'saving node embeddings to {node_embeddings_file_path}')
    node2vec_model.wv.save_word2vec_format(node_embeddings_file_path)

    node2vec_model_file_path = os.path.join(save_models_dir, "n2v.model")
    logger.info(f'saving model to {node2vec_model_file_path}')
    node2vec_model.save(node2vec_model_file_path)
    # should we test saving and loading models and embeddings?
