import pandas as pd


def generate_vectors(vector_list):
    for nid, prob in vector_list:
        yield nid, prob


def compute_top_n_for_content_id_list(df_path, n, content_id_node_id_mapping):
    pages_links = []
    missing = []
    for page in df_path.values:
        try:

            pages_links.append(
                compute_top_n_for_node_id(node_id, n))
        except KeyError:
            missing.append(page)
    #             print("Page {} is missing from training set".format(page))

    return pd.DataFrame(pages_links), missing


def compute_top_n_for_node_id(source_content_id, n, model, content_id_node_id_mapping,
                              node_id_content_id_mapping, content_id_to_base_path_mapping):
    node_id = content_id_node_id_mapping[source_content_id]
    source_node_id = str(node_id)
    count = 0
    list_link_content_ids = []
    most_similar_vectors = generate_vectors(
        model.wv.most_similar(source_node_id, topn=1000))
    while count <= n:

        link_node_id, prob = next(most_similar_vectors)
        link_content_id = node_id_content_id_mapping[int(link_node_id)]
        link_base_path = content_id_to_base_path_mapping[link_content_id]

        # the node IDS shouldn't be the same, so the content IDs shouldn't be the same
        if all([
            source_content_id != link_content_id,
            # check topic and browse aren't in URL
            all(t not in link_base_path for t in ["/topic", "/browse"]),
            # check content ID isn't in recommended link content IDs already
            link_content_id not in list_link_content_ids,
            # check link isn't in existing embedded links
            link_content_id not in content_id_embedded_link_content_ids_mapping[source_content_id]
        ]):
            list_link_content_ids.append(link_content_id)
            page_link = {"source_node_id": int(source_node_id),
                         "source_content_id": source_content_id,
                         "source_base_path": content_id_to_base_path_mapping[source_content_id],
                         "link_base_path": link_base_path,
                         "link_content_id": link_content_id,
                         "probability": round(prob, 3)}
            pages_links.append(page_link)
            count += 1