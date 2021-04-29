"""Helper functions to interact with a ST-based topic model."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import umap.plot
from bokeh.plotting import output_notebook


def most_similar_topics(model, method="tv"):
    """Compute most similar topics via cosine similarity of topic vectors."""
    if method == "tv":
        topic_vectors = model.topic_vectors
    elif method == "tf_idf":
        topic_vectors = model.topic_vectors_tfidf
    topic_sim_mat = cosine_similarity(topic_vectors, topic_vectors)
    topic_sim_mat -= np.diag(np.diag(topic_sim_mat))

    top_top_sims = []
    for top1 in np.unique(model.doc_topic):
        for top2 in np.unique(model.doc_topic):
            if top2 > top1:
                top_top_sims.append((top1, top2, topic_sim_mat[top1, top2]))

    top_top_sims_sorted = sorted(top_top_sims, key=lambda tup: tup[2],
                                 reverse=True)

    return top_top_sims_sorted


def get_topic_docs(model, topic, facet=None):
    """
    Retrieve documents of a topic/facet.

    The returned documents are sorted by similarity to the topic/facet vector.
    """
    if facet is None:
        # Retrieve topic's docs
        docs_idxs = np.where(model.doc_topic == topic)[0]
        sims = model.doc_topic_sim[docs_idxs]
    else:
        # Retrieve facet's docs
        docs_idxs = model.topic_facets[topic][facet]["doc_idxs"]
        sims = np.array([model.doc_topic_facet[idx]["facet_similarity"]
                         for idx in docs_idxs])
    docs = model.documents[docs_idxs]
    doc_ids = model.document_ids[docs_idxs]

    # Sort documents by decreasing similarity to topic vector
    idxs_sorted_sims = np.flip(np.argsort(sims))
    docs_sorted = docs[idxs_sorted_sims]
    sims_sorted = sims[idxs_sorted_sims]
    doc_ids_sorted = doc_ids[idxs_sorted_sims]

    return docs_sorted, sims_sorted, doc_ids_sorted


def plot_topics(model, topics_subset=None, mark_noisy=False,
                show_legend=True, umap_model_2d=None):
    """Plot extracted topics in 2D."""
    if umap_model_2d is None:
        topic_ext_para = model.topic_extraction_parameters
        umap_model_2d = model._compute_umap(model.document_vectors,
                                            n_components=2,
                                            n_neighbors=topic_ext_para["n_neighbors"],
                                            random_state=topic_ext_para["random_state"])
    labels = model.doc_topic
    if mark_noisy:
        idxs_noisy = np.where(model.doc_topic_noisy)
        labels[idxs_noisy] = -1
    if topics_subset is not None:
        labels = np.array([c if c in topics_subset else -99 for c in model.doc_topic])

    axs = umap.plot.points(umap_model_2d, labels=labels, background='black',
                           height=1200, width=1200, show_legend=show_legend)
    return axs


def plot_facets(model, topic, mark_noisy=False, interactive=True,
                show_legend=True):
    """Assess quality of intra-topic clustering of claims (facets)."""
    # Compute array of cluster labels
    docs_idxs = np.where(model.doc_topic == topic)[0]
    doc_facet = []
    for doc_idx in docs_idxs:
        facet = model.doc_topic_facet[doc_idx]["facet"]
        is_noisy = model.doc_topic_facet[doc_idx]["facet_noisy"]
        if mark_noisy and is_noisy:
            # Place points marked as noisy by HDBSCAN in a separate cluster
            doc_facet.append(-1)
        else:
            doc_facet.append(facet)
    doc_facet = np.array(doc_facet)
    doc_ids = model.document_ids[docs_idxs]
    docs = model.documents[docs_idxs]
    document_vectors = model.document_vectors[docs_idxs]

    # Plot 2D representation of intra-topic clustering
    facet_ext_para = model.facet_extraction_parameters
    umap_model = model._compute_umap(document_vectors,
                                     n_components=2,
                                     n_neighbors=facet_ext_para["n_neighbors"],
                                     random_state=facet_ext_para["random_state"])
    if interactive:
        hover_data = pd.DataFrame({'doc_id': doc_ids,
                                   'cluster': doc_facet,
                                   'doc': docs
                                   })
        plot = umap.plot.interactive(umap_model, labels=doc_facet,
                                     hover_data=hover_data,
                                     background='black',
                                     height=1200, width=1200,
                                     point_size=8)
        output_notebook()
        umap.plot.show(plot)
    else:
        axs = umap.plot.points(umap_model, labels=doc_facet, background='black',
                               height=1200, width=1200, show_legend=show_legend)
        return axs


def valid_sample(sim_pair, sim_samples, quantiles_sim, n_samples_per_quant):
    """"""
    if sim_pair > 0.99:
        # Exclude pairs of the same document
        return False
    # Test if sampling the pair keeps a balanced distribution
    sim_pair_digit = np.digitize(sim_pair, bins=quantiles_sim)
    sim_samples_digits = list(np.digitize(sim_samples, bins=quantiles_sim))
    digits_q = list(range(len(quantiles_sim)))
    samples_digits_counts = [sim_samples_digits.count(d) for d in digits_q]
    test = (samples_digits_counts[sim_pair_digit] <= n_samples_per_quant)
    return test


def sample_doc_pairs(model, pct_per_topic, quantiles_sim, low_memory=False):
    """
    Sample pairs of documents in each topic.

    The sampling is made so as balancing the pairs by similarity.
    """
    n_to_sample = (pct_per_topic * model.topic_sizes).round()
    digits_q = list(range(len(quantiles_sim)))

    dict_samples = {}
    for topic in range(len(model.topic_sizes)):
        doc_idxs = np.where(model.doc_topic == topic)[0]
        n_samples_per_quant = round(n_to_sample[topic] / len(quantiles_sim))
        t_samples = set()
        if low_memory:
            # Random sampling of doc pairs until reaching a balanced quantile
            # distribution. Can take a long time if some quantile classes are rare.
            t_sim_samples = []
            while len(t_samples) < n_to_sample[topic]:
                doc1_idx, doc2_idx = np.random.choice(doc_idxs, size=2,
                                                      replace=False)
                not_sampled = (doc1_idx, doc2_idx) not in t_samples
                not_sampled_dup = (doc2_idx, doc1_idx) not in t_samples
                if not_sampled and not_sampled_dup:
                    sim_pair = cosine_similarity(model.document_vectors[doc1_idx].reshape(1, -1),
                                                 model.document_vectors[doc2_idx].reshape(1, -1)
                                                 )[0][0]
                    if valid_sample(sim_pair, t_sim_samples, quantiles_sim,
                                    n_samples_per_quant):
                        t_samples.add((doc1_idx, doc2_idx))
                        t_sim_samples.append(sim_pair)
        else:
            # Sample directly from the similarity matrix
            # Much faster, but can cause OOM error if topic has a lot of documents
            t_sim_mat = cosine_similarity(model.document_vectors[doc_idxs],
                                          model.document_vectors[doc_idxs])
            t_sim_mat_digits = np.digitize(t_sim_mat, bins=quantiles_sim)
            for digit in digits_q:
                x_idxs, y_idxs = np.where(t_sim_mat_digits == digit)
                samples_idx = np.random.randint(low=0, high=len(x_idxs),
                                                size=n_samples_per_quant)
                for s_idx in samples_idx:
                    t_samples.add((x_idxs[s_idx], y_idxs[s_idx]))
                    # t_sim_samples.append(t_sim_mat[x[idx], y[idx]])

        dict_samples[topic] = list(t_samples)

    return dict_samples
