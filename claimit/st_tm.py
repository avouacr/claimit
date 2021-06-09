"""Main class of topic2facet."""
import logging

import numpy as np
import umap
import hdbscan
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


logger = logging.getLogger('STTopicModel')
logger.setLevel(logging.WARNING)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


class STTopicModel:
    """Extract topics and facets from a corpus using sentence transformers."""
    def __init__(self,
                 embedding_model,
                 documents,
                 document_ids=None,
                 token_pattern=r"(?u)\b\w\w+\b",  # Default sklearn pattern
                 verbose=False
                 ):
        # General parameters
        # TODO: logging
        if verbose:
            logger.setLevel(logging.DEBUG)
            self.verbose = True
        else:
            logger.setLevel(logging.WARNING)
            self.verbose = False

        # Store documents
        self.documents = np.array(documents, dtype="object")
        if document_ids is not None:
            self.document_ids = np.array(document_ids)
        else:
            self.document_ids = np.array(range(0, len(documents)))

        # Preprocess vocabulary
        self.token_pattern = token_pattern
        self.vocab = self._preprocess_voc(self.documents, self.token_pattern)

        # Initialize embedding variables
        self.embedding_model = embedding_model
        self.document_vectors = None

        # Initialize topic/facet extraction variables
        self.topic_facets = {}
        self.doc_topic_facet = {}
        self.topic_extraction_parameters = {}
        self.facet_extraction_parameters = {}

    def save(self, file):
        """Save model to file."""
        joblib.dump(self, file)

    @staticmethod
    def load(file):
        """Load model from file."""
        model = joblib.load(file)
        return model

    def get_attr(self, attr, topic=None, facet=None):
        """Query topic/facet attributes."""
        if topic == "all" and facet is None:
            res = np.array([self.topic_facets[t][attr] for t in self.topic_facets])
        elif topic in self.topic_facets and facet is None:
            res = self.topic_facets[topic][attr]
        elif topic in self.topic_facets and facet == "all":
            res = np.array([self.topic_facets[topic]["facets"][f][attr]
                            for f in self.topic_facets[topic]["facets"]])
        elif topic in self.topic_facets and facet in self.topic_facets[topic]["facets"]:
            res = self.topic_facets[topic]["facets"][facet][attr]
        else:
            raise ValueError("Invalid combination of parameters provided.")
        return res

    def embed_corpus(self, pooling_method="mean", device=None, batch_size=32):
        """Compute document embeddings using a transfomer model."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load embedding model
        tokenizer, model = self._load_embedding_model(self.embedding_model)
        model.to(device)

        # Load corpus in a Torch dataloader
        dataset = list(self.documents)
        dataloader = DataLoader(dataset, batch_size)

        # Embed corpus by batches
        batches_embedding = []
        for __, batch in tqdm(enumerate(dataloader)):
            # Tokenize documents
            encoded_input = tokenizer(batch, padding=True, truncation=True,
                                      return_tensors='pt')
            encoded_input = encoded_input.to(device)
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Perform pooling to get document embeddings
            token_embeddings = model_output[0]
            input_mask_expanded = (encoded_input['attention_mask'].unsqueeze(-1)
                                   .expand(token_embeddings.size()).float())
            token_embeddings = token_embeddings.cpu().detach().numpy()
            input_mask_expanded = input_mask_expanded.cpu().detach().numpy()
            if pooling_method == "mean":
                # Take attention mask into account for correct averaging
                sum_embeddings = np.sum(token_embeddings * input_mask_expanded,
                                        axis=1)
                sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9,
                                   a_max=None)
                batch_emb = sum_embeddings / sum_mask
            elif pooling_method == "max":
                batch_emb = np.max(token_embeddings, axis=1)
            else:
                raise ValueError("Invalid pooling method provided.")

            batches_embedding.append(batch_emb)

        document_vectors = np.vstack(batches_embedding)
        self.document_vectors = document_vectors

        return document_vectors

    def topic_extraction(self, n_components, n_neighbors, min_size,
                         min_samples=None, n_words=30, random_state=None):
        """Extract and characterize high-level topics from the corpus."""
        if self.document_vectors is None:
            raise ValueError("Corpus embeddings must be computed prior to topic extraction.")

        # Dimension reduction
        umap_model = self._compute_umap(self.document_vectors,
                                        n_components,
                                        n_neighbors,
                                        random_state)
        document_vectors_ld = umap_model.embedding_

        # Topic clustering
        hdbscan_model = self._compute_hdbscan(document_vectors_ld,
                                              min_size,
                                              min_samples,
                                              cluster_selection_method='leaf')
        clusters = hdbscan_model.labels_
        doc_topic_noisy = (clusters == -1)

        # Compute topic vectors
        topic_vectors = self._compute_topic_vectors(self.document_vectors,
                                                    hdbscan_model)

        # Assign noisy documents to topics
        doc_topic, doc_topic_sim = self._assign_noisy_docs(self.document_vectors,
                                                           topic_vectors,
                                                           clusters)

        # Reorder topics by decreasing size
        doc_topic, topic_vectors = self._reorder_topics(doc_topic, topic_vectors)

        # Characterize topics by top N words with highest average TF-IDF
        # among the topic's documents subset
        topic_words, __ = self._topic_characterization(doc_topic,
                                                       subset_docs=None,
                                                       n_words=n_words)

        # Store general topic information
        for topic, t_vector in enumerate(topic_vectors):
            doc_idxs = np.where(doc_topic == topic)[0]
            # TODO: replace idxs by doc ids
            self.topic_facets[topic] = {
                "doc_idxs": doc_idxs,
                "size": len(doc_idxs),
                "vector": t_vector,
                "words": topic_words[topic],
                "facets": {}
            }

        # Store topic information for each document
        for idx, __ in enumerate(doc_topic):
            self.doc_topic_facet[idx] = {"topic": doc_topic[idx],
                                         "topic_similarity": doc_topic_sim[idx],
                                         "topic_noisy": doc_topic_noisy[idx]
                                         }

        # Store parameters used for topic extraction
        self.topic_extraction_parameters = {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_topic_size": min_size,
            "min_samples": min_samples,
            "n_words": n_words,
            "random_state": random_state
        }

        # Clean previously stored facet extraction parameters
        self.facet_extraction_parameters = None

        return self.doc_topic_facet, self.topic_facets

    def facet_extraction(self, n_components, n_neighbors, min_size,
                         min_samples=None, n_words=30, random_state=None):
        """Extract and characterize facets (sub-topics) of each topic."""
        if self.doc_topic_facet == {}:
            raise ValueError("Topic extraction must be performed prior to facet extraction.")

        for topic in self.topic_facets:

            # Get topic's document vectors
            doc_idxs = [idx for idx in self.doc_topic_facet
                        if self.doc_topic_facet[idx]["topic"] == topic]
            document_vectors = self.document_vectors[doc_idxs]
            idxs_to_sub_idxs = {idx: i for i, idx in enumerate(doc_idxs)}
            sub_idxs_to_idxs = {i: idx for idx, i in idxs_to_sub_idxs.items()}

            # Dimension reduction
            umap_model = self._compute_umap(document_vectors,
                                            n_components,
                                            n_neighbors,
                                            random_state)
            document_vectors_ld = umap_model.embedding_

            # Facet clustering
            hdbscan_model = self._compute_hdbscan(document_vectors_ld,
                                                  min_size,
                                                  min_samples,
                                                  cluster_selection_method='leaf')
            clusters = hdbscan_model.labels_
            doc_facet_noisy = (clusters == -1)

            # Compute facet vectors
            facet_vectors = self._compute_topic_vectors(document_vectors,
                                                        hdbscan_model)

            # Assign noisy documents to facets
            doc_facet, doc_facet_sim = self._assign_noisy_docs(document_vectors,
                                                               facet_vectors,
                                                               clusters)

            # Reorder topics by decreasing size
            doc_facet, facet_vectors = self._reorder_topics(doc_facet, facet_vectors)
            __, facet_sizes = np.unique(doc_facet, return_counts=True)

            # Characterize facets by top N words with highest average TF-IDF
            # among the facet's documents subset
            facet_words, __ = self._topic_characterization(doc_topic=doc_facet,
                                                           subset_docs=doc_idxs,
                                                           n_words=n_words)

            # Store facets information for each topic
            for facet, size in enumerate(facet_sizes):
                facet_doc_idxs = np.array([sub_idxs_to_idxs[i]
                                           for i in np.where(doc_facet == facet)[0]])
                # TODO: replace idxs by doc ids
                self.topic_facets[topic]["facets"][facet] = {"doc_idxs": facet_doc_idxs,
                                                             "size": size,
                                                             "vector": facet_vectors[facet],
                                                             "words": facet_words[facet]
                                                             }

            # Store facet information for each document
            for idx in doc_idxs:
                idx_sub = idxs_to_sub_idxs[idx]
                self.doc_topic_facet[idx]["facet"] = doc_facet[idx_sub]
                self.doc_topic_facet[idx]["facet_similarity"] = doc_facet_sim[idx_sub]
                self.doc_topic_facet[idx]["facet_noisy"] = doc_facet_noisy[idx_sub]

        # Store parameters used for topic extraction
        self.facet_extraction_parameters = {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_facet_size": min_size,
            "min_samples": min_samples,
            "n_words": n_words,
            "random_state": random_state
        }

        return self.doc_topic_facet, self.topic_facets

    @staticmethod
    def _preprocess_voc(documents, token_pattern):
        """Extract individuals tokens from the corpus."""
        vectorizer = TfidfVectorizer(strip_accents="ascii",
                                     lowercase=True,
                                     token_pattern=token_pattern)
        vectorizer.fit(documents)
        vocab = vectorizer.get_feature_names()
        return np.array(vocab)

    @staticmethod
    def _load_embedding_model(model):
        """Load AutoModel from huggingface model repository."""
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModel.from_pretrained(model)
        return tokenizer, model

    @staticmethod
    def _compute_umap(document_vectors, n_components, n_neighbors,
                      random_state=None):
        """Compute low dimensional embeddings using the UMAP algorithm."""
        umap_model = umap.UMAP(n_neighbors=n_neighbors,
                               n_components=n_components,
                               min_dist=0,  # Maximize points density
                               metric='cosine',
                               low_memory=True,
                               random_state=random_state
                               )
        umap_model.fit(document_vectors)
        return umap_model

    @staticmethod
    def _compute_hdbscan(document_vectors, min_cluster_size, min_samples,
                         cluster_selection_method):
        """Perform density-based clustering using the HDBSCAN algorithm."""
        if min_samples is None:
            min_samples = min_cluster_size

        # Compute HDBSCAN clusters
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                        min_samples=min_samples,
                                        metric='euclidean',
                                        cluster_selection_method=cluster_selection_method)
        hdbscan_model.fit(document_vectors)

        return hdbscan_model

    @staticmethod
    def _compute_topic_vectors(document_vectors, hdbscan_model):
        """Compute a representative vector for each cluster found by HDBSCAN."""
        # Extract hdbscan elements
        clusters = hdbscan_model.labels_
        probs = hdbscan_model.probabilities_

        # Exclude noisy documents from topic vector computation
        # If all noisy (no clusters), include everything
        unique_labels = np.unique(clusters)
        if len(unique_labels) > 1 and -1 in unique_labels:
            unique_labels = unique_labels[1:]

        # Compute topic vectors as average of topic's document vectors,
        # weighted by hdbscan confidence score
        topic_vectors = []
        for topic in unique_labels:
            doc_idxs = np.where(clusters == topic)
            if all(probs == 0):
                # All docs noisy case
                topic_vec = np.average(document_vectors[doc_idxs],
                                       axis=0)
            else:
                # Normal case
                topic_vec = np.average(document_vectors[doc_idxs],
                                       weights=probs[doc_idxs],
                                       axis=0)
            topic_vectors.append(topic_vec)

        topic_vectors = np.array(topic_vectors)

        return topic_vectors

    @staticmethod
    def _assign_noisy_docs(document_vectors, topic_vectors, doc_topic):
        """Assign documents classified as noise by HDBSCAN to closest topic."""
        # Compute most similar topic for each document
        doc_topic_sim_mat = cosine_similarity(document_vectors, topic_vectors)
        most_sim_top = np.argmax(doc_topic_sim_mat, axis=1)

        # Assign noisy documents to closest topic
        doc_topic_new = []
        doc_topic_sim = []
        for i, clust in enumerate(doc_topic):
            if clust != -1:
                doc_topic_new.append(clust)
                doc_topic_sim.append(doc_topic_sim_mat[i, clust])
            else:
                new_topic = most_sim_top[i]
                doc_topic_new.append(new_topic)
                doc_topic_sim.append(doc_topic_sim_mat[i, new_topic])

        doc_topic_new = np.array(doc_topic_new)
        doc_topic_sim = np.array(doc_topic_sim)

        return doc_topic_new, doc_topic_sim

    @staticmethod
    def _reorder_topics(doc_topic, topic_vectors):
        """Reorder topics by decreasing size."""
        topics, sizes = np.unique(doc_topic, return_counts=True)
        topics_sorted = topics[np.flip(np.argsort(sizes))]
        mapping_dict = dict(zip(topics_sorted, topics))
        doc_topic_new = [mapping_dict[i] for i in doc_topic]
        topic_vectors_new = topic_vectors[topics_sorted]

        doc_topic_new = np.array(doc_topic_new)
        topic_vectors_new = np.array(topic_vectors_new)

        return doc_topic_new, topic_vectors_new

    def _topic_characterization(self, doc_topic, subset_docs=None, n_words=30):
        """Characterize each topic by the top n words wight highest tf-idf score."""
        if subset_docs is None:
            # In topic extraction, the corpus is the full collection of socuments
            corpus = self.documents
        else:
            # In facet extraction, the corpus is the topic's documents
            # A subset of document ids is passed to restrict the corpus
            corpus = self.documents[subset_docs]

        # Compute tf-idf matrix on the relevant corpus
        vectorizer = TfidfVectorizer(strip_accents="ascii",
                                     lowercase=True,
                                     token_pattern=self.token_pattern,
                                     stop_words="english"
                                     )
        tfidf_model = vectorizer.fit(corpus)
        words = np.array(tfidf_model.get_feature_names())

        topic_rep_words = []
        topic_vectors_tfidf = []
        for topic in np.unique(doc_topic):
            # Join all documents of the topic as a single string
            docs_idxs = np.where(doc_topic == topic)[0]
            big_doc_topic = " ".join(corpus[docs_idxs])
            # Compute tf-idf embedding of this meta -document
            topic_tfidf = vectorizer.transform([big_doc_topic]).toarray()[0]
            topic_vectors_tfidf.append(topic_tfidf)
            # Characterize topic with top n words wight highest tf-idf score
            top_scores = np.flip(np.argsort(topic_tfidf))
            top_words = words[top_scores][:n_words]
            topic_rep_words.append(top_words)

        topic_rep_words = np.array(topic_rep_words)
        topic_vectors_tfidf = np.array(topic_vectors_tfidf)

        return topic_rep_words, topic_vectors_tfidf
