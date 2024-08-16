import numpy as np
from scipy.io import savemat
from scipy import sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_20ng():
    print("Loading the 20 Newsgroups dataset...")
    newsgroups_data = fetch_20newsgroups(subset='all')
    documents = newsgroups_data.data
    print(f"Loaded {len(documents)} documents.")

    print("\nCreating a consistent vocabulary using CountVectorizer...")
    count_vectorizer = CountVectorizer(max_df=0.7, min_df=10, stop_words='english')
    term_doc_matrix = count_vectorizer.fit_transform(documents)
    vocab = count_vectorizer.get_feature_names_out()
    print(f"Vocabulary size: {len(vocab)} terms.")

    # Save the vocabulary as a 1D array
    print("\nSaving the vocabulary...")
    savemat("tf_idf_terms_time_window_1.mat", {'terms': np.array(vocab, dtype=object)})
    print("Vocabulary saved to tf_idf_terms_time_window_1.mat.")

    print("\nSaving the document-term matrix...")
    sparse_matrix = sparse.csr_matrix(term_doc_matrix)  # Convert to sparse matrix
    savemat("tf_idf_doc_terms_matrix_time_window_1.mat", {'doc_terms_matrix': sparse_matrix})
    print("Document-term matrix saved to tf_idf_doc_terms_matrix_time_window_1.mat.")

if __name__ == "__main__":
    preprocess_20ng()
