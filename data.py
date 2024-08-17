import numpy as np
import torch 
from scipy.io import loadmat
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def read_mat_file(key, path):
    """
    read the preprocess mat file whose key and and path are passed as parameters

    Args:
        key ([type]): [description]
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    #term_path = Path().cwd().joinpath('data', 'preprocess', path)
    term_path = Path(path).with_suffix('.mat')
    doc = loadmat(term_path)[key]
    return doc


def split_train_test_matrix(dataset):
    """Split the dataset into the train set , the validation and the test set

    Args:
        dataset ([type]): [description]

    Returns:
        [type]: [description]
    """
    X_train, X_test = train_test_split(dataset, test_size=0.2, random_state=1)
    X_test_1, X_test_2 = train_test_split(X_test, test_size=0.5, random_state=1)
    X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    return X_train, X_val, X_test_1, X_test_2

def get_data(doc_terms_file_name="tf_idf_doc_terms_matrix", terms_filename="tf_idf_terms"):
    """read the data and return the vocabulary as well as the train, test and validation tests

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    doc_term_matrix = read_mat_file("doc_terms_matrix", doc_terms_file_name)
    terms = read_mat_file("terms", terms_filename)
    vocab = terms.flatten() #did not use flatten before
    train, validation, test_1, test_2 = split_train_test_matrix(doc_term_matrix)
    return vocab, train, validation, test_1, test_2

def get_batch(doc_terms_matrix, indices, device):
    """
    get the a sample of the given indices 

    Basically get the given indices from the dataset

    Args:
        doc_terms_matrix ([type]): the document term matrix
        indices ([type]):  numpy array 
        vocab_size ([type]): [description]

    Returns:
        torch.Tensor: A tensor with the data passed as parameter, moved to the specified device.
    """
    data_batch = doc_terms_matrix[indices, :]
    data_batch = torch.from_numpy(data_batch.toarray()).float().to(device)
    return data_batch


def read_embedding_matrix(vocab, device,  load_trainned=True):
    """
    read the embedding  matrix passed as parameter and return it as an vocabulary of each word 
    with the corresponding embeddings

    Args:
        path ([type]): [description]

    # we need to use tensorflow embedding lookup heer
    """
    embeddings_path = Path().cwd().joinpath('data', 'preprocess', "embedding_matrix.npy")

    if load_trainned:
        embeddings_matrix = np.load(embeddings_path, allow_pickle=True)
    else:
        # Use Hugging Face's BERT model to generate embeddings
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased").to(device)

        embeddings_matrix = np.zeros((len(vocab), model.config.hidden_size))
        print("Starting to generate word embeddings using Hugging Face model...")
        vocab = vocab.ravel()

        for index, word in tqdm(enumerate(vocab), total=len(vocab)):
            inputs = tokenizer(str(word), return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # Use the mean of the token embeddings as the word embedding
            embeddings_matrix[index] = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        print("Done generating the word embeddings.")
        with open(embeddings_path, 'wb') as file_path:
            np.save(file_path, embeddings_matrix)

    embeddings = torch.from_numpy(embeddings_matrix).float().to(device)
    return embeddings