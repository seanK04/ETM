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


def read_embedding_matrix(vocab, device, load_trainned=True, batch_size=32):
    """
    Read the embedding matrix passed as a parameter and return it as a vocabulary of each word 
    with the corresponding embeddings.

    Args:
        vocab (np.ndarray): Vocabulary array containing words.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
        load_trainned (bool): Whether to load pre-trained embeddings or generate new ones.
        batch_size (int): Number of words to process in each batch.

    Returns:
        torch.Tensor: Embedding matrix as a PyTorch tensor.
    """
    embeddings_dir = Path().cwd().joinpath('data', 'preprocess')
    embeddings_path = embeddings_dir.joinpath("embedding_matrix.npy")

    # Ensure the directory exists
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    if load_trainned:
        if embeddings_path.exists():
            embeddings_matrix = np.load(embeddings_path, allow_pickle=True)
        else:
            raise FileNotFoundError(f"Pre-trained embedding file not found at {embeddings_path}")
    else:
        # Use Hugging Face's BERT model to generate embeddings
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased").to(device)

        embeddings_matrix = np.zeros((len(vocab), model.config.hidden_size))
        print("Starting to generate word embeddings using Hugging Face model...")
        vocab = vocab.ravel()

        # Ensure vocabulary is a list of strings
        vocab = [word.decode('utf-8') if isinstance(word, bytes) else str(word) for word in vocab]

        for i in tqdm(range(0, len(vocab), batch_size), total=len(vocab) // batch_size):
            batch_words = vocab[i:i + batch_size]
            inputs = tokenizer(batch_words, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # Use the mean of the token embeddings as the word embedding
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings_matrix[i:i + batch_size] = batch_embeddings

        print("Done generating the word embeddings.")
        with open(embeddings_path, 'wb') as file_path:
            np.save(file_path, embeddings_matrix)

    embeddings = torch.from_numpy(embeddings_matrix).float().to(device)
    return embeddings
