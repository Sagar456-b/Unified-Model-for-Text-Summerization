import pandas as pd
import re
import torch
from torch.utils.data import Dataset
# from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import contractions
import nltk
from nltk.corpus import stopwords, wordnet

from nltk.stem import WordNetLemmatizer
import string
import numpy as np
from torchtext.data.utils import get_tokenizer

# from torchtext.vocab import build_vocab_from_iterator, vocab
# from collections import Counter

# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk.tag import pos_tag
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.corpus import wordnet
#
# import string
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# import nltk
# # nltk.download("wordnet")
# # nltk.download('averaged_perceptron_tagger')
# from nltk.stem import WordNetLemmatizer
# import contractions
# from nltk.corpus import wordnet
#


# Ensure NLTK resources are downloaded (run once)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')


lemmatizer = WordNetLemmatizer()


def load_data_Summerization():
    Train_df = pd.read_csv("data/cnn_dailymail/train.csv")
    val_df = pd.read_csv("data/cnn_dailymail/validation.csv")
    Test_df = pd.read_csv("data/cnn_dailymail/test.csv")
    # droping the unnecessary column
    Updated_train_data = Train_df.drop(['id'], axis=1)
    Updated_val_data = val_df.drop(['id'], axis=1)
    Updated_Test_data = Test_df.drop(['id'], axis=1)

    return Updated_train_data, Updated_val_data, Updated_Test_data


def get_wordnet_pos(tag):
    """Map POS tag to first character lemmatize() accepts"""
    # tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
def preprocessing(data, data_column):
    processed_sent = []
    for sent in data[data_column]:
        # Remove text in brackets and fix contractions
        processed_sent_sub = re.sub("[\(\[].*?[\)\]]", "", sent)
        processed_sent_sub = contractions.fix(processed_sent_sub)
        processed_sent.append(processed_sent_sub)

    WORD = re.compile(r'\w+')
    sentence = []
    for art in processed_sent:
        words = WORD.findall(art)
        sentence.append(words)

    punctuation = string.punctuation
    format_sent = []

    for sntc in sentence:
        format_words = []
        for word in sntc:
            # Only check for punctuation and word length; stopwords are not removed
            if word not in punctuation and len(word) > 2:
                format_words.append(word.lower())
        format_words = nltk.pos_tag(format_words)
        doc = [lemmatizer.lemmatize(word[0], get_wordnet_pos(word[1][0].upper())) for word in format_words]
        format_sent.append(" ".join(doc))

    return format_sent


def batch_preprocess(data_with_size):
    article_column = 'article'
    summaries_column = 'highlights'
    preprocessed_articles = preprocessing(data_with_size, article_column)
    preprocessed_summaries = preprocessing(data_with_size, summaries_column)
    return preprocessed_articles, preprocessed_summaries


# By using this custom collation function with collate_fn=collate_batch when creating a DataLoader, you can ensure that your data samples are collated and padded appropriately into batches before being passed to your model during training or evaluation.
def collate_batch(batch):
    articles, ext_enc_inps, summaries = [], [], []
    for article, ext_enc_inp, summary in batch:
        articles.append(article)
        ext_enc_inps.append(ext_enc_inp)  # Assuming there's an extended encoder input to handle as well
        summaries.append(summary)

    # Padding
    articles_padded = pad_sequence(articles, batch_first=True, padding_value=0)
    ext_enc_inps_padded = pad_sequence(ext_enc_inps, batch_first=True,
                                       padding_value=0)  # Handle padding for extended inputs too, if applicable
    summaries_padded = pad_sequence(summaries, batch_first=True, padding_value=0)

    return articles_padded, ext_enc_inps_padded, summaries_padded


def extend_vocab_sequences(raw_texts, my_vocab, max_len):
    tokenizer = get_tokenizer("basic_english")
    extended_sequences = []
    oov_dict = {}
    vocab = my_vocab
    eos_token_id = my_vocab['<eos>']
    sos_token_id = my_vocab['<sos>']
    pad_token_id = my_vocab['<pad>']
    unk_token_id = my_vocab['<unk>']
    current_max_index = max(vocab.values()) + 1  # Ensure we start adding new ids after the existing ones

    for text in raw_texts:
        # Tokenize the text and add <sos> and <eos>
        tokens = ["<sos>"] + tokenizer(text.lower()) + ["<eos>"]
        extended_sequence = []
        for token in tokens:
            if token == "<sos>":
                token_id = sos_token_id
            elif token == "<eos>":
                token_id = eos_token_id
            else:
                token_id = vocab.get(token, None)
                if token_id is None:
                    # Handle OOV tokens
                    if token not in oov_dict:
                        oov_dict[token] = current_max_index
                        current_max_index += 1
                    token_id = oov_dict[token]
                # For tokens not found in the vocab and not special tokens, use unk_id
                token_id = token_id if token_id is not None else unk_token_id
            extended_sequence.append(token_id)

        # Ensure the sequence does not exceed max_len, adjusting for <eos>
        if len(extended_sequence) > max_len:
            extended_sequence = extended_sequence[:max_len - 1] + [extended_sequence[-1]]

        extended_sequences.append(torch.tensor(extended_sequence, dtype=torch.long))

    return extended_sequences


# inlcude ext inp here -use it later
class TextSummarizationDataset_with_enc_inp(Dataset):
    def __init__(self, articles, ext_inp, summaries, my_vocab, article_max_len, summary_max_len):
        self.articles = articles
        self.summaries = summaries
        self.ext_inp = ext_inp  # Include external input
        self.vocab = my_vocab
        self.article_max_len = article_max_len
        self.summary_max_len = summary_max_len
        self.tokenizer = get_tokenizer("basic_english")

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.vectorize(self.articles[idx], self.article_max_len)
        ext_enc_input = self.ext_inp[idx]
        summary = self.vectorize(self.summaries[idx], self.summary_max_len)

        # ext_enc_input = self.vectorize(self.ext_inp[idx], self.article_max_len)
        # Get external input for the current index
        # ext_enc_input = self.ext_inp[idx]
        return article, ext_enc_input, summary  # Return article,external input and  summary

    def vectorize(self, text, max_len):
        # Tokenize the text, adding <sos> at the beginning and <eos> at the end
        tokens = ["<sos>"] + self.tokenizer(text.lower()) + ["<eos>"]
        # Convert tokens to their corresponding IDs in the vocabulary
        # If a token is not found, use the ID for <unk> (unknown token)
        token_ids = [self.vocab.get(token, self.vocab.get('<unk>', 3)) for token in tokens]
        # Ensure the token sequence, including <sos> and <eos>, does not exceed max_len
        # If it does, truncate the sequence while ensuring <eos> is at the end
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len - 1] + [token_ids[-1]]
        # Convert the sequence of token IDs into a PyTorch tensor
        return torch.tensor(token_ids, dtype=torch.long)


def load_glove_embeddings(path):
    """
    Load GloVe embeddings from a file.

    Args:
    - path (str): Path to the GloVe embeddings file.

    Returns:
    - embeddings_dict (dict): A dictionary where keys are words and values are embedding vectors.
    """
    embeddings_dict = {}
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict


# Example path might be "glove.6B.100d.txt" for 100-dimensional vectors.
# glove_path = "glove.6B.200d.txt"
# embeeding_dict = load_glove_embeddings(glove_path)


def create_embedding_matrix(word_index):
    """
    Create an embedding matrix tailored to your vocabulary.

    Args:
    - word_index (dict): A dictionary mapping words to their integer index.
    - embedding_dict (dict): A dictionary with words as keys and their embedding vectors as values.
    - dimension (int): The dimensionality of the embeddings.

    Returns:
    - embedding_matrix (numpy array): An array where each row number corresponds to a word index and contains that word's embedding vector.
    """
    glove_path = "glove.6B.200d.txt"
    embedding_dict = load_glove_embeddings(glove_path)
    dimension = 200
    embedding_matrix = np.zeros((len(word_index) + 1, dimension))
    for word, i in word_index.items():
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix



def remove_whitespace_and_unk(text):
    # Split the text into parts (assuming double spaces indicate original spaces between words)
    parts = text.split('  ')
    cleaned_parts = []

    for part in parts:
        # Remove extra spaces within each word
        cleaned_word = ''.join(part.split())
        cleaned_parts.append(cleaned_word)
        # If the cleaned word is '<unk>', we skip adding it to the cleaned parts
        # if cleaned_word != '<unk>':
        #     cleaned_parts.append(cleaned_word)

    # Rejoin the cleaned parts with a single space, as they represent separate words
    cleaned_text = ' '.join(cleaned_parts)
    return cleaned_text
