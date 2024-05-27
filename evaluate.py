from matplotlib import pyplot as plt
from rouge import Rouge

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import words
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import load_data_Summerization, batch_preprocess, collate_batch, extend_vocab_sequences, \
    TextSummarizationDataset_with_enc_inp
from model import PointerGeneratorNetwork
from torch.utils.data import DataLoader

TrainData, ValidationData, TestData = load_data_Summerization()

train_articles_subset, train_summaries_subset = batch_preprocess(TrainData[0:100])
val_articles_subset, val_summaries_subset = batch_preprocess(ValidationData[0:100])
test_articles_subset, test_summaries_subset = batch_preprocess(TestData[0:100])

data = train_articles_subset + train_summaries_subset + val_articles_subset + val_summaries_subset + test_articles_subset + test_summaries_subset
tokens = [word for sentence in data for word in word_tokenize(sentence.lower())]

freq_dist = FreqDist(tokens)
nltk_words = set(words.words())
for word in nltk_words:
    if word.lower() not in freq_dist:
        freq_dist[word.lower()] = 1

special_tokens = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
my_vocab = {token: i + len(special_tokens) for i, token in enumerate(freq_dist)}
my_vocab.update(special_tokens)

vocab_size = len(my_vocab)
print("vocab size:", vocab_size)
article_max_len = max(len(seq) for seq in train_articles_subset)
summary_max_len = max(len(seq) for seq in train_summaries_subset)

lengths = [len(seq) for seq in train_articles_subset]
max_seq_length = int(np.percentile(lengths, 95))
sum_lengths = [len(seq) for seq in train_summaries_subset]
summary_length = 100

enc_extended_inp = extend_vocab_sequences(train_articles_subset, my_vocab, max_seq_length)
val_enc_extended_inp = extend_vocab_sequences(val_articles_subset, my_vocab, max_seq_length)
test_enc_extended_inp = extend_vocab_sequences(test_articles_subset, my_vocab, max_seq_length)

train_dataset = TextSummarizationDataset_with_enc_inp(train_articles_subset, enc_extended_inp, train_summaries_subset,
                                                      my_vocab, max_seq_length, summary_length)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

val_dataset = TextSummarizationDataset_with_enc_inp(val_articles_subset, val_enc_extended_inp, val_summaries_subset,
                                                    my_vocab, max_seq_length, summary_length)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

test_dataset = TextSummarizationDataset_with_enc_inp(test_articles_subset, test_enc_extended_inp, test_summaries_subset,
                                                     my_vocab, max_seq_length, summary_length)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_oovs(input_sequence, vocab):
    """
    Counts the number of out-of-vocabulary (OOV) words in an input sequence.

    Args:
    input_sequence (str): A single input sequence as a string.
    vocab (dict): A dictionary representing the vocabulary mapping of words to indices.

    Returns:
    int: The count of OOV words in the input sequence.
    """
    # Tokenize the input sequence
    tokenizer = word_tokenize(input_sequence)

    # Count OOVs by checking if each token is not in the vocabulary
    oov_count = sum(1 for token in tokenizer if token not in vocab)

    return oov_count


# Calculate OOV counts for each sequence in your dataset
oov_counts = [count_oovs(sequence, my_vocab) for sequence in data]

# Calculate the 90th percentile of OOV counts to cover most of the dataset
max_oov = int(np.percentile(oov_counts, 90))



# Assuming your saved model file is named 'pointer_generator_network_best.pth'
model = PointerGeneratorNetwork(vocab_size=278056, embedding_dim=200, enc_units=128, dec_units=128, max_oov=0, max_len=150, vocab=my_vocab, device=device)
model.load_state_dict(torch.load('pointer_generator_network_final.pth'))
model.eval()  # Set the model to evaluation mode


def indices_to_text(indices, vocab):
    # print("Input indices type:", type(indices))  # Debug print
    # print("Input indices:", indices)  # Debug print

    if isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()
    elif isinstance(indices, torch.Tensor):
        indices = indices.flatten().tolist()
    elif not isinstance(indices, (list, tuple)):
        raise TypeError("Indices must be a list, tuple, numpy.ndarray, or torch.Tensor.")

    idx_to_token = {index: token for token, index in vocab.items()}
    special_indices = [vocab.get('<pad>'), vocab.get('<sos>'), vocab.get('<eos>')]
    unk_token = "<unk>"  # Ensure this is consistent with your vocab

    mytokens = []
    for idx in indices:
        if idx in special_indices:
            continue  # Skip special tokens for output
        token = idx_to_token.get(idx, unk_token)  # Handle OOV tokens
        mytokens.append(token)

    # Concatenate tokens into a string
    text = ' '.join(mytokens).strip()

    # print("Output text:", text)  # Debug print

    return text


def generate_predictions(model, test_dataloader, vocab, device, max_len, beam_width):
    model.eval()  # Set the model to evaluation mode.
    greedy_texts = []
    beam_texts = []
    actual_texts = []
    actual_summaries = []

    with torch.no_grad():  # No need to track gradients during evaluation.
        for input_seqs, ext_inps, targets in test_dataloader:
            # Process each example in the batch
            for idx in range(input_seqs.size(0)):
                # Greedy decoding for each sequence in the batch.
                greedy_sequence = model.greedy_search(input_seqs[idx:idx + 1], ext_inps[idx:idx + 1],
                                                      max_length=max_len)
                greedy_words = indices_to_text(greedy_sequence, vocab)
                greedy_texts.append(greedy_words)

                # Beam search for each sequence in the batch.
                beam_sequence = model.beam_search(input_seqs[idx:idx + 1], ext_inps[idx:idx + 1], beam_width, max_len)
                beam_words = indices_to_text(beam_sequence, vocab)
                beam_texts.append(beam_words)

                # Optionally store actual sequences for comparison, converted to words.
                actual_sequence = targets[idx].tolist()  # Assuming targets is a tensor.
                actual_words = indices_to_text(actual_sequence, vocab)
                actual_summaries.append(actual_words)

                # Actual input text
                actual_input_text = indices_to_text(input_seqs[idx], vocab)
                actual_texts.append(actual_input_text)

    return {
        "greedy_predictions": greedy_texts,
        "beam_predictions": beam_texts,
        "actual_texts": actual_texts,
        "actual_summaries": actual_summaries
    }


# Set the maximum length and beam width for beam search
max_len = 100
beam_width = 4


# Generate predictions using the model
results = generate_predictions(model, test_loader, my_vocab, device, max_len, beam_width)

# Access the predictions and actual summaries from the results
greedy_predictions = results['greedy_predictions']
beam_predictions = results['beam_predictions']
actual_texts = results['actual_texts']
actual_summaries = results['actual_summaries']
for i in range(min(5, len(greedy_predictions))):
    print("*****************************************************************")
    print(f"Actual text: {actual_texts[i]}\n")
    print("__________________________________________________________________")
    print(f"Actual Summery: {actual_summaries[i]}\n")
    print("-----------------------------------------------------\n")
    #
    print(f"Greedy decoding Summary: {greedy_predictions[i]}\n")
    print("__________________________________________________________________")
    print(f"PREDICTED WITH BEAM SEARCH: {beam_predictions[i]}\n")

#
# # Now, you can use the ROUGE evaluation code to compute ROUGE scores based on these predictions and actual summaries
# # Define ROUGE scorer
# # from rouge_score import rouge_scorer
# scorer = Rouge()
#
#
# def compute_rouge(hypotheses, references):
#     rouge_scores = scorer.get_scores(hypotheses, references)
#     return rouge_scores
#
#
# def evaluate_with_rouge(model, test_loader, vocab, device, max_len, beam_width):
#     model.eval()  # Set the model to evaluation mode.
#     greedy_predictions = []
#     beam_predictions = []
#     actual_summaries = []
#
#     with torch.no_grad():  # No need to track gradients during evaluation.
#         for input_seqs, ext_inps, targets in test_loader:
#             # Process each example in the batch
#             for idx in range(input_seqs.size(0)):
#                 # Greedy decoding for each sequence in the batch.
#                 greedy_sequence = model.greedy_search(input_seqs[idx:idx + 1], ext_inps[idx:idx + 1],
#                                                       max_length=max_len)
#                 greedy_words = indices_to_text(greedy_sequence, vocab)
#                 greedy_predictions.append(greedy_words)
#
#                 # Beam search for each sequence in the batch.
#                 beam_sequence = model.beam_search(input_seqs[idx:idx + 1], ext_inps[idx:idx + 1], beam_width,
#                                                   max_len)
#                 beam_words = indices_to_text(beam_sequence, vocab)
#                 beam_predictions.append(beam_words)
#
#                 # Reference summaries
#                 actual_sequence = targets[idx].tolist()  # Assuming targets is a tensor.
#                 actual_words = indices_to_text(actual_sequence, vocab)
#                 actual_summaries.append(actual_words)
#
#     # Compute ROUGE scores
#     rouge_scores_greedy = compute_rouge(greedy_predictions, actual_summaries)
#     rouge_scores_beam = compute_rouge(beam_predictions, actual_summaries)
#
#     return rouge_scores_greedy, rouge_scores_beam
#
#
# # Call the function to evaluate with ROUGE
# rouge_scores_greedy, rouge_scores_beam = evaluate_with_rouge(model, test_loader, my_vocab, device, max_len,
#                                                              beam_width)
#
# # # Print ROUGE scores
# # print("ROUGE scores for greedy decoding:")
# # print(rouge_scores_greedy)
# # print("ROUGE scores for beam search decoding:")
# # print(rouge_scores_beam)
# #
# # # print("Debug: rouge_scores_greedy =", rouge_scores_greedy)
# # Calculate average F1 scores for each ROUGE metric for both methods
# average_greedy_scores = {
#     'rouge-1': np.mean([d['rouge-1']['f'] for d in rouge_scores_greedy]),
#     'rouge-2': np.mean([d['rouge-2']['f'] for d in rouge_scores_greedy]),
#     'rouge-l': np.mean([d['rouge-l']['f'] for d in rouge_scores_greedy])
# }
#
# average_beam_scores = {
#     'rouge-1': np.mean([d['rouge-1']['f'] for d in rouge_scores_beam]),
#     'rouge-2': np.mean([d['rouge-2']['f'] for d in rouge_scores_beam]),
#     'rouge-l': np.mean([d['rouge-l']['f'] for d in rouge_scores_beam])
# }
#
# categories = ['rouge-1', 'rouge-2', 'rouge-l']
# greedy_scores = [average_greedy_scores[cat] for cat in categories]
# beam_scores = [average_beam_scores[cat] for cat in categories]
#
# # Data Preparation for Plotting
# fig, ax = plt.subplots()
# bar_width = 0.35
# index = np.arange(len(categories))
#
# bar1 = plt.bar(index, greedy_scores, bar_width, label='Greedy')
# bar2 = plt.bar(index + bar_width, beam_scores, bar_width, label='Beam')
#
# plt.xlabel('ROUGE Metrics')
# plt.ylabel('F1 Scores')
# plt.title('Comparison of ROUGE Scores for Greedy vs. Beam Search')
# plt.xticks(index + bar_width / 2, categories)
# plt.legend()
#
# plt.tight_layout()
# plt.show()
