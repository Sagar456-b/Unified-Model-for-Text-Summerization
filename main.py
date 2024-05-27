# Importing required libraries for tensor computations, neural network construction, and optimization
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# NLTK for natural language processing tasks like tokenization and frequency distribution
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import words

# Torch utility for dynamically adjusting learning rate
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Custom modules for data loading and model definition
from data import load_data_Summerization, batch_preprocess, collate_batch, extend_vocab_sequences, TextSummarizationDataset_with_enc_inp, create_embedding_matrix
from model import PointerGeneratorNetwork
from torch.utils.data import DataLoader

# Load data splits
TrainData, ValidationData, TestData = load_data_Summerization()

# Preprocess and select subsets of data for training, validation, and testing
train_articles_subset, train_summaries_subset = batch_preprocess(TrainData[0:100])
val_articles_subset, val_summaries_subset = batch_preprocess(ValidationData[0:100])
test_articles_subset, test_summaries_subset = batch_preprocess(TestData[0:100])

# Combine all articles and summaries for vocabulary building
data = train_articles_subset + train_summaries_subset + val_articles_subset + val_summaries_subset + test_articles_subset + test_summaries_subset

# Tokenize and count frequencies of words in the combined dataset
tokens = [word for sentence in data for word in word_tokenize(sentence.lower())]
freq_dist = FreqDist(tokens)

# Ensure all words in the nltk corpus are in the frequency distribution
nltk_words = set(words.words())
for word in nltk_words:
    if word.lower() not in freq_dist:
        freq_dist[word.lower()] = 1

# Define special tokens and build the vocabulary
special_tokens = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
my_vocab = {token: i + len(special_tokens) for i, token in enumerate(freq_dist)}
my_vocab.update(special_tokens)

# Calculate vocabulary size and maximum sequence lengths
vocab_size = len(my_vocab)
article_max_len = max(len(seq) for seq in train_articles_subset)
summary_max_len = max(len(seq) for seq in train_summaries_subset)

# Determine 95th percentile length for article sequences to limit sequence padding length
lengths = [len(seq) for seq in train_articles_subset]
max_seq_length = int(np.percentile(lengths, 95))

# Set fixed summary length for consistent training and evaluation
summary_length = 150

# Extend the vocabulary of training, validation, and test datasets
enc_extended_inp = extend_vocab_sequences(train_articles_subset, my_vocab, max_seq_length)
val_enc_extended_inp = extend_vocab_sequences(val_articles_subset, my_vocab, max_seq_length)
test_enc_extended_inp = extend_vocab_sequences(test_articles_subset, my_vocab, max_seq_length)

# Create datasets and data loaders
train_dataset = TextSummarizationDataset_with_enc_inp(train_articles_subset, enc_extended_inp, train_summaries_subset, my_vocab, max_seq_length, summary_length)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

val_dataset = TextSummarizationDataset_with_enc_inp(val_articles_subset, val_enc_extended_inp, val_summaries_subset, my_vocab, max_seq_length, summary_length)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

test_dataset = TextSummarizationDataset_with_enc_inp(test_articles_subset, test_enc_extended_inp, test_summaries_subset, my_vocab, max_seq_length, summary_length)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

# Setup device configuration for GPU/CPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to count out-of-vocabulary words
def count_oovs(input_sequence, vocab):
    tokenizer = word_tokenize(input_sequence)
    oov_count = sum(1 for token in tokenizer if token not in vocab)
    return oov_count

# Calculate Out-of-Vocabulary (OOV) for all sequences
oov_counts = [count_oovs(sequence, my_vocab) for sequence in data]
max_oov = int(np.percentile(oov_counts, 90))

# Initialize and setup the Pointer Generator Network model
Unifiedmodel = PointerGeneratorNetwork(vocab_size=vocab_size, embedding_dim=200, enc_units=128, dec_units=128, max_oov=max_oov, max_len=summary_length, vocab=my_vocab, device=device)
Unifiedmodel.to(device)

# Create embedding matrix and configure it in the model
embedding_matrix = create_embedding_matrix(my_vocab)
Unifiedmodel.encoder.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix[:-1, :]))
Unifiedmodel.encoder.embedding.weight.requires_grad = False

# Define optimizer and loss function
optimizer = optim.Adam(Unifiedmodel.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=my_vocab['<pad>'])
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)

# Train and validate the model
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_train_loss = 0
    total_samples = len(train_loader.dataset)

    for batch_idx, (articles, ext_enc_inp, summaries) in enumerate(train_loader):
        articles, ext_enc_inp, summaries = articles.to(device), ext_enc_inp.to(device), summaries.to(device)
        optimizer.zero_grad()
        loss = model(articles, ext_enc_inp, summaries[:, :-1], summaries[:, 1:], teacher_forcing_ratio=0.5)
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        total_train_loss += loss.item() * articles.size(0)
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Current Batch Loss: {loss.item()}")

    average_loss = total_train_loss / total_samples
    return average_loss

# Function to evaluate model on validation dataset
def validate_epoch(model, validation_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    total_samples = len(validation_loader.dataset)

    with torch.no_grad():
        for articles, ext_enc_inp, summaries in validation_loader:
            articles, ext_enc_inp, summaries = articles.to(device), ext_enc_inp.to(device), summaries.to(device)
            outputs, _ = model(articles, ext_enc_inp, summaries[:, :-1])
            loss = criterion(outputs.view(-1, outputs.size(-1)), summaries[:, 1:].reshape(-1))
            total_val_loss += loss.item() * articles.size(0)

    avg_val_loss = total_val_loss / total_samples
    return avg_val_loss

# Function to evaluate model on test dataset
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = len(test_loader.dataset)

    with torch.no_grad():
        for articles, ext_enc_inp, summaries in test_loader:
            articles, ext_enc_inp, summaries = articles.to(device), ext_enc_inp.to(device), summaries.to(device)
            outputs, _ = model(articles, ext_enc_inp, summaries[:, :-1], teacher_forcing_ratio=0)
            loss = criterion(outputs.view(-1, outputs.size(-1)), summaries[:, 1:].reshape(-1))
            total_loss += loss.item() * articles.size(0)

    average_loss = total_loss / total_samples
    print(f'Test Loss: {average_loss:.4f}')
    return average_loss
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
                greedy_texts.append(' '.join(greedy_words))

                # Beam search for each sequence in the batch.
                beam_sequence = model.beam_search(input_seqs[idx:idx + 1], ext_inps[idx:idx + 1], beam_width, max_len)
                beam_words = indices_to_text(beam_sequence, vocab)
                beam_texts.append(' '.join(beam_words))

                # Optionally store actual sequences for comparison, converted to words.
                actual_sequence = targets[idx].tolist()  # Assuming targets is a tensor.
                actual_words = indices_to_text(actual_sequence, vocab)
                actual_summaries.append(' '.join(actual_words))

                # Actual input text
                actual_input_text = indices_to_text(input_seqs[idx], vocab)
                actual_texts.append(actual_input_text)

    return {
        "greedy_predictions": greedy_texts,
        "beam_predictions": beam_texts,
        "actual_texts": actual_texts,
        "actual_summaries": actual_summaries
    }

# Main function to train and test the model
def main():
    num_epochs = 30
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        try:
            train_loss = train_epoch(Unifiedmodel, train_loader, optimizer, criterion, device)
            val_loss = validate_epoch(Unifiedmodel, val_loader, criterion, device)
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # torch.save(Unifiedmodel.state_dict(), 'PN_best.pth')
                # print("Saved improved model checkpoint.")
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")

    # torch.save(Unifiedmodel.state_dict(), 'pointer_generator_network_final.pth')
    # print("Saved final model state.")
    test_loss = test(Unifiedmodel, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')

# Run the main function if this file is executed as a script
if __name__ == "__main__":
    main()
    results = generate_predictions(Unifiedmodel, test_loader, my_vocab, device, 150, 4)

    # You can then access results like this:
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
