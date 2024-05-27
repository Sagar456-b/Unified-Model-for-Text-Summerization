# Importing the necessary PyTorch modules for constructing neural network layers, activation functions, and utilities
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the custom GRU layer implementation from a separate module
from CustomGRU import CustomGRULayer

# Encoder class definition inheriting from nn.Module for modular neural network construction
class Encoder(nn.Module):
    """
    Encoder module using a GRU layer for processing input sequences into a continuous representation.
    """
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer to convert token indices to vectors
        self.gru = CustomGRULayer(embedding_dim, enc_units, batch_first=True)  # Custom GRU layers for sequence processing
        self.layer_norm = nn.LayerNorm(enc_units)  # Layer normalization for stabilizing the neural network's output

    def forward(self, x):
        embedded = self.embedding(x)  # Pass input through the embedding layer
        output, state = self.gru(embedded)  # GRU processing
        output = self.layer_norm(output)  # Normalize the output of the GRU
        return output, state  # Return the final output and the hidden state

# BahdanauAttention class definition, implementing the attention mechanism for neural networks
class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention for the decoder.
    """
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(units, units)
        self.W2 = nn.Linear(units, units)
        self.V = nn.Linear(units, 1)
        self.score_norm = nn.LayerNorm(1)

    def forward(self, query, values):
        if query.dim() == 3 and query.size(0) == 1:
            query = query.squeeze(0)
        query = query.unsqueeze(1)

        transformed_values = self.W1(values)
        transformed_query = self.W2(query)
        score = self.V(torch.tanh(transformed_values + transformed_query))
        score = self.score_norm(score)
        attention_weights = F.softmax(score, dim=1)
        context_vector = torch.sum(attention_weights * values, dim=1)
        return context_vector, attention_weights.squeeze(-1)

# Pointer class definition for implementing the pointer mechanism in neural networks
class Pointer(nn.Module):
    """
    Pointer network used to calculate probabilities of pointing to elements in the input sequence.
    """
    def __init__(self, input_feature):
        super(Pointer, self).__init__()
        self.linear_layers = nn.ModuleDict({
            'w_s_reduce': nn.Linear(input_feature, 1),
            'w_i_reduce': nn.Linear(input_feature, 1),
            'w_c_reduce': nn.Linear(input_feature, 1)
        })

    def forward(self, context_vector, state, dec_inp):
        # Calculate the sigmoid activation for deciding whether to point or generate a token
        sigmoid_input = sum(self.linear_layers[layer](tensor) for layer, tensor in
                            zip(['w_s_reduce', 'w_i_reduce', 'w_c_reduce'], [state, context_vector, dec_inp]))
        return torch.sigmoid(sigmoid_input)

# PGDecoder class definition, incorporating both Bahdanau attention and a pointer mechanism
class PGDecoder(nn.Module):
    """
    Decoder module for the Pointer-Generator network, integrating both generation and pointing capabilities.
    """
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(PGDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = CustomGRULayer(embedding_dim + dec_units, dec_units, batch_first=True)
        self.fc = nn.Linear(dec_units, vocab_size)  # Fully connected layer to predict vocabulary tokens
        self.attention = BahdanauAttention(dec_units)  # Attention mechanism
        self.pointer = Pointer(dec_units)  # Pointer mechanism
        self.W1 = nn.Linear(dec_units * 2, dec_units)  # Linear layer for combining context vector and GRU output
        self.W2 = nn.Linear(dec_units, vocab_size)  # Linear layer for outputting vocabulary probabilities

    def forward(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        decoder_emb_x = self.embedding(x)
        # Here, the embedded input decoder_emb_x is concatenated with the context_vector along the last dimension. The context vector is unsqueezed to add a singleton dimension, aligning its dimensions with the embedded input for concatenation. This combination feeds more information into the GRU.
        x_combined = torch.cat((decoder_emb_x, context_vector.unsqueeze(1)), dim=-1)
        output, state = self.gru(x_combined, hidden)
        output = output.squeeze(1)
        concat_vector = torch.cat([output, context_vector], dim=-1)
        # These lines represent feed-forward neural network layers applied to the concatenated vector. self.W1 and self.W2 are likely linear (fully connected) layers that transform the concatenated vector into outputs that are used to predict the next word in the sequence.
        FF1_out = self.W1(concat_vector)
        FF2_out = self.W2(FF1_out)
        # The output of the second feed-forward layer is passed through a softmax function to produce a probability distribution over the vocabulary (p_vocab), representing the likelihood of each word being the next word in the generated sequence.
        p_vocab = F.softmax(FF2_out, dim=1)
        # The decoder state is adjusted (squeezed) and used along with the context vector and decoder output to compute p_gen using a pointer network. The pointer network can choose to copy words directly from the input sequence, aiding in tasks like text summarization where reproducing exact details can be crucial.
        adjusted_state = state.squeeze(0)
        p_gen = self.pointer(context_vector, adjusted_state, output)
        return p_vocab, adjusted_state, p_gen, attention_weights

# PointerGeneratorNetwork class definition, which combines the Encoder and PGDecoder modules
class PointerGeneratorNetwork(nn.Module):
    """
    Complete Pointer-Generator Network incorporating the encoder, decoder, and attention modules.
    """
    def __init__(self, vocab_size, embedding_dim, enc_units, dec_units, max_oov, max_len, vocab, device):
        super(PointerGeneratorNetwork, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.dec_units = dec_units
        self.max_oov = max_oov
        self.device = device
        self.vocab = vocab
        self.encoder = Encoder(vocab_size, embedding_dim, enc_units)
        self.decoder = PGDecoder(vocab_size + max_oov, embedding_dim, dec_units)
        self.decoder.embedding = self.encoder.embedding  # Share embedding layer between encoder and decoder
        self.max_length = max_len
        self.sos_token_id = vocab['<sos>']
        self.eos_token_id = vocab['<eos>']
        self.pad_token_id = vocab['<pad>']

    def forward(self, enc_input, enc_input_ext, dec_input, target=None, teacher_forcing_ratio=0.5):
        # Process input through the encoder and decoder while managing the training process
        enc_output, enc_hidden = self.encoder(enc_input)
        dec_hidden = enc_hidden
        batch_size, seq_len = dec_input.size()
        if self.training:
            loss = 0
            # The decoder processes each token of the input sequence one at a time.
            # The dec_input_t is the current token, and true_output is the expected output token at the next time step.
            for t in range(dec_input.shape[1] - 1):
                dec_input_t = dec_input[:, t].unsqueeze(1)
                true_output = target[:, t + 1]
                p_vocab, dec_hidden, p_gen, attn = self.decoder(dec_input_t, dec_hidden, enc_output)
                final_dist = self.get_final_distribution(enc_input_ext, p_gen, p_vocab, attn, self.max_oov,
                                                         self.vocab_size, batch_size)
                loss += F.cross_entropy(final_dist, true_output, ignore_index=self.pad_token_id)
            return loss / (seq_len - 1)
        else:
            # Inference mode, generate output sequences based on the trained model
            generated_tokens = []
            dec_input_t = torch.tensor([[self.sos_token_id]] * batch_size, device=self.device)  # Start with <sos>
            finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
            predictions = torch.zeros(batch_size, seq_len, self.vocab_size, device=self.device)

            for t in range(dec_input.shape[1] - 1):
                if finished.all():
                    break  # Exit loop if all sequences are completed
                p_vocab, dec_hidden, p_gen, attn = self.decoder(dec_input_t, dec_hidden, enc_output)
                final_dist = self.get_final_distribution(enc_input_ext, p_gen, p_vocab, attn, self.max_oov,
                                                         self.vocab_size, batch_size)
                predictions[:, t, :] = final_dist
                topi = final_dist.max(1)[1]
                generated_tokens.append(topi.detach())
                finished |= (topi == self.eos_token_id)
                dec_input_t = topi.unsqueeze(1).clone()
                dec_input_t[finished] = self.pad_token_id  # Replace <eos> with <pad> for finished sequences

            if not generated_tokens:
                # Handle case where no tokens were generated
                generated_tokens_tensor = torch.empty(0, dtype=torch.long, device=self.device)
            else:
                generated_tokens_tensor = torch.stack(generated_tokens, dim=1)

            return predictions, generated_tokens_tensor

    def get_final_distribution(self, enc_batch_extend_vocab, p_gen, p_vocab, attention_weights, max_oov, vocab_size,
                               batch_size):
        # Compute the final distribution for vocabulary and pointer mechanisms
        p_gen = torch.clamp(p_gen, 0.001, 0.999)  # Clamp p_gen to avoid numerical instability
        p_vocab_weighted = p_gen * p_vocab
        attention_weighted = (1 - p_gen) * attention_weights
        extended_size = vocab_size + max_oov
        extension = torch.zeros((batch_size, max_oov), dtype=torch.float, device=p_vocab.device)
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=1)
        attn_dist_extended = torch.zeros((batch_size, extended_size), dtype=torch.float, device=p_vocab.device)
        vocab_indices = enc_batch_extend_vocab.long()
        attn_flat_expanded = attention_weighted.view(batch_size, -1)
        attn_dist_extended.scatter_add_(1, vocab_indices, attn_flat_expanded)
        if p_vocab_extended.shape != attn_dist_extended.shape:
            raise ValueError("Shape mismatch in final distribution calculation.")
        final_distribution = p_vocab_extended + attn_dist_extended
        final_distribution /= final_distribution.sum(dim=1, keepdim=True)
        return final_distribution

    def temperature_top_k_sampling(self, logits, last_token_idx, k=10, temperature=1.0):
        """
        Apply temperature scaling and top-k filtering to logits before sampling the next token,
        explicitly preventing the immediate repetition of the last word.
       **
        temperature parameter affects how the model makes choices about the next word or token in a sequence.

        Args:
        logits (torch.Tensor): Raw logits from the model output.
        last_token_idx (int): Index of the last token that was generated.
        k (int): The number of highest probability logits to keep.
        temperature (float): Factor to apply to logits. Higher values increase randomness.

        Returns:
        torch.Tensor: The index of the next token.
        """
        if temperature != 1.0:
            # Scale logits by the specified temperature
            logits = logits / temperature

        # Ensure last_token_idx is an integer and within the valid range
        if last_token_idx is not None and last_token_idx >= 0:
            last_token_idx = int(last_token_idx)  # Convert to integer if not
            if last_token_idx < logits.size(-1):
                # Prevent the immediate repetition of the last word by setting its logit to negative infinity
                logits[0, last_token_idx] = float('-inf')

        # Apply top-k filtering if k is less than the size of logits
        if k < logits.size(-1):
            top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
            inf_mask = torch.full_like(logits, float('-inf'))
            logits = inf_mask.scatter_(dim=-1, index=top_k_indices, src=top_k_values)

        # Calculate probabilities using softmax
        probabilities = F.softmax(logits, dim=-1)

        # Sample from the probabilities to get the next token
        next_token = torch.multinomial(probabilities, num_samples=1)
        return next_token.squeeze()

    def greedy_search(self, enc_input, enc_input_ext, max_length, temperature=1.0, k=10):
        # Perform greedy search to generate text sequences
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            batch_size = enc_input.size(0)
            device = self.device
            enc_output, enc_hidden = self.encoder(enc_input)
            x_t = torch.full((batch_size, 1), fill_value=self.sos_token_id, dtype=torch.long, device=device)
            decoded_tokens = [[] for _ in range(batch_size)]
            finished = torch.zeros(batch_size, dtype=torch.bool).to(device)
            for t in range(max_length):
                p_vocab, dec_hidden, p_gen, attn = self.decoder(x_t, enc_hidden, enc_output)
                final_dist = self.get_final_distribution(enc_input_ext, p_gen, p_vocab, attn, self.max_oov,
                                                         self.vocab_size, batch_size)
                next_tokens = self.temperature_top_k_sampling(final_dist, temperature, k)
                next_tokens = next_tokens.squeeze() if next_tokens.ndim > 1 else next_tokens.unsqueeze(0)
                x_t = next_tokens.unsqueeze(1)  # Prepare for next iteration
                for i in range(batch_size):
                    if not finished[i]:
                        token_id = next_tokens[i].item()
                        decoded_tokens[i].append(token_id)
                        if token_id == self.eos_token_id:
                            finished[i] = True
                if finished.all():
                    break
            trimmed_tokens = [seq[:seq.index(self.eos_token_id) + 1] if self.eos_token_id in seq else seq for seq in decoded_tokens]
            return trimmed_tokens[0]
    # narrow down the model's choices to the k most likely next
    def top_k_sampling(self, logits, k=15, prev_tokens=None):
        # k number of samples to be drawn from the model
        # Apply top-k sampling strategy to logits
        if not isinstance(logits, torch.Tensor):
            raise ValueError("logits must be a torch.Tensor")
        if logits.numel() == 0:
            raise ValueError("logits tensor is empty")
        if k <= 0 or not isinstance(k, int):
            raise ValueError("k must be a positive integer")
        if prev_tokens is not None:
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[prev_tokens] = True
            logits = logits.masked_fill(mask, float('-inf'))
        k = min(k, logits.size(-1))
        # extracting the top from the topk
        top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
        kth_value = top_k_values[:, -1].unsqueeze(-1)
        mask = logits < kth_value
        filtered_logits = logits.masked_fill(mask, float('-inf'))
        probabilities = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        return next_token

    # Method for performing beam search to generate text sequences
    def beam_search(self, enc_input, enc_input_ext, beam_width, max_length, k=10):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            batch_size, seq_len = enc_input.size()
            enc_output, enc_hidden = self.encoder(enc_input)
            initial_hypothesis = BeamHypothesis(tokens=[self.sos_token_id], log_prob=0.0, hidden=enc_hidden)
            beams = [initial_hypothesis]
            for step in range(max_length):
                new_beams = []
                # Each current beam is expanded by decoding the latest token and updating the hidden state. The decoder outputs a vocabulary distribution (p_vocab), a new hidden state, a pointer generation probability (p_gen), and attention weights (attn).
                for beam in beams:
                    x_t = torch.tensor([beam.latest_token], device=self.device).unsqueeze(0)
                    p_vocab, hidden, p_gen, attn = self.decoder(x_t, beam.hidden, enc_output)
                    # The model computes the final probability distribution over the vocabulary, potentially augmented with an extended vocabulary for handling out-of-vocabulary (OOV) terms.
                    # It then samples k tokens from this distribution.
                    final_dist = self.get_final_distribution(enc_input_ext, p_gen, p_vocab, attn, self.max_oov,
                                                             self.vocab_size, batch_size)  # Assuming batch_size = 1
                    top_tokens = self.top_k_sampling(final_dist, k)  # Returns [batch_size, 1]
                    if top_tokens.ndim == 1:
                        top_tokens = top_tokens.unsqueeze(0)  # Fix dimension if necessary
                    # New hypotheses are formed based on the tokens sampled. Each new hypothesis includes the new token, the updated log probability, and the current hidden state.
                    for i in range(top_tokens.size(0)):
                        token = top_tokens[i].item()
                        new_log_prob = beam.log_prob + torch.log(final_dist[0, token])
                        new_beam = beam.extend(token=token, log_prob=new_log_prob, hidden=hidden)
                        new_beams.append(new_beam)
                # After expanding, the list of beams is pruned to keep only the top beam_width hypotheses based on their log probabilities. The loop breaks if all top beams end with the end-of-sequence token (self.eos_token_id), indicating complete sequences.
                beams = sorted(new_beams, key=lambda b: b.total_log_prob, reverse=True)[:beam_width]
                if all(beam.tokens[-1] == self.eos_token_id for beam in beams):
                    break
            best_beam = max(beams, key=lambda b: b.total_log_prob)
            return best_beam.tokens

# BeamHypothesis class used to manage individual hypotheses during beam search
class BeamHypothesis:
    """
    Represents a hypothesis in beam search, storing the sequence of tokens and their cumulative log probability.
    """
    def __init__(self, tokens, log_prob, hidden):
        self.tokens = tokens  # List of all tokens from start to current step
        self.log_prob = log_prob  # Cumulative log probability of the tokens
        self.hidden = hidden  # Hidden state corresponding to the last token

    def extend(self, token, log_prob, hidden):
        """
        Extends the current hypothesis with a new token, updating the cumulative log probability and hidden state.
        Args:
            token: The new token to add to the hypothesis.
            log_prob: The log probability of the new token.
            hidden: The new hidden state after adding the token.
        Returns:
            A new BeamHypothesis object with the updated state.
        """
        return BeamHypothesis(tokens=self.tokens + [token], log_prob=self.log_prob + log_prob, hidden=hidden)

    @property
    def latest_token(self):
        """
        Returns the most recently added token in the hypothesis.
        """
        return self.tokens[-1]

    @property
    def total_log_prob(self):
        """
        Returns the cumulative log probability of the hypothesis.
        """
        return self.log_prob
