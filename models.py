import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.tensor
import tensor_utils
from data_utils import MAIN_ENTITIES

"""
Provides classes LSTM_basic, UtteranceEmbedder, Attender, StaticEntitySimilarizer
"""

class LSTM_basic(nn.Module):

    def __init__(self, args, padding_idx=-1):
        """

        :param args: Namespace containing at least the parsed 'model' section of the config file.
        :param padding_idx: padding is used in several places for memory efficiency
        """

        super(LSTM_basic, self).__init__()

        self.padding_idx = padding_idx

        # Input layers:
        input_to_lstm = []
        # emb_reduction = [0,400] # TODO @Future: implement and add as config option.
        embedder = UtteranceEmbedder(args, padding_idx=padding_idx)
        input_to_lstm.append(embedder)
        if args.dropout_prob_1 > 0.0:       input_to_lstm.append(torch.nn.Dropout(p=args.dropout_prob_1))
        if args.nonlinearity_1 == 'tanh':   input_to_lstm.append(torch.nn.Tanh())
        elif args.nonlinearity_1 == 'relu': input_to_lstm.append(torch.nn.ReLU())
        elif args.nonlinearity_1 == 'sigmoid': input_to_lstm.append(torch.nn.Sigmoid())
        self.input_to_lstm = nn.Sequential(*input_to_lstm)

        # LSTM:
        self.lstm = nn.LSTM(embedder.embedding_dim,
                            args.hidden_lstm_1 // (2 if args.bidirectional else 1),
                            num_layers=args.layers_lstm,
                            batch_first=True,
                            bidirectional=args.bidirectional)

        # Apply attention over LSTM's outputs
        if isinstance(args.attention_lstm, str):
            self.attention_lstm = Attender(embedder.embedding_dim,
                                           self.lstm.hidden_size * (2 if args.bidirectional else 1),
                                           args.attention_lstm, args.nonlinearity_a,
                                            attention_window = args.attention_window,
                                           window_size = args.window_size)
        else:
            self.attention_lstm = None      # For easy checking in forward()

        # After LSTM:
        self.dropout2 = None
        if args.dropout_prob_2 > 0.0:
            self.dropout2 = torch.nn.Dropout(p=args.dropout_prob_2)

        self.to_class_scores = None
        self.to_entity = None
        if not args.entity_library:
            self.to_class_scores = nn.Linear(self.lstm.hidden_size * (2 if args.bidirectional else 1), args.num_entities)
        else:
            self.to_entity = nn.Linear(self.lstm.hidden_size * (2 if args.bidirectional else 1), embedder.speaker_emb.weight.data.shape[1])

        self.nonlinearity2 = None
        if args.nonlinearity_2 == 'tanh':
            self.nonlinearity2 = torch.nn.Tanh()
        elif args.nonlinearity_2 == 'relu':
            self.nonlinearity2 = torch.nn.ReLU()
        elif args.nonlinearity_2 == 'sigmoid':
            self.nonlinearity2 = torch.nn.Sigmoid()

        self.entity_library = None
        if args.entity_library:
            self.entity_library = LibraryFixedSize(args.num_entities, embedder.speaker_emb.weight.data.shape[1], args, embedder=embedder.speaker_emb if args.entlib_sharedinit else None)

        self.final_softmax = None
        if not args.gate_softmax:
            self.final_softmax = nn.LogSoftmax(dim=-1)

    def init_hidden(self, batch_size):
        """
        Resets the LSTM's hidden layer activations. To be called before applying the LSTM to a batch (not chunk).
        With batch_first True, the axes of the hidden layer are (minibatch_size, num_hidden_layers, hidden_dim).
        :param batch_size:
        """
        hidden1 = torch.zeros(self.lstm.num_layers * (2 if self.lstm.bidirectional else 1),
                              batch_size,
                              self.lstm.hidden_size)
        # Hidden state will be cuda if whatever first parameter is.
        if next(self.parameters()).is_cuda: hidden1 = hidden1.cuda()
        hidden2 = torch.zeros_like(hidden1)
        self.hidden = (autograd.Variable(hidden1), autograd.Variable(hidden2))
        if self.entity_library is not None:
            self.entity_library.init_activations(batch_size, use_cuda=next(self.parameters()).is_cuda)

    def detach_hidden(self):
        """
        This function is called to truncate backpropagation (e.g., at the start of each chunk).
        By wrapping the hidden states in a new variable it resets the grad_fn history for gradient computation.
        :return: nothing
        """
        self.hidden = tuple([autograd.Variable(hidden.data) for hidden in self.hidden])
        if self.entity_library is not None:
            self.entity_library.detach()

    def forward(self, padded_batch, desired_outputs_mask=None, desired_outputs=None):
        """
        Applies the model to a padded batch of chunked sequences.
        :param padded_batch: padded batch of shape batch_size x chunk_size x chunk_length.
        :param desired_outputs_mask: boolean mask (batch size x chunk size) of which rows in each
                chunk require an output (i.e., which are entity mentions).
        :param desired_outputs: target outputs for this batch (only used in case of supervised updating).
        :return: softmaxed scores, as a list of tensors of varying lengths if mask is given; padded tensor with all outputs otherwise.
        """
        # For backwards compatibility, used only for TESTING:
        if not isinstance(padded_batch, autograd.Variable):
            padded_batch = autograd.Variable(padded_batch)

        embeddings = self.input_to_lstm(padded_batch)

        # TODO @Future: At some point we should use pack_padded etc; then maybe nicer if input to forward() is a list of unpadded sequences.
        # embeddings = torch.nn.utils.rnn.pack_padded_sequence(embeddings, chunk_lengths, batch_first=True)
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        # lstm_out, lstm_out_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # If necessary apply attention to lstm's outputs:
        if self.attention_lstm is not None:
            lstm_out = self.attention_lstm(embeddings, lstm_out)

        # If a mask was given, continue only with the outputs of certain rows (per chunk):
        # TODO @Future: For the way the dynamic entity library works, a mask is no longer optional.
        if desired_outputs_mask is not None:
            # Empty mask means we can already stop right away.
            if desired_outputs_mask.sum() == 0:
                # Return list of num_chunk empty tensors.
                empty_tensor = torch.FloatTensor()
                if padded_batch.is_cuda: empty_tensor = empty_tensor.cuda()
                empty_tensor = autograd.Variable(empty_tensor)
                output = [empty_tensor.clone() for _ in padded_batch]
                return output
            # # Else (i.e., non-empty mask):            NOTE: This was before adding dynamic entity library.
            # lstm_out, lengths = tensor_utils.pack_batch_masked(lstm_out, desired_outputs_mask)
            lstm_out = tensor_utils.batch_masked_select(lstm_out, desired_outputs_mask, repad=True)

        if self.dropout2 is not None:
            lstm_out = self.dropout2(lstm_out)

        if self.to_class_scores is not None:
            lstm_out = self.to_class_scores(lstm_out)

        if self.to_entity is not None:
            lstm_out = self.to_entity(lstm_out)

        if self.nonlinearity2 is not None:
            lstm_out = self.nonlinearity2(lstm_out)

        if self.entity_library is not None:
            scores_of_sequence = []
            # Using transpose() to iterate over second dimension of the query batch (sequence steps):
            # NOTE: Strictly speaking this for-loop over time steps is necessary only for the dynamic case...
            #     ...but for the static case it helps avoid out of memory issues (which otherwise may occur with MLP/cos gates).
            if desired_outputs is not None:
                desired_outputs = tensor_utils.batch_masked_select(desired_outputs, desired_outputs_mask, repad=True)
                for i, (queries_step, targets_step) in enumerate(zip(lstm_out.transpose(0, 1), desired_outputs.transpose(0,1))):
                    scores, library = self.entity_library(queries_step, targets_step=targets_step)
                    scores_of_sequence.append(scores)
            else:
                for i, queries_step in enumerate(lstm_out.transpose(0, 1)):
                    scores, library = self.entity_library(queries_step)
                    scores_of_sequence.append(scores)

            lstm_out = torch.stack(scores_of_sequence, dim=1)  # Stack in sequence length dimension
            # B x L x N

        if self.final_softmax is not None:
            lstm_out = self.final_softmax(lstm_out)
        else:
            lstm_out = torch.log(lstm_out)      # Otherwise there was a softmax gate and only a log is needed.

        # If it was padded, make sure to unpad it afterwards (without repadding):
        if desired_outputs_mask is not None:
            lengths = torch.sum(desired_outputs_mask, dim=-1).long()
            # TODO The following is an ugly fix that serves to avoid empty tensor problems...
            empty_tensor = torch.Tensor()
            if lstm_out.is_cuda: empty_tensor = empty_tensor.cuda()
            empty_tensor = autograd.Variable(empty_tensor)
            lstm_out = [(lstm_out[i,0:lengths[i]] if lengths[i] > 0 else empty_tensor) for i in range(len(lstm_out))]

        return lstm_out


# TODO Delete this in published code.
class LSTM_basic_extract(LSTM_basic):

    def __init__(self, original_model, args, padding_idx=-1, zero_speaker=False):
        super(LSTM_basic_extract, self).__init__(args, padding_idx=-1)
        self.entity_library = None
        self.softmax = None
        if zero_speaker == True:
            self.input_to_lstm[0].speaker_emb.weight.data = self.input_to_lstm[0].speaker_emb.weight.data*0
        
    def forward(self, padded_batch, desired_outputs_mask=None):
        """
        Applies the model to a padded batch of chunked sequences.
        :param padded_batch: padded batch of shape batch_size x chunk_size x chunk_length.
        :param desired_outputs_mask: boolean mask (batch size x chunk size) of which rows in each
                chunk require an output (entity mention).
        :return: softmaxed scores, as a list of tensors of varying lengths if mask is given; padded tensor with all outputs otherwise.
        """
        # For backwards compatibility, used only for TESTING:
        if not isinstance(padded_batch, autograd.Variable):
            padded_batch = autograd.Variable(padded_batch)

        embeddings = self.input_to_lstm(padded_batch)

        # TODO @Future: At some point we should use pack_padded etc; then maybe nicer if input to forward() is a list of unpadded sequences.
        # embeddings = torch.nn.utils.rnn.pack_padded_sequence(embeddings, chunk_lengths, batch_first=True)
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        # lstm_out, lstm_out_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # If necessary apply attention to lstm's outputs:
        if self.attention_lstm is not None:
            lstm_out = self.attention_lstm(embeddings, lstm_out)

        #if self.dropout2 is not None:
        #    lstm_out = self.dropout2(lstm_out)

        #if self.to_class_scores is not None:
        #    lstm_out = self.to_class_scores(lstm_out)

        if self.to_entity is not None:
            lstm_out = self.to_entity(lstm_out)

        if self.nonlinearity2 is not None:
            lstm_out = self.nonlinearity2(lstm_out)

        if self.entity_library is not None:
            scores_of_sequence = []
            # Using transpose() to iterate over second dimension of the query batch (sequence steps):
            # NOTE: Strictly speaking this for-loop over time steps is necessary only for the dynamic case...
            #     ...but for the static case it helps avoid out of memory issues (which otherwise may occur with MLP/cos gates).
            for i, queries_step in enumerate(lstm_out.transpose(0, 1)):
                scores, library = self.entity_library(queries_step)
                scores_of_sequence.append(scores)

            lstm_out = torch.stack(scores_of_sequence, dim=1)  # Stack in sequence length dimension
            # B x L x N

        return lstm_out

class LSTM_basic_extract_semeval(LSTM_basic):

    def __init__(self, original_model, args, padding_idx=-1):
        super(LSTM_basic_extract, self).__init__(args, padding_idx=-1)
        print(original_model)
        self.lstm_to_output = nn.Sequential(*list(original_model.lstm_to_output.children())[:-2])

    def forward(self, padded_batch, desired_outputs_mask=None):
        """
        Applies the model to a padded batch of chunked sequences.
        :param padded_batch: padded batch of shape batch_size x chunk_size x chunk_length.
        :param desired_outputs_mask: boolean mask (batch size x chunk size) of which rows in each
                chunk require an output (entity mention).
        :return: softmaxed scores, as a list of tensors of varying lengths if mask is given; padded tensor with all outputs otherwise.
        """
        # TODO: note that desired_outputs_mask is not used (yet) -- forward actually does not need to be defined again.
        # For backwards compatibility, used only for TESTING:
        if not isinstance(padded_batch, autograd.Variable):
            padded_batch = autograd.Variable(padded_batch)

        embeddings = self.input_to_lstm(padded_batch)

        # TODO At some point we may want to uncomment these 2 lines (but now gives error). And if we're going to use pack_padded etc, then maybe nicer if input to forward() is a list of sequences, unpadded etc.
        # embeddings = torch.nn.utils.rnn.pack_padded_sequence(embeddings, chunk_lengths, batch_first=True)
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        # lstm_out, lstm_out_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # If necessary apply attention to lstm's outputs:
        if self.attention_lstm is not None:
            lstm_out = self.attention_lstm(embeddings, lstm_out)

        output = self.lstm_to_output(lstm_out)
        return output


class UtteranceEmbedder(nn.Module):

    def __init__(self, args, padding_idx=-1):
        super(UtteranceEmbedder, self).__init__()

        self.padding_idx = padding_idx

        input_dims = [args.vocabulary_size, args.num_entities]
        embeddings_specs = [args.token_emb, args.speaker_emb]

        # Token embeddings, either pre-trained or random weights
        if isinstance(embeddings_specs[0], np.ndarray):
            emb_weights = torch.Tensor(embeddings_specs[0])
            self.token_emb = nn.Embedding(emb_weights.shape[0], emb_weights.shape[1])
            self.token_emb.weight.data = emb_weights
        else:
            self.token_emb = nn.Embedding(input_dims[0], embeddings_specs[0])

        # Speaker embeddings, either pre-trained or random weights
        if isinstance(embeddings_specs[1], np.ndarray):
            emb_weights = torch.Tensor(embeddings_specs[1])
            self.speaker_emb = nn.EmbeddingBag(emb_weights.shape[0], emb_weights.shape[1])
            self.speaker_emb.weight.data = emb_weights
        else:
            self.speaker_emb = nn.EmbeddingBag(input_dims[1], embeddings_specs[1], mode='sum')

        self.embedding_dim = self.token_emb.embedding_dim + self.speaker_emb.embedding_dim

    def forward(self, padded_batch):

        # Compute embeddings of input
        token_embeddings = self._padded_embedding(self.token_emb, padded_batch[:, :, 0])
        speaker_embeddings = self._padded_embedding(self.speaker_emb, padded_batch[:, :, 1:])

        embeddings = torch.cat((token_embeddings, speaker_embeddings), 2)

        return embeddings

    def _padded_embedding(self, emb, ids):
        """
        This wrapper gets rid of padding indices prior to computing embeddings, returning all-zeros.
        This was necessary because PyTorch's Embedding cannot (yet) handle padding_idx = -1, and because
        EmbeddingBag cannot (yet?) handle padding_idx at all.
        :param emb: Embedding or EmbeddingBag
        :param ids: BxS (for Embedding) or BxSxN (for EmbeddingBag)
        :return:
        """
        # First remove all padding indices to obtain the actual ids
        mask = ids != self.padding_idx
        actual_ids = torch.masked_select(ids, mask)

        # Prepare for, and feed through, EmbeddingBag (3D) or Embedding (2D)
        if len(ids.shape) == 3:
            # If 3D tensor of indices, prepare them for EmbeddingBag
            sum_ids = mask.long().sum(2)
            sum_ids = torch.masked_select(sum_ids, mask[:, :, 0])
            cumsum_speakers = sum_ids.cumsum(0)
            offsets = torch.zeros_like(cumsum_speakers)
            if len(offsets) > 1:  # Necessary to avoid slice yielding empty tensor, which apparently is forbidden.
                offsets[1:] = cumsum_speakers[:-1]
            # Now that they're a long 1D tensor, compute their embeddings
            actual_embs = emb(actual_ids, offsets)
            # Compute a mask to put them back together in a new tensor below
            embedding_mask = mask[:, :, 0].contiguous().view(
                ids.shape[0], ids.shape[1], 1).expand(-1, -1, emb.weight.shape[1])
        else:
            # Else, assuming 2D tensor of indices, feed them through plain Embedding
            actual_embs = emb(actual_ids)
            # Compute a mask to put the results back together in a new tensor below
            embedding_mask = mask.view(ids.shape[0], ids.shape[1], 1).expand(-1, -1, emb.weight.shape[1])

        # Tensor that will hold the embeddings (batch size x chunk length x embedding dim) amidst zeros
        embeddings = torch.zeros(ids.shape[0], ids.shape[1], emb.weight.shape[1])
        if ids.is_cuda: embeddings = embeddings.cuda()
        embeddings = autograd.Variable(embeddings)
        # Scatter the computed embeddings into the right places in the new tensor
        embeddings.masked_scatter_(embedding_mask, actual_embs)

        return embeddings



class LibraryFixedSize(nn.Module):

    def __init__(self, library_size, slot_dim, settings, embedder=None):
        super(LibraryFixedSize, self).__init__()

        self.size = library_size
        self.slot_dim = slot_dim
        self.use_keys = settings.entlib_key
        self.dynamic = (settings.entity_library == "dynamic")
        self.gate_type = settings.gate_type
        self.normalization = settings.entlib_normalization
        self.value_weights = settings.entlib_value_weights
        self.values = None     # Will contain activations

        # Initial library activations (i.e., after calling reset_activations())
        if not settings.entlib_weights:
            # Ininialize as zeroes every new sequence; not trained weights.
            self.weights = None
        else:
            if embedder is None:
                # Initialize to an initially random embedding, trained with backprop.
                self.weights = nn.EmbeddingBag(library_size, slot_dim, mode='sum')
            elif settings.entlib_shared:
                # Initialize to an existing embedding; trained with weight sharing.
                self.weights = embedder
            else:
                # Initialize to a clone of an existing embedding; trained separately.
                self.weights = nn.EmbeddingBag(library_size, slot_dim, mode='sum')
                self.weights.weight.data = embedder.weight.data.clone()

        # Gates compute weights over the entity library given the entity query:
        if settings.gate_type == 'mlp':
            if self.use_keys:
                self.gate = MLPGate(settings.gate_mlp_hidden, settings.gate_nonlinearity, slot_dim, slot_dim, slot_dim)
            else:
                self.gate = MLPGate(settings.gate_mlp_hidden, settings.gate_nonlinearity, slot_dim, slot_dim)
        else: # similarity-based gate instead:
            self.gate = SimGate(settings.gate_type, settings.gate_nonlinearity, settings.gate_sum_keys_values)

        # Optionally take a 'global' perspective (followed by (log)softmax):
        self.gate_keeper = None
        self.gate_softmax = None
        if settings.gate_softmax:
            self.gate_keeper = nn.Linear(self.size, self.size)
            self.gate_softmax = nn.Softmax(dim=-1)

        # Components for computing information to update dynamic library with:
        if self.dynamic:
            self.nonlin = nn.PReLU()
            self.value_linear = nn.Linear(self.slot_dim, self.slot_dim)
            self.query_linear = nn.Linear(self.slot_dim, self.slot_dim)
            if self.use_keys:
                self.key_linear = nn.Linear(self.slot_dim, self.slot_dim)

    def init_activations(self, batch_size, use_cuda):
        """
        Resets activations (not the weights to which it is initialized (if any)).
        :param batch_size:
        :return:
        """
        # Initialize either with zeros vector or with weights:
        if not self.value_weights:   # may be either false (if weights but not used as value inits) or None (if no weights at all)
            self.values = torch.zeros(self.size, self.slot_dim)
            if use_cuda:
                self.values = self.values.cuda()
            self.values = autograd.Variable(self.values)
        else:
            self.values = self.weights.weight.clone()
        # Expand values to batch size:
        self.values = self.values.unsqueeze(dim=0).expand((batch_size, self.size, self.slot_dim))

    def detach(self):
        """
        This function is called to truncate backpropagation (e.g., at the start of each chunk).
        By wrapping the param in a new variable it resets the grad_fn history for gradient computation.
        :return: nothing
        """
        self.values = autograd.Variable(self.values.data)

    def forward(self, queries_step, targets_step=None):
        """
        Takes in a batch of queries B x D for one timestep; outputs gate values and new state of library.
        :param queries_step: a batch (size B) of queries (query dimension D), for one token per chunk/sequence.
        :param targets_step: if given, will update library based on targets, not based on gates (supervised updating).
        :return: gates and values
        """
        if self.use_keys:
            # TODO @Ionut: (Plz delete when you read this.) This is what I meant: you don't need to clone the ...
            #    ... weights in init_activations() in order for them to be used as keys.
            gates = self.gate(self.values, queries_step, keys=self.weights.weight)
            # B x N               B x N x D     B x D           N x D
        else:
            gates = self.gate(self.values, queries_step)
            # B x N               B x N x D     B x D

        if self.gate_softmax is not None:
            gates = self.gate_keeper(gates)
            gates = self.gate_softmax(gates)
            # B x N

        if self.dynamic:

            # Piece of code to enable printing out the values of the library, for the first scene:
            if False:
                with open('values.txt', 'a', encoding='utf-8') as file:
                    for i, scene_values in enumerate(self.values.data):
                        if i == 0:  # temporarily restrict to scene 0
                            for j, entity_values in enumerate(scene_values):
                                if j in MAIN_ENTITIES:
                                    print('{0},{1},{2}'.format(i,j,','.join([str(e) for e in entity_values])), file=file)
                file.close()

            # TODO We may want some config settings to play around with here.
            new_info = self.query_linear(queries_step.unsqueeze(-2)) + self.value_linear(self.values)
            # B x N x D              B x 1 x D  +  B x N x D
            if self.use_keys:
                new_info = new_info + self.key_linear(self.weights.weight)
                # B x N x D  =   B x N x D  +  N x D
            # new_info = queries_step.unsqueeze(-2)          # TODO This is much simplified.
            new_info = self.nonlin(new_info)

            gates_unsqueezed = gates.unsqueeze(-1)
            # B x N x 1   <- B x N

            # In case to use targets instead of actual gate value to update library
            if targets_step is not None:
                target_gates = torch.zeros(gates.size())
                target_gates = target_gates.cuda()
                # print(targets_step.size(), target_gates.size())
                for i in range(len(target_gates)):
                    target = targets_step[i].cpu().data[0]      # TODO My goodness, ugly stuff.
                    # print(i, target)
                    if target >= 0:
                        target_gates[i][target] = 1
                gates_unsqueezed = autograd.Variable(target_gates.unsqueeze(-1))

            self.values = self.values + gates_unsqueezed * new_info
            # B x N x D      B x N x D         B x N x 1 * B x N x D

            if self.normalization:
                self.values = F.normalize(self.values, p = 2, dim = -1)

        # Output gate and library state
        return gates, self.values


class SimGate(nn.Module):
    """
    Similarity module that takes entity library and queries (B x D) and returns a weight for each cell (B x N).
    """

    def __init__(self, sim_type='dot', nonlinearity='sigmoid', sum_keys_values=False):
        super(SimGate, self).__init__()

        self.sum_keys_values = sum_keys_values

        self.nonlin = None
        if nonlinearity == 'sigmoid':
            self.nonlin = nn.Sigmoid()
        elif nonlinearity == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlinearity == 'tanh':
            self.nonlin = nn.Tanh()

        if sim_type == 'dot':
            self.similarity = lambda keys, queries: torch.matmul(keys, queries.unsqueeze(-1)).squeeze(dim=-1)
            # B x N (x 1)                                     B x N x D  dot  B x D (x 1)
        elif sim_type == 'cos':
            cos = nn.CosineSimilarity(dim=-1)
            self.similarity = lambda keys, queries: cos(keys, queries.unsqueeze(-2))

    def forward(self, library, queries, keys=None):

        if keys is None:
            out = self.similarity(library, queries)
        elif self.sum_keys_values:
            keys_plus_values = keys + library
            out = self.similarity(keys_plus_values, queries)
        else:
            out = self.similarity(keys, queries) + self.similarity(library, queries)


        if self.nonlin is not None:
            out = self.nonlin(out)

        return out


class MLPGate(nn.Module):
    """
    Multi-Layer Perceptron that takes keys (B x N x D) and queries (B x D) and returns a weight for each key (B x N).
    """

    def __init__(self, hidden_size, nonlinearity, query_dim, library_dim=None, key_dim=None):
        super(MLPGate, self).__init__()

        self.nonlin = None
        if nonlinearity == 'sigmoid':
            self.nonlin = nn.Sigmoid()
        elif nonlinearity == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlinearity == 'tanh':
            self.nonlin = nn.Tanh()

        self.entlib_key = False
        if key_dim is None:
            self.layer1 = nn.Linear(query_dim + library_dim, hidden_size)
        else:
            self.layer1 = nn.Linear(query_dim + library_dim + key_dim, hidden_size)

        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, library, queries, keys=None):
                    # B x N x D    B x D

        if keys is None:
            cat_query_library = torch.cat([library, queries.unsqueeze(-2).expand(library.size())], dim=-1)
             # B x N x 2D =  B x N x D  concat B x 1(N) x D
        else:
            cat_query_library = torch.cat([library, keys.unsqueeze(0).expand(library.size()), queries.unsqueeze(-2).expand(library.size())], dim=-1)

        hidden = self.layer1(cat_query_library)
        # B x N x H  <- B x N x 2D
        if self.nonlin is not None:
            hidden = self.nonlin(hidden)

        out = self.layer2(hidden)
        # B x N x 1       <- B x N x H
        if self.nonlin is not None:
            out = self.nonlin(out)

        out = out.squeeze(dim=-1)
        # B x N

        return out


class Attender(nn.Module):
    # TODO To avoid redundancy, try to reuse the Gate modules defined for the entity library.

    def __init__(self, query_dim, key_dim, attention_type, nonlinearity,
                 attention_window = True, window_size = 20, max_chunk_size=999):
        super(Attender, self).__init__()
        self.attention_type = attention_type

        self.attention_window = None
        if attention_window:
            self.attention_window = self._get_attention_tensor_with_window_size(window_size, max_chunk_size)

        if self.attention_type == 'feedforward':
            if nonlinearity == 'tanh':
                nonlin_activation = torch.nn.Tanh()
                self.attention_layer = nn.Sequential(nn.Linear(query_dim + key_dim, 1),
                                                     nonlin_activation)
            elif nonlinearity == 'relu':
                nonlin_activation = torch.nn.ReLU()
                self.attention_layer = nn.Sequential(nn.Linear(query_dim + key_dim, 1),
                                                 nonlin_activation)
            else:
                self.attention_layer = nn.Linear(query_dim + key_dim, 1)

        elif self.attention_type == 'dot':
            if key_dim >= query_dim:
                self.reduce = nn.Linear(key_dim, query_dim)
                self.match_dims = lambda queries, keys: (queries, self.reduce(keys))
            else:
                self.reduce = nn.Linear(query_dim, key_dim)
                self.match_dims = lambda queries, keys: (self.reduce(queries), keys)

    def forward(self, queries, keys, values=None):

        debug = False

        # If no separate values are given, do self-attention:
        if values == None:
            values = keys
        if debug: print("IN: Queries:", queries.shape, "Keys:", keys.shape)

        chunk_size = keys.size()[1]
        batch_size = keys.size()[0]

        if self.attention_type == 'feedforward':
            similarities = torch.Tensor(batch_size, chunk_size, chunk_size)
            if queries.is_cuda: similarities = similarities.cuda()
            similarities = autograd.Variable(similarities)
            # Compute similarities one chunk at a time, otherwise we risk out of memory error :(
            for i in range(0, batch_size, 1):
                some_queries = queries[i:i+1].unsqueeze(2).expand(-1, -1, chunk_size, -1)
                some_keys = keys[i:i+1].unsqueeze(1).expand(-1, chunk_size, -1, -1)
                pairs = torch.cat((some_queries, some_keys), dim=-1)
                similarities[i:i+1] = self.attention_layer(pairs).view(1, chunk_size, chunk_size)
        elif self.attention_type == 'dot':
            queries, keys = self.match_dims(queries, keys)
            similarities = torch.bmm(queries, keys.transpose(-2,-1))
            #                        Bx(C1xD) @  Bx(DxC2)     = Bx(C1xC2)

        if self.attention_window is not None:
            if not self.attention_window.is_cuda and queries.is_cuda:
                # Will execute just once:
                self.attention_window = self.attention_window.cuda()
            similarities = similarities + self.attention_window[0:chunk_size, 0:chunk_size]

        # For every query (C1), similarities to all its keys (C2) must sum to 1 -- that's the last axis (-1)
        weights = F.softmax(similarities, dim=-1)

        # Multiply the values by the weights
        weighted_values = torch.bmm(weights, values)
        #                          Bx(C1xC2) @ Bx(C2xD) = Bx(C1xD)

        return weighted_values

    def _get_attention_tensor_with_window_size(self, window_size, chunk_size):
        """
        TODO @Future: This can be simplified to 2 or 3 lines of code.
        Computes something like 11000000
                                11100000
                                01110000
                                00111000
                                00011100
                                00001110
                                00000111
                                00000011
        :param window_size:
        :param chunk_size:
        :return:
        """
        dim_inner = chunk_size - 2 * window_size

        # Construct left part of tensor
        left_values = torch.Tensor([])
        number_zeros = window_size
        number_negatives = chunk_size - window_size
        for i in range(1, window_size + 1):
            number_zeros += 1
            number_negatives -= 1
            left_values = torch.cat((left_values, torch.zeros(number_zeros),
                                     torch.from_numpy(np.array([-10 ** (8) for i in range(number_negatives)])).float()), dim=0)

        # Construct inner part of tensor
        number_zeros = 2 * window_size + 1
        number_negatives = chunk_size - number_zeros + 1
        column_inner = torch.cat((torch.zeros(number_zeros),
                                  torch.from_numpy(np.array([-10**(8) for i in range(number_negatives)])).float()), dim=0)
        inner_values = column_inner.unsqueeze(0).expand(dim_inner, -1).contiguous().view(-1)[:-number_negatives]

        # Construct right part of tensor
        right_values = torch.from_numpy(np.flip(left_values.numpy(), 0).copy())

        # Put everything together
        values_vector = torch.cat((left_values, inner_values, right_values), dim=0)
        matrix_values = values_vector.contiguous().view(chunk_size, chunk_size)

        matrix_values = autograd.Variable(matrix_values, requires_grad=False)

        return matrix_values
