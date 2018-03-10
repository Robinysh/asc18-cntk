import cntk as C
import numpy as np
from helpers import *
import pickle
import importlib
import os

class PolyMath:
    def __init__(self, config_file):
        data_config = importlib.import_module(config_file).data_config
        model_config = importlib.import_module(config_file).model_config

        self.word_count_threshold = data_config['word_count_threshold']
        self.char_count_threshold = data_config['char_count_threshold']
        self.word_size = data_config['word_size']
        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_file = os.path.join(self.abs_path, data_config['pickle_file'])

        with open(pickle_file, 'rb') as vf:
            known, self.vocab, self.chars = pickle.load(vf) 
        self.wg_dim = known
        self.vocab_size = len(self.vocab)
        self.wn_dim = self.vocab_size - known
        self.c_dim = len(self.chars)
        self.a_dim = 1
        self.hidden_dim = model_config['hidden_dim']
        self.num_layers = model_config['num_layers']
        self.attention_dim = model_config['attention_dim']
        self.convs = model_config['char_convs']
        self.dropout = model_config['dropout']
        self.char_emb_dim = model_config['char_emb_dim']
        self.highway_layers = model_config['highway_layers']
        self.two_step = model_config['two_step']
        self.use_cudnn = model_config['use_cudnn']
        self.pointer_importance = model_config['pointer_importance']
        self.use_sparse = False
       
        self.sentence_start = C.one_hot(self.vocab_size, self.vocab_size+1, sparse_output=self.use_sparse) 
        self.sentence_end_index = self.vocab['</s>']
        self.sentence_max_length = 0.1
        print('vocab size',self.vocab_size)
        print('dropout', self.dropout)
        print('use_cudnn', self.use_cudnn)
        print('use_sparse', self.use_sparse)

    def charcnn(self, x):
        conv_out = C.layers.Sequential([
            C.layers.Embedding(self.char_emb_dim),
            C.layers.Dropout(self.dropout),
            C.layers.Convolution2D((5,self.char_emb_dim), self.convs, activation=C.relu, init=C.glorot_uniform(), bias=True, init_bias=0, name='charcnn_conv')])(x)
        return C.reduce_max(conv_out, axis=1) # workaround cudnn failure in GlobalMaxPooling

    def embed(self):
        # load glove
        npglove = np.zeros((self.wg_dim, self.hidden_dim), dtype=np.float32)
        with open(os.path.join(self.abs_path, 'glove.6B.100d.txt'), encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0].lower()
                if word in self.vocab:
                    npglove[self.vocab[word],:] = np.asarray([float(p) for p in parts[1:]])
        glove = C.constant(npglove)
        nonglove = C.parameter(shape=(self.vocab_size - self.wg_dim, self.hidden_dim), init=C.glorot_uniform(), name='TrainableE')
        
        def func(wg, wn):
            return C.times(wg, glove) + C.times(wn, nonglove)
        return func

    def input_layer(self,cgw,cnw,cc,qgw,qnw,qc):
        cgw_ph = C.placeholder()
        cnw_ph = C.placeholder()
        cc_ph  = C.placeholder()
        qgw_ph = C.placeholder()
        qnw_ph = C.placeholder()
        qc_ph  = C.placeholder()
   #     agw_ph = C.placeholder()
   #     anw_ph = C.placeholder()
   #     ac_ph  = C.placeholder()
        
        input_chars = C.placeholder(shape=(1,self.word_size,self.c_dim))
        input_glove_words = C.placeholder(shape=(self.wg_dim,))
        input_nonglove_words = C.placeholder(shape=(self.wn_dim,))

        # we need to reshape because GlobalMaxPooling/reduce_max is retaining a trailing singleton dimension
        # todo GlobalPooling/reduce_max should have a keepdims default to False embedded = C.splice(
        embedded = C.splice(
            C.reshape(self.charcnn(input_chars), self.convs),
            self.embed()(input_glove_words, input_nonglove_words), name='splice_embed')
        highway = HighwayNetwork(dim=2*self.hidden_dim, highway_layers=self.highway_layers)(embedded)
        highway_drop = C.layers.Dropout(self.dropout)(highway)
        processed = OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='input_rnn')(highway_drop)
        
        qce = C.one_hot(qc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
        cce = C.one_hot(cc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
#        ace = C.one_hot(ac_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
                
        q_processed = processed.clone(C.CloneMethod.share, {input_chars:qce, input_glove_words:qgw_ph, input_nonglove_words:qnw_ph})
        c_processed = processed.clone(C.CloneMethod.share, {input_chars:cce, input_glove_words:cgw_ph, input_nonglove_words:cnw_ph})

        return C.as_block(
            C.combine([c_processed, q_processed]),
            [(cgw_ph, cgw),(cnw_ph, cnw),(cc_ph, cc),(qgw_ph, qgw),(qnw_ph, qnw),(qc_ph, qc)],
            'input_layer',
            'input_layer')
        
    def attention_layer(self, context, query):
        q_processed = C.placeholder(shape=(2*self.hidden_dim,))
        c_processed = C.placeholder(shape=(2*self.hidden_dim,))

        #convert query's sequence axis to static
        qvw, qvw_mask = C.sequence.unpack(q_processed, padding_value=0).outputs

        # This part deserves some explanation
        # It is the attention layer
        # In the paper they use a 6 * dim dimensional vector
        # here we split it in three parts because the different parts
        # participate in very different operations
        # so W * [h; u; h.* u] becomes w1 * h + w2 * u + w3 * (h.*u)
        ws1 = C.parameter(shape=(2 * self.hidden_dim, 1), init=C.glorot_uniform())
        ws2 = C.parameter(shape=(2 * self.hidden_dim, 1), init=C.glorot_uniform())
        ws3 = C.parameter(shape=(1, 2 * self.hidden_dim), init=C.glorot_uniform())
        att_bias = C.parameter(shape=(), init=0)

        wh = C.times (c_processed, ws1)
        wu = C.reshape(C.times (qvw, ws2), (-1,))
        whu = C.reshape(C.reduce_sum(c_processed * C.sequence.broadcast_as(qvw * ws3, c_processed), axis=1), (-1,))
        S = wh + whu + C.sequence.broadcast_as(wu, c_processed) + att_bias
        # mask out values outside of Query, and fill in gaps with -1e+30 as neutral value for both reduce_log_sum_exp and reduce_max
        qvw_mask_expanded = C.sequence.broadcast_as(qvw_mask, c_processed)
        S = C.element_select(qvw_mask_expanded, S, C.constant(-1e+30))
        q_attn = C.reshape(C.softmax(S), (-1,1))
        c2q = C.reshape(C.reduce_sum(C.sequence.broadcast_as(qvw, q_attn) * q_attn, axis=0),(-1))
        
        max_col = C.reduce_max(S)
        c_attn = C.sequence.softmax(max_col)

        htilde = C.sequence.reduce_sum(c_processed * c_attn)
        q2c = C.sequence.broadcast_as(htilde, c_processed)
        q2c_out = c_processed * q2c

        att_context = C.splice(c_processed, c2q, c_processed * c2q, q2c_out)

        return C.as_block(
            att_context,
            [(c_processed, context), (q_processed, query)],
            'attention_layer',
            'attention_layer')
            
    def modeling_layer(self, attention_context):
        att_context = C.placeholder(shape=(8*self.hidden_dim,))

        #modeling layer
        mod_context = C.layers.Sequential([
            C.layers.Dropout(self.dropout),
            OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='model_rnn0'),
            C.layers.Dropout(self.dropout),
            OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='model_rnn1'),
])(att_context)

        return C.as_block(
            mod_context,
            [(att_context, attention_context)],
            'modeling_layer',
            'modeling_layer')

    def output_layer(self, attention_context, model_context, aw, q_processed, c_processed):
        att_context = C.placeholder(shape=(8*self.hidden_dim,))
        query_processed = C.placeholder(shape=(2*self.hidden_dim,))
        context_processed = C.placeholder(shape=(2*self.hidden_dim,))
        mod_context = C.placeholder(shape=(2*self.hidden_dim))
        a_onehot = C.placeholder(shape=(self.vocab_size+1,))

        start_logits = C.layers.Dense(1, name='out_start')(C.dropout(C.splice(mod_context, att_context), self.dropout))
        start_hardmax = seq_hardmax(start_logits)
        att_mod_ctx = C.sequence.last(C.sequence.gather(mod_context, start_hardmax))
        att_mod_ctx_expanded = C.sequence.broadcast_as(att_mod_ctx, att_context)
        end_input = C.splice(att_context, mod_context, att_mod_ctx_expanded, mod_context * att_mod_ctx_expanded)
        m2 = OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='output_rnn')(end_input)
        end_logits = C.layers.Dense(1, name='out_end')(C.dropout(C.splice(m2, att_context), self.dropout))

        start_flag = C.hardmax(start_logits)
        end_flag = C.hardmax(end_logits)
     
        def create_model():
            # Encoder: (input*) --> (h0, c0)
            # Create multiple layers of LSTMs by passing the output of the i-th layer
            # to the (i+1)th layer as its input
            with C.layers.default_options(enable_self_stabilization=True, go_backwards=False):
                LastRecurrence = C.layers.Recurrence
                encode = C.layers.Sequential([
                    C.layers.Stabilizer(),
                    C.layers.For(range(self.num_layers-1), lambda:
                        C.layers.Recurrence(C.layers.LSTM(2*self.hidden_dim))),
                    OptimizedRnnStack(self.hidden_dim, return_full_state=True),
                ])

                encode_c = C.layers.Sequential([
                    C.layers.Stabilizer(),
                    C.layers.For(range(self.num_layers-1), lambda:
                        C.layers.Recurrence(C.layers.LSTM(2*self.hidden_dim+2))),
                    OptimizedRnnStack(self.hidden_dim, return_full_state=True),
                ])
            
            # Decoder: (history*, input*) --> unnormalized_word_logp*
            # where history is one of these, delayed by 1 step and <s> prepended:
            #  - training: labels
            #  - testing:  its own output hardmax(z) (greedy decoder)
            with C.layers.default_options(enable_self_stabilization=True):
                # sub-layers
                stab_in = C.layers.Stabilizer()
                rec_blocks = [C.layers.LSTM(self.hidden_dim) for i in range(self.num_layers)]
                stab_out = C.layers.Stabilizer()
                proj_out = C.layers.Dense(self.vocab_size+1, name='out_proj')
                # attention model
                attention_model = C.layers.AttentionModel(self.attention_dim, 
                                                              name='attention_model') # :: (h_enc*, h_dec) -> (h_dec augmented)
                hstate_dense = C.layers.Dense(self.hidden_dim, activation=C.tanh, input_rank=1)
                cstate_dense = C.layers.Dense(self.hidden_dim, activation=C.tanh, input_rank=1)
                # layer function
                @C.Function
                def decode(history, q, c, start_logits, end_logits):
                    q = encode(q)
                    c = encode_c(C.splice(c, start_logits, end_logits, axis=0))
                    r = history
                    r = stab_in(r)

                    q_last_h = C.sequence.last(q.outputs[0])
                    q_last_c = C.sequence.last(q.outputs[1])
                    c_last_h = C.sequence.last(c.outputs[0])
                    c_last_c = C.sequence.last(c.outputs[1])
                    initial_hstate = hstate_dense(C.splice(q_last_h, c_last_h))
                    initial_cstate = cstate_dense(C.splice(q_last_c, c_last_c))

                    for i in range(self.num_layers):
                        rec_block = rec_blocks[i]   # LSTM(hidden_dim)  # :: (dh, dc, x) -> (h, c)
                        if i == 0:
                            @C.Function
                            def lstm_with_attention(dh, dc, x):
                                h_att = attention_model(c.outputs[0], dh)
                                q_att = attention_model(q.outputs[0], dh)
                                att = C.splice(h_att, q_att)
                                x = C.splice(x, att)
                                return rec_block(dh, dc, x)
                            r = C.layers.RecurrenceFrom(lstm_with_attention)(initial_hstate, initial_cstate, r)
                        else:
                            r = C.layers.RecurrenceFrom(rec_block)(initial_hstate, initial_cstate, r)
                    r = stab_out(r)
                    r = proj_out(r)
                    r = C.layers.Label('out_proj_out')(r)
                    return r
            return decode

        def create_model_train(s2smodel):
            # model used in training (history is known from labels)
            # note: the labels must NOT contain the initial <s>
            @C.Function
            def model_train(labels, q, c, start_logits, end_logits): # (input*, labels*) --> (word_logp*)

                # The input to the decoder always starts with the special label sequence start token.
                # Then, use the previous value of the label sequence (for training) or the output (for execution).
                past_labels = C.layers.Delay(initial_state=self.sentence_start)(labels)
                return s2smodel(past_labels, q, c, start_logits, end_logits)
            return model_train

        def create_model_greedy(s2smodel):
            # model used in (greedy) decoding (inferencing) (history is decoder's own output)
            @C.Function
            def model_greedy(q, c, start_logits, end_logits): # (input*) --> (word_sequence*)
                # Decoding is an unfold() operation starting from sentence_start.
                # We must transform s2smodel (history*, input* -> word_logp*) into a generator (history* -> output*)
                # which holds 'input' in its closure.
                unfold = C.layers.UnfoldFrom(\
                                    lambda history: s2smodel(history, q, c, start_logits, end_logits) >> C.hardmax,
                                    # stop once sentence_end_index was max-scoring output
                                    until_predicate=lambda w: w[...,self.sentence_end_index],
                                    length_increase=self.sentence_max_length)
                return unfold(initial_state=self.sentence_start, dynamic_axes_like=c)
            return model_greedy
        
        s2smodel = create_model()
      
        model_train = create_model_train(s2smodel)(a_onehot, query_processed, context_processed, start_logits, end_logits)
        model_greed = create_model_greedy(s2smodel)(query_processed, context_processed, start_logits, end_logits)
        model_greedy = C.argmax(model_greed,0)

        return C.as_block(
            C.combine((model_train, model_greedy, start_logits, end_logits)),
            [(att_context, attention_context), (mod_context, model_context), (a_onehot, aw), (query_processed, q_processed), (context_processed, c_processed)],
            'attention_layer',
            'attention_layer')

    def create_criterion_function(self):
        @C.Function
        def criterion(input, labels):
            # criterion function must drop the <s> from the labels
            #postprocessed_labels = C.sequence.slice(labels, 1, 0) # <s> A B C </s> --> A B C </s>
            ce = C.cross_entropy_with_softmax(input, labels, name='loss')

            errs = C.classification_error(input, labels, name='NumOfDiffWords')
            return (ce, errs)

        return criterion

    def model(self):
        c = C.Axis.new_unique_dynamic_axis('c')
        q = C.Axis.new_unique_dynamic_axis('q')
        a = C.Axis.new_unique_dynamic_axis('a')
        b = C.Axis.default_batch_axis()
        cgw = C.input_variable(self.wg_dim, dynamic_axes=[b,c], is_sparse=True, name='cgw')
        cnw = C.input_variable(self.wn_dim, dynamic_axes=[b,c], is_sparse=True, name='cnw')
        qgw = C.input_variable(self.wg_dim, dynamic_axes=[b,q], is_sparse=True, name='qgw')
        qnw = C.input_variable(self.wn_dim, dynamic_axes=[b,q], is_sparse=True, name='qnw')
        aw = C.input_variable(self.vocab_size+1, dynamic_axes=[b,a], is_sparse=False, name='aw')
        cc = C.input_variable((1,self.word_size), dynamic_axes=[b,c], name='cc')
        qc = C.input_variable((1,self.word_size), dynamic_axes=[b,q], name='qc')
        ab = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ab')
        ae = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ae')
#        ac = C.input_variable((1,self.word_size), dynamic_axes=[b,a], name='ac')

        #input layer
        c_processed, q_processed = self.input_layer(cgw,cnw,cc,qgw,qnw,qc).outputs
        
        # attention layer
        att_context = self.attention_layer(c_processed, q_processed)

        # modeling layer
        mod_context = self.modeling_layer(att_context) 
     
        # output layer
        outputs = self.output_layer(att_context, mod_context, aw, q_processed, c_processed)
        train_logits, test_output = outputs[0], outputs[1] #workaround for bug
        start_logits, end_logits = outputs[2], outputs[3]
   
       
        #test_output, train_logits = self.output_layer(mod_context, aw)
        #test_output = print_node(test_output)        
        #train_logits =  print_node(train_logits)
        #aw = print_node(aw)
        seq_loss = self.create_criterion_function()
        loss = seq_loss(train_logits, aw)[0] #TODO Feed onehot answer into it

        start_loss = seq_loss(start_logits, ab)
        end_loss = seq_loss(end_logits, ae)
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae)
 
        loss += self.pointer_importance*new_loss
        # loss
        #start_loss = seq_loss(start_logits)
        #end_loss = seq_loss(end_logits)
        #paper_loss = start_loss + end_loss
        #new_loss = all_spans_loss(start_logits, ab, end_logits, ae)
        return C.combine([test_output]), loss
