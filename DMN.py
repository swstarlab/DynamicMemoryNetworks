import numpy as np
import theano
import theano.tensor as T 
import lasagne
import lasagne.nonlinearities as nonlin
from   lasagne.init   import Normal, Constant, GlorotUniform
from   lasagne.layers import Layer, MergeLayer, InputLayer, GRULayer
from   theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from itertools import chain
from six.moves import cPickle as pickle
import h5py

class SemMemModule(MergeLayer):
    # Semantic Memory Module (= Word Embedding Layer)
    # Lasagne Library has Merge Layer, which is basic layer class accepting multiple inputs.
    # Semantic Memory Module and its parameters ared shared into Input Module and Question Module.
    # Therefore, It might not act as ordinary feed-forward layer, and needs extra codes to be trained.
    def __init__(self, incomings, voc_size, hid_state_size, W=Normal(), **kwargs):
        # Initialize parameters and create theano variables
        super(SemMemModule, self).__init__(incomings, **kwargs)
        self.hid_state_size = hid_state_size
        self.W = self.add_param(W, (voc_size, hid_state_size), name='Word_Embedding', regularizable=False)
        self.rand_stream = RandomStreams(np.random.randint(1, 2147462579))
    
    def get_output_shape_for(self, input_shapes):
        # Define output shape for certain input shapes (helps debugging)
        return (None, None, self.hid_state_size)

    def get_output_for(self, inputs, **kwargs):
        # Core part that actually describes how the theano variables work to produce output
        # input is in shape of (batch, sentence, word)
        # word_dropout is the varible determines the proportion of words to be masked to 0-vectors
        input         = inputs[0]
        word_dropout  = inputs[1]

        # Apply an input tensor to word embedding matrix and word_dropout.
        # And then, flatten it to shape of (batch*sentence, word, hid_state) to be fit into GRU library
        # Used Numpy style indexing instead of masking
        return T.reshape(self.W[input], (-1, input.shape[2], self.hid_state_size)) * self.rand_stream.binomial((input.shape[0]*input.shape[1], input.shape[2]), p=1-word_dropout, dtype=theano.config.floatX).dimshuffle((0,1,'x'))


class InputModule(MergeLayer):
    # Input Module, which uses SemMemModule and GRULayer(lasgne)
    def __init__(self, incomings, voc_size, hid_state_size,
                 SemMem=None, GRU=None, **kwargs):
        super(InputModule, self).__init__(incomings, **kwargs)
        
        if SemMem is not None:
            self.SemMem = SemMem
        else:
            self.SemMem = SemMemModule(incomings[0],voc_size,hid_state_size,**kwargs)
        if GRU is not None:
            self.GRU = GRU
        else:
            self.GRU = GRULayer(SemMem, hid_state_size)
        self.voc_size = voc_size
        self.hid_state_size = hid_state_size

    def get_params(self, **tags):
        # Because InputModules uses external GRULayer's parameters,
        # We have to notify this information to train the GRU's parameters. 
        return self.GRU.get_params(**tags)
    def get_output_shape_for(self, input_shape):
        return (None, None, self.hid_state_size)
    def get_output_for(self, inputs, **kwargs):
        input          = inputs[0]
        input_word     = T.flatten(inputs[1])
        word_dropout   = inputs[2]        
        
        # Apply word embedding
        sentence_rep = self.SemMem.get_output_for([input, word_dropout])
        
        # Apply GRU Layer
        gru_outs = self.GRU.get_output_for([sentence_rep])
        
        # Extract candidate fact from GRU's output by input_word variable
        # resolving input with adtional word
        # e.g. John when to the hallway nil nil nil -> [GRU1, ... ,GRU8] -> GRU5
        candidate_facts = T.reshape(
            gru_outs[T.arange(gru_outs.shape[0],dtype='int32'), input_word-1], 
            (-1, input.shape[1], self.hid_state_size))
        return candidate_facts
           
class QuestionModule(MergeLayer):
    # Almost same as Input Module, where its sentense's size is one.
    def __init__(self, incomings, voc_size, hid_state_size,
                 SemMem, GRU, **kwargs):
        super(QuestionModule, self).__init__(incomings, **kwargs)
        self.SemMem = SemMem
        self.GRU    = GRU
        self.voc_size = voc_size
        self.hid_state_size = hid_state_size
    def get_output_shape_for(self, input_shape):
        return (None, self.hid_state_size)
    def get_output_shape_for(self, input_shape):
        return (None, self.hid_state_size)
    def get_output_for(self, inputs, **kwargs):
        qustion       = inputs[0]
        question_word = T.flatten(inputs[1])
        word_dropout  = inputs[2]
        
        q_rep = self.SemMem.get_output_for([qustion, word_dropout])
        gru_outs = self.GRU.get_output_for([q_rep])
        q = T.reshape(
            gru_outs[T.arange(gru_outs.shape[0],dtype='int32'),question_word-1],
            (-1, self.hid_state_size))
        return q
    
class GRU_Gate(object):
    # Helper function of GRU (modified in lasagne library)
    # Hint: We have to impelement custom GRU in later Modules. 
    def __init__(self, W_in=Normal(0.1), W_hid=Normal(0.1),
                 b=Constant(0.), nonlinearity=nonlin.sigmoid):
        self.W_in  = W_in
        self.W_hid = W_hid
        self.b     = b
        if nonlinearity is None:
            self.nonlinearity = nonlin.identity
        else:
            self.nonlinearity = nonlinearity

class EpMemModule(MergeLayer):
    # Episodic Memory Module.
    # This has many varibles and complex operations, 
    # so it would be very hard to understand(and debug) this implememntation. 
    def __init__(self, incomings, hid_state_size, max_sentence,
                 Wb=GlorotUniform(), W1=GlorotUniform(), W2=GlorotUniform(),
                 b1=Constant(0.), b2=Constant(0,),
                 resetgate  = GRU_Gate(), updategate = GRU_Gate(),
                 hid_update = GRU_Gate(nonlinearity=nonlin.tanh),
                 n_pass=2, time_embedding=False, T_=Normal(), **kwargs):
        
        super(EpMemModule, self).__init__(incomings, **kwargs)
        
        # Create parameters for computing gate
        self.Wb = self.add_param(Wb, (1, hid_state_size), name="Wb")
        
        self.W1 = self.add_param(W2, (1, 9), name="W1")
        self.W2 = self.add_param(W1, (hid_state_size, 1), name="W2")
        self.b1 = self.add_param(b2, (hid_state_size,), name="b1", regularizable=False)
        self.b2 = self.add_param(b1, (1,),   name="b2", regularizable=False)
        
        self.max_sentence = max_sentence
        
        # sentence masking
        # sentence_mask_mat[i] = [1111 ... (i times) ... 11110000 ... (n-i times) ... 000]
        smat= np.zeros((max_sentence, max_sentence), dtype=theano.config.floatX)
        for i in xrange(smat.shape[0]):
            for j in xrange(smat.shape[1]):
                smat[i,j] = (0 if j-i > 0 else 1)
        self.sentence_mask_mat=theano.shared(smat,name="sentence_mask_mat",borrow=True)
        
        self.hid_state_size = hid_state_size
        
        # The lines below is modified from lasagne's GRU
        input_shape = self.input_shapes[0]
        num_inputs = np.prod(input_shape[2:])

        self.resetgate= resetgate
        self.updategate=updategate
        self.hid_update=hid_update        
        
        def add_gate(gate, gate_name):
            return (self.add_param(gate.W_in, (num_inputs, hid_state_size),
                        name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (hid_state_size, hid_state_size),
                        name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (hid_state_size,),
                        name="b_{}".format(gate_name), regularizable=False), 
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate,
         self.W_hid_to_updategate,
         self.b_updategate,
         self.nonlinearity_updategate)= add_gate(updategate, 'updategate')
        (self.W_in_to_resetgate,
         self.W_hid_to_resetgate,
         self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate(resetgate, 'resetgate')
        (self.W_in_to_hid_update,
         self.W_hid_to_hid_update,
         self.b_hid_update,
         self.nonlinearity_hid)       = add_gate(hid_update, 'hid_update')
        
        self.n_pass = n_pass
        
        # We use time embedding proposed in End-to-end MemNN(Facebook)
        self.time_embedding=time_embedding
        if time_embedding:
            self.T_ = self.add_param(T_, (int(max_sentence*1.2), hid_state_size), name='Time_Embedding', regularizable=False)
        
    def get_output_shape_for(self, input_shapes):
        # summarized memory's shape
        return (None, self.hid_state_size)
  
    def get_output_for(self, inputs, **kwargs):
        # input_sentence: sentence size
        # input_time : sentence position
        C              = inputs[0]
        q              = inputs[1]
        input_sentence = inputs[2]
        input_time     = inputs[3]
        
        # Apply time embedding
        C = C + self.T_[input_time].dimshuffle(('x',0,1))
        
        # Reshape for parallelizing computation of gates
        C_reshaped = T.reshape(C,(-1,C.shape[1],1,self.hid_state_size))
        tiled_q    = T.tile(T.reshape(
            q,(-1,1,1,self.hid_state_size)),(1,C.shape[1],1,1))
        
        input_sentence_mask = self.sentence_mask_mat[input_sentence-1,:C.shape[1]]

        W_in_stacked  = T.concatenate([self.W_in_to_resetgate, 
                                       self.W_in_to_updategate,
                                       self.W_in_to_hid_update], axis=1)
        W_hid_stacked = T.concatenate([self.W_hid_to_resetgate,
                                       self.W_hid_to_updategate,
                                       self.W_hid_to_hid_update], axis=1)
        b_stacked     = T.concatenate([self.b_resetgate,       
                                       self.b_updategate,       
                                       self.b_hid_update], axis=0)
        
        def Ep_Gate(c, m, q, Wb, W1, W2, b1, b2):
            z = T.concatenate([c,m,q,c*q,c*m,T.abs_(c-q),T.abs_(c-m),c*Wb*q,c*Wb*m], axis=2)
            #g = (T.dot(W2, nonlin.tanh(T.dot(z, W1) + b1)) + b2) <- (big mistake :)
            g = (T.dot(nonlin.tanh(T.dot(W1, z) + b1), W2) + b2)
            return g
    
        def slice_w(x, n):
            return x[:, n*self.hid_state_size:(n+1)*self.hid_state_size]

        # Step for computing summarized episodes recurrently
        def step(hid_previous):
            # Computing a summarized episode.
            tiled_hid_prev = T.tile(T.reshape(
                hid_previous,(-1,1,1,self.hid_state_size)),(1,C.shape[1],1,1))

            g = Ep_Gate(C_reshaped, tiled_hid_prev, tiled_q,
                        self.Wb, self.W1, self.W2, self.b1, self.b2)

            g = T.reshape(g,(-1,C.shape[1]))
            g = T.switch(T.eq(input_sentence_mask, 1), g, np.float32(-np.inf))
            g = nonlin.softmax(g)
            e = T.sum(T.reshape(g,(g.shape[0],g.shape[1],1)) * C, axis=1)

            # After computing the episode, now it is typical GRU.
            input_n = e
            
            hid_input = T.dot(hid_previous, W_hid_stacked)
            input_n = T.dot(input_n, W_in_stacked) + b_stacked

            resetgate  = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate  = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            hid_update_in  = slice_w(input_n, 2)
            hid_update_hid = slice_w(hid_input, 2)
            hid_update     = hid_update_in + resetgate*hid_update_hid

            hid_update = self.nonlinearity_hid(hid_update)

            hid = (1 - updategate)*hid_previous + updategate+hid_update

            return hid

        hid = q

        # Repeat step process in n_pass times.
        for i in xrange(self.n_pass):
            hid = step(hid)
      
        return hid

class EpGateOut(MergeLayer):
    # This is passive layer shares parameters of EpMemModule, JUST FOR GATE ACTIVATION (SOFTMAX) TRAINING.
    def __init__(self, incomings, E, **kwargs):
        super(EpGateOut, self).__init__(incomings, **kwargs)
        self.E = E
        self.max_sentence = E.max_sentence
        self.hid_state_size = E.hid_state_size        
        
        self.Wb = E.Wb
        
        self.W1 = E.W1
        self.W2 = E.W2
        self.b1 = E.b1
        self.b2 = E.b2
        
        self.sentence_mask_mat=E.sentence_mask_mat
            
        input_shape = self.input_shapes[0]
        num_inputs = np.prod(input_shape[2:])
        
        self.resetgate= E.resetgate
        self.updategate=E.updategate
        self.hid_update=E.hid_update
        
        self.W_in_to_updategate = E.W_in_to_updategate
        self.W_hid_to_updategate = E.W_hid_to_updategate
        self.b_updategate = E.b_updategate
        self.nonlinearity_updategate = E.nonlinearity_updategate

        self.W_in_to_resetgate = E.W_in_to_resetgate
        self.W_hid_to_resetgate = E.W_hid_to_resetgate
        self.b_resetgate = E.b_resetgate
        self.nonlinearity_resetgate = E.nonlinearity_updategate
      
        self.W_in_to_hid_update = E.W_in_to_hid_update
        self.W_hid_to_hid_update = E.W_hid_to_hid_update
        self.b_hid_update = E.b_hid_update
        self.nonlinearity_hid = E.nonlinearity_hid
        
        
        self.n_pass = E.n_pass
        
        self.time_embedding=E.time_embedding
        if E.time_embedding:
            self.T_ = E.T_
        
    def get_output_shape_for(self, input_shapes):
        return (None, None)
    
    def get_params(self, **tags):
        return self.E.get_params(**tags)
    
    def get_output_for(self, inputs, **kwargs):
        C              = inputs[0]
        q              = inputs[1]
        input_sentence = inputs[2]
        input_time     = inputs[3]
        
        C = C + self.T_[input_time].dimshuffle(('x',0,1))
        
        C_reshaped = T.reshape(C,(-1,C.shape[1],1,self.hid_state_size))
        tiled_q    = T.tile(T.reshape(
            q,(-1,1,1,self.hid_state_size)),(1,C.shape[1],1,1))
        
        input_sentence_mask = self.sentence_mask_mat[input_sentence-1,:C.shape[1]]

        W_in_stacked  = T.concatenate([self.W_in_to_resetgate, 
                                       self.W_in_to_updategate,
                                       self.W_in_to_hid_update], axis=1)
        W_hid_stacked = T.concatenate([self.W_hid_to_resetgate,
                                       self.W_hid_to_updategate,
                                       self.W_hid_to_hid_update], axis=1)
        b_stacked     = T.concatenate([self.b_resetgate,       
                                       self.b_updategate,       
                                       self.b_hid_update], axis=0)
        
        def Ep_Gate(c, m, q, Wb, W1, W2, b1, b2):
            z = T.concatenate([c,m,q,c*q,c*m,T.abs_(c-q),T.abs_(c-m),c*Wb*q,c*Wb*m], axis=2)
            #g = (T.dot(W2, nonlin.tanh(T.dot(z, W1) + b1)) + b2) <- (big mistake :)
            g = (T.dot(nonlin.tanh(T.dot(W1, z) + b1), W2) + b2)
            return g
    
        def slice_w(x, n):
            return x[:, n*self.hid_state_size:(n+1)*self.hid_state_size]

        def step(hid_previous):
            tiled_hid_prev = T.tile(T.reshape(
                hid_previous,(-1,1,1,self.hid_state_size)),(1,C.shape[1],1,1))
            
            g = Ep_Gate(C_reshaped, tiled_hid_prev, tiled_q,
                        self.Wb, self.W1, self.W2, self.b1, self.b2)
            
            g = T.reshape(g,(-1,C.shape[1]))
            g = T.switch(T.eq(input_sentence_mask, 1), g, np.float32(-np.inf))
            g = nonlin.softmax(g)
            e = T.sum(T.reshape(g,(g.shape[0],g.shape[1],1)) * C, axis=1)

            input_n = e
            
            hid_input = T.dot(hid_previous, W_hid_stacked)
            input_n = T.dot(input_n, W_in_stacked) + b_stacked

            resetgate  = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate  = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            hid_update_in  = slice_w(input_n, 2)
            hid_update_hid = slice_w(hid_input, 2)
            hid_update     = hid_update_in + resetgate*hid_update_hid

            hid_update = self.nonlinearity_hid(hid_update)

            hid = (1 - updategate)*hid_previous + updategate+hid_update

            return (hid, g)

        hid = q

         
        G = []
        for i in xrange(self.n_pass):
            hid, g = step(hid)
            G.append(T.reshape(g, (-1,1,C.shape[1])))
        
        return T.reshape(T.concatenate(G, axis=1), (-1,C.shape[1]))
        
    
class AnswerModule(MergeLayer):
    # Anser Module.
    # Also, it has custom GRU
    def __init__(self, incomings, hid_state_size, voc_size,
                 resetgate  = GRU_Gate(), updategate = GRU_Gate(),
                 hid_update = GRU_Gate(nonlinearity=nonlin.tanh),
                 W=Normal(), max_answer_word=1, **kwargs):
        super(AnswerModule, self).__init__(incomings, **kwargs)
    
        self.hid_state_size = hid_state_size

        #FOR GRU
        input_shape = self.input_shapes[0]
        num_inputs = np.prod(input_shape[1]) + voc_size # concatenation of previous prediction

        def add_gate(gate, gate_name):
            return (self.add_param(gate.W_in, (num_inputs, hid_state_size),
                        name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (hid_state_size, hid_state_size),
                        name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (hid_state_size,),
                        name="b_{}".format(gate_name), regularizable=False), 
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate,
         self.W_hid_to_updategate,
         self.b_updategate,
         self.nonlinearity_updategate)= add_gate(updategate, 'updategate')
        (self.W_in_to_resetgate,
         self.W_hid_to_resetgate,
         self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate(resetgate, 'resetgate')
        (self.W_in_to_hid_update,
         self.W_hid_to_hid_update,
         self.b_hid_update,
         self.nonlinearity_hid)       = add_gate(hid_update, 'hid_update')

        self.W = self.add_param(W, (hid_state_size, voc_size), name="W")
        self.max_answer_word = max_answer_word

        self.rand_stream = RandomStreams(np.random.randint(1, 2147462579))

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0], self.max_answer_word, voc_size)
  
    def get_output_for(self, inputs, **kwargs):
        # typical GRU, but prediction produced by softmax layer is applied to GRU's input
        
        q = inputs[0]
        m = inputs[1]
        epmem_dropout = inputs[2]

        #q = q * self.rand_stream.binomial(q.shape, p=1-epmem_dropout, dtype=theano.config.floatX)
        m = m * self.rand_stream.binomial(m.shape, p=1-epmem_dropout, dtype=theano.config.floatX)

        W_in_stacked  = T.concatenate([self.W_in_to_resetgate, 
                                       self.W_in_to_updategate,
                                       self.W_in_to_hid_update], axis=1)
        W_hid_stacked = T.concatenate([self.W_hid_to_resetgate,
                                       self.W_hid_to_updategate,
                                       self.W_hid_to_hid_update], axis=1)
        b_stacked     = T.concatenate([self.b_resetgate,       
                                       self.b_updategate,       
                                       self.b_hid_update], axis=0)
        def slice_w(x, n):
            return x[:, n*self.hid_state_size:(n+1)*self.hid_state_size]

        def get_output(a):
            return nonlin.softmax(T.dot(a,self.W))
        def step(hid_previous, out_previous, *args):
            input_n = T.concatenate([out_previous, q], axis=1)

            hid_input = T.dot(hid_previous, W_hid_stacked)
            input_n = T.dot(input_n, W_in_stacked) + b_stacked

            resetgate  = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate  = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            hid_update_in  = slice_w(input_n, 2)
            hid_update_hid = slice_w(hid_input, 2)
            hid_update     = hid_update_in + resetgate*hid_update_hid

            hid_update = self.nonlinearity_hid(hid_update)

            hid = (1 - updategate)*hid_previous + updategate+hid_update
            out = nonlin.softmax(T.dot(hid, self.W))

            return (hid, out)
        
        non_seqs = [W_in_stacked, b_stacked, W_hid_stacked, q, m, self.W]
        hid_and_out, b = theano.scan(
            fn=step,
            outputs_info=[m, get_output(m)],
            non_sequences=non_seqs,
            strict=True,
            n_steps=self.max_answer_word)

        return T.transpose(hid_and_out[1], (1,0,2))

class DMN(object):
    def __init__(self, config, word_dict):
        self.config = config
        self.word_dict = word_dict
        
        # Configuration
        voc_size           = config['voc_size']
        hid_state_size     = config['hid_state_size']
        max_word           = config['max_word']      
        max_sentence       = config['max_sentence']
        max_answer_word    = config['max_answer_word']
        ep_pass            = config['ep_pass']
        word_dict          = word_dict

        # Creating Input Tensor
        input_var          = T.itensor3('input')
        input_sentence_var = T.ivector('input_sentence')
        input_word_var     = T.imatrix('input_word')
        question_var       = T.itensor3('question')
        question_word_var  = T.imatrix('question_word')
        target_answer_var  = T.ivector('target_answer')
        input_time_var     = T.ivector('input_time')
        word_dropout_var   = T.scalar('word_dropout', dtype=theano.config.floatX)
        target_gate_var    = T.ivector('target_gate')
        
        epmem_dropout_var = T.scalar('epmem_dropout', dtype=theano.config.floatX)

        # Creating Input Layer
        input = InputLayer(
            shape=(None, None, None),
            input_var=input_var, name='input')
        input_sentence = InputLayer(
            shape=(None,),
            input_var=input_sentence_var, name='input_sentence')
        input_word = InputLayer(
            shape=(None, None),
            input_var=input_word_var, name='input_word')
        question = InputLayer(
            shape=(None, 1, None),
            input_var=question_var, name='question')
        question_word = InputLayer(
            shape=(None, 1),
            input_var=question_word_var, name='question_word')
        input_time = InputLayer(
            shape=(None,),
            input_var=input_time_var, name='input_time')        
        word_dropout  = InputLayer(
            shape=(),
            input_var=word_dropout_var, name='word_dropout')
        epmem_dropout  = InputLayer(
            shape=(),
            input_var=epmem_dropout_var, name='epmem_dropout')

        # Creating DMN's Module
        S = SemMemModule(
            [input, word_dropout], 
            voc_size=voc_size,
            hid_state_size=hid_state_size,
            W=config['word_embedding'])
        I = InputModule(
            [input, input_word, word_dropout],
            voc_size=voc_size,
            hid_state_size=hid_state_size,
            SemMem=S)
        Q = QuestionModule(
            [question, question_word, word_dropout],
            voc_size=voc_size,
            hid_state_size=hid_state_size,
            SemMem=S, GRU=I.GRU)
        E = EpMemModule(
            [I, Q, input_sentence, input_time],
            hid_state_size=hid_state_size,
            max_sentence=max_sentence, n_pass=ep_pass, time_embedding=True)        
        A = AnswerModule(
            [Q, E, epmem_dropout],
            hid_state_size=hid_state_size,
            voc_size=voc_size, max_answer_word=1)
        
        # Gate's Out Layer
        E_G = EpGateOut([I, Q, input_sentence, input_time], E)

        # Making this variable accessible by DMN class
        self.voc_size           = voc_size           
        self.hid_state_size     = hid_state_size     
        self.max_word           = max_word           
        self.max_sentence       = max_sentence       
        self.max_answer_word    = max_answer_word
        self.word_dict          = word_dict

        self.input_var          = input_var         
        self.input_sentence_var = input_sentence_var
        self.input_word_var     = input_word_var    
        self.question_var       = question_var      
        self.question_word_var  = question_word_var 
        self.target_answer_var  = target_answer_var
        self.input_time_var     = input_time_var
        self.word_dropout_var   = word_dropout_var
        self.target_gate_var    = target_gate_var
        self.epmem_dropout_var  = epmem_dropout_var 

        self.input              = input         
        self.input_sentence     = input_sentence
        self.input_word         = input_word    
        self.question           = question      
        self.question_word      = question_word
        self.input_time         = input_time
        self.word_dropout       = word_dropout
        self.epmem_dropout      = epmem_dropout

        self.S                  = S
        self.I                  = I
        self.Q                  = Q
        self.E                  = E
        self.A                  = A
        self.E_G                = E_G
        
    def save_params(self, fname):
        layers = [self.S] + lasagne.layers.get_all_layers(self.A)
        params = chain.from_iterable(l.get_params() for l in layers)
        params = lasagne.utils.unique(params)
        
        npy_list = [param.get_value(borrow=True) for param in params]

        with open(fname + ".pkl", 'wb') as f:
            pickle.dump(npy_list, f, pickle.HIGHEST_PROTOCOL)
        
    def load_params(self, fname):
        layers = [self.S] + lasagne.layers.get_all_layers(self.A)
        params = chain.from_iterable(l.get_params() for l in layers)
        params = lasagne.utils.unique(params)
        
        with open(fname + ".pkl", "rb") as f:
            npy_list = pickle.load(f)
        
        for i in xrange(len(params)):
            params[i].set_value(npy_list[i])
        
        
        
        
        
