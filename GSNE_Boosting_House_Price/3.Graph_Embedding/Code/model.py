
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.sparse as sp 
import numpy as np 
import cyclic_learning_rate.clr as clr

seed = 42

def sparse_feeder(M):
    M = sp.coo_matrix(M, dtype = np.float32)
    return np.vstack((M.row, M.col)).T, M.data, M.shape

class GLACE:
    def __init__(self, args):
        tf.set_random_seed(seed)
        #tf.random.set_seed(seed)
        self.X1 = tf.SparseTensor(*sparse_feeder(args.X1))
        self.X2 = tf.SparseTensor(*sparse_feeder(args.X2))
        self.X3 = tf.SparseTensor(*sparse_feeder(args.X3))
        self.X4 = tf.SparseTensor(*sparse_feeder(args.X4))

        self.N1, self.D1 = args.X1.shape
        self.N2, self.D2 = args.X2.shape
        self.N3, self.D3 = args.X3.shape
        self.N4, self.D4 = args.X4.shape  

        self.L = args.embedding_dim

        #PLEASE ENSURE THE LAST LAYER DIMENSION IS SAME FOR EVERYONE

        self.n_hidden1 = [6, 12, 28]
        self.n_hidden2 = [10, 16, 28]
        self.n_hidden3 = [10, 16, 28]
        self.n_hidden4 = [42, 36, 28]

        '''self.n_hidden1 = [14]
        self.n_hidden2 = [14]
        self.n_hidden3 = [14]
        self.n_hidden4 = [14]'''

        self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.label = tf.placeholder(name='label', dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])
        self.node_type1 = tf.placeholder(name='node_type1', dtype=tf.int32, shape = ())
        self.node_type2 = tf.placeholder(name='node_type2', dtype=tf.int32, shape = ())
        
        # base_val_int= tf.ones(shape=[args.batch_size * (args.K + 1)],dtype=tf.int32)
        # base_val_float = tf.ones(shape=[args.batch_size * (args.K + 1)],dtype=tf.float32) 
        # base_val = tf.ones(shape=[1],dtype=tf.int32)
    
        # self.u_i = tf.Variable(initial_value=base_val_int,name='u_i', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        # self.u_j = tf.Variable(initial_value=base_val_int,name='u_j', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        # self.label = tf.Variable(initial_value=base_val_float,name='label', dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])
        # self.node_type1 = tf.Variable(initial_value=0,name='node_type1', dtype=tf.int32, shape = ())
        # self.node_type2 = tf.Variable(initial_value=0,name='node_type2', dtype=tf.int32, shape = ())
        
       
        

        self.__create_model(args.proximity)
        self.val_set = False
        
        #tf.train.create_global_step()
        tf.compat.v1.train.create_global_step() 
        # softmax loss
        
        self.energy = -self.energy_kl(self.u_i, self.u_j, args.proximity, self.node_type1, self.node_type2)
        self.loss = -tf.reduce_mean(tf.compat.v1.log_sigmoid(self.label * self.energy))
        tf.summary.scalar('loss', self.loss)
        print(args.learning_rate)

        '''for cyclic learning rate'''
        global_step = tf.compat.v1.train.get_global_step()
        #learning_rate = tf.train.exponential_decay((1e-9), global_step=global_step,decay_steps=10, decay_rate=1.04)
        learning_rate = clr.cyclic_learning_rate(global_step=global_step, learning_rate=1e-4,
                         max_lr=19e-5,
                         step_size=100, mode='exp_range', gamma = 0.99999)
        original_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar("current_step",global_step)


        ###########################################################


        #original_optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        
        #self.optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=5.0)
        
        self.optimizer =original_optimizer # change here

        ''' tvs = tf.trainable_variables()
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]                                        
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
        gvs = self.optimizer.compute_gradients(self.loss, tvs)
        self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

        #After getting all the gradients in the five steps, we calculate the train step
        self.train_step = self.optimizer.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])
 '''
        self.train_op = self.optimizer.minimize(self.loss, global_step = global_step)
        #self.train_op = self.optimizer.minimize(self.loss)
        self.merged_summary = tf.summary.merge_all()
        #self.merged_summary = tf.compat.v1.summary.merge_all()

    def __create_model(self, proximity):
        #w_init = tf.keras.initializers.glorot_normal
        #w_init = tf.contrib.layers.xavier_initializer
        w_init = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None)
        #w_init = tf.keras.initializers.random_normal
        sizes1 = [self.D1] + self.n_hidden1
        sizes2 = [self.D2] + self.n_hidden2
        sizes3 = [self.D3] + self.n_hidden3
        sizes4 = [self.D4] + self.n_hidden4

        #feature 1
        TRAINABLE = True
        with tf.name_scope("Train"):
          for i in range(1, len(sizes1)):
             with tf.name_scope("enc{}".format(i)):
              # W = tf.get_variable(name='W1{}'.format(i), shape=[sizes1[i - 1], sizes1[i]], dtype=tf.float32,
              #                     initializer=w_init(), trainable = TRAINABLE)
              # b = tf.get_variable(name='b1{}'.format(i), shape=[sizes1[i]], dtype=tf.float32, initializer=w_init(), trainable = TRAINABLE)
              #---after-replace this line of code           
              W = tf.Variable( name='W1{}'.format(i) , initial_value = w_init( shape=[sizes1[i - 1] , sizes1[i]] , dtype=tf.float32) , trainable = TRAINABLE)
              b = tf.Variable( name='b1{}'.format(i) , initial_value = w_init( shape=[sizes1[i]], dtype=tf.float32) , trainable = TRAINABLE )

              if i == 1:
                  #encoded1 = tf.sparse_tensor_dense_matmul(self.X1, W) + b
                  #---after-replace this line of code
                  encoded1 = tf.sparse.sparse_dense_matmul(self.X1, W) + b
              else:
                  encoded1 = tf.matmul(encoded1, W) + b

              encoded1 = tf.nn.relu(encoded1)

              tf.summary.histogram('Weight', W)
              tf.summary.histogram('bias', b)
              tf.summary.histogram('activations', encoded1)


        #encoded1 = tf.Print(encoded1, [encoded1], message = "feature 1 encoder triggered")

        #feature 2
        TRAINABLE = True
        with tf.name_scope("Region"):
          for i in range(1, len(sizes2)):
            with tf.name_scope("enc{}".format(i)):
              # W = tf.get_variable(name='W2{}'.format(i), shape=[sizes2[i - 1], sizes2[i]], dtype=tf.float32,
              #                     initializer=w_init(), trainable = TRAINABLE)
              # b = tf.get_variable(name='b2{}'.format(i), shape=[sizes2[i]], dtype=tf.float32, initializer=w_init(), trainable = TRAINABLE)
              #--after-change-this-line-of-code
              W = tf.Variable( name='W2{}'.format(i) , initial_value = w_init( shape=[sizes2[i - 1] , sizes2[i]] , dtype=tf.float32) , trainable = TRAINABLE)
              b = tf.Variable( name='b2{}'.format(i) , initial_value = w_init( shape=[sizes2[i]], dtype=tf.float32) , trainable = TRAINABLE )

              if i == 1:
                  #encoded2 = tf.sparse_tensor_dense_matmul(self.X2, W) + b
                  #---after-replace this line of code
                  encoded2 = tf.sparse.sparse_dense_matmul(self.X2, W) + b
              else:
                  encoded2 = tf.matmul(encoded2, W) + b

              encoded2 = tf.nn.relu(encoded2)

              tf.summary.histogram('Weight', W)
              tf.summary.histogram('bias', b)
              tf.summary.histogram('activations', encoded2)

        #encoded2 = tf.Print(encoded2, [encoded2], message = "feature 2 encoder triggered")

        #feature 3
        TRAINABLE = True
        with tf.name_scope("School"):
          for i in range(1, len(sizes3)):
            with tf.name_scope("enc{}".format(i)):
              # W = tf.get_variable(name='W3{}'.format(i), shape=[sizes3[i - 1], sizes3[i]], dtype=tf.float32,
              #                     initializer=w_init(), trainable = TRAINABLE)
              # b = tf.get_variable(name='b3{}'.format(i), shape=[sizes3[i]], dtype=tf.float32, initializer=w_init(), trainable = TRAINABLE)
              #---after-replace this line of code           
              W = tf.Variable( name='W3{}'.format(i) , initial_value = w_init( shape=[sizes3[i - 1] , sizes3[i]] , dtype=tf.float32) , trainable = TRAINABLE)
              b = tf.Variable( name='b3{}'.format(i) , initial_value = w_init( shape=[sizes3[i]], dtype=tf.float32) , trainable = TRAINABLE )


              if i == 1:
                  #encoded3 = tf.sparse_tensor_dense_matmul(self.X3, W) + b
                  #---after-replace this line of code
                  encoded3 = tf.sparse.sparse_dense_matmul(self.X3, W) + b
              else:
                  encoded3 = tf.matmul(encoded3, W) + b

              encoded3 = tf.nn.relu(encoded3)

              tf.summary.histogram('Weight', W)
              tf.summary.histogram('bias', b)
              tf.summary.histogram('activations', encoded3)


        #encoded3 = tf.Print(encoded3, [encoded3], message = "feature 3 encoder triggered")

        #feature 4
        TRAINABLE = True
        with tf.name_scope("House"):
          for i in range(1, len(sizes4)):
            with tf.name_scope("enc{}".format(i)):
              # W = tf.get_variable(name='W4{}'.format(i), shape=[sizes4[i - 1], sizes4[i]], dtype=tf.float32,
              #                     initializer=w_init(), trainable = TRAINABLE)
              # b = tf.get_variable(name='b4{}'.format(i), shape=[sizes4[i]], dtype=tf.float32, initializer=w_init(), trainable = TRAINABLE)
              #---after-replace this line of code           
              W = tf.Variable( name='W4{}'.format(i) , initial_value = w_init( shape=[sizes4[i - 1] , sizes4[i]] , dtype=tf.float32) , trainable = TRAINABLE)
              b = tf.Variable( name='b4{}'.format(i) , initial_value = w_init( shape=[sizes4[i]], dtype=tf.float32) , trainable = TRAINABLE )

              if i == 1:
                  # encoded4 = tf.sparse_tensor_dense_matmul(self.X4, W) + b
                  #---after-replace this line of code
                  encoded4 = tf.sparse.sparse_dense_matmul(self.X4, W) + b
              else:
                  encoded4 = tf.matmul(encoded4, W) + b

              encoded4 = tf.nn.relu(encoded4)

              tf.summary.histogram('Weight', W)
              tf.summary.histogram('bias', b)
              tf.summary.histogram('activations', encoded4)
        
        

        #encoded4 = tf.Print(encoded4, [encoded4], message = "feature 4 encoder triggered")

        #W-MU/SIGMA AND B-MU/SIGMA IS SHARED BETWEEN ALL FEATURES
        #SHAPE: THOUGH WE USED SIZES1[-1], KEEP IN MIND THAT'S SAME FOR ALL SHAPES

        """ W_mu = tf.get_variable(name='W_mu', shape=[sizes1[-1], self.L], dtype=tf.float32, initializer=w_init())
        b_mu = tf.get_variable(name='b_mu', shape=[self.L], dtype=tf.float32, initializer=w_init()) 
        self.embedding1 = tf.matmul(encoded1, W_mu) + b_mu
        self.embedding2 = tf.matmul(encoded2, W_mu) + b_mu
        self.embedding3 = tf.matmul(encoded3, W_mu) + b_mu
        self.embedding4 = tf.matmul(encoded4, W_mu) + b_mu

        with tf.name_scope("shared"):
          with tf.name_scope("mu"):
            tf.summary.histogram('Weight', W_mu)
            tf.summary.histogram('bias', b_mu)
            mu_embed_activations = [self.embedding1, self.embedding2, self.embedding3, self.embedding4]
            tf.summary.histogram('activations', tf.concat(mu_embed_activations, 0)) """

      
        '''self.embedding1 = tf.nn.sigmoid(tf.matmul(encoded1, W_mu) + b_mu) + 1 + 1e-14
        self.embedding2 = tf.nn.sigmoid(tf.matmul(encoded2, W_mu) + b_mu) + 1 + 1e-14
        self.embedding3 = tf.nn.sigmoid(tf.matmul(encoded3, W_mu) + b_mu) + 1 + 1e-14
        self.embedding4 = tf.nn.sigmoid(tf.matmul(encoded4, W_mu) + b_mu) + 1 + 1e-14'''

        """ W_sigma = tf.get_variable(name='W_sigma', shape=[sizes1[-1], self.L], dtype=tf.float32, initializer=w_init())
        b_sigma = tf.get_variable(name='b_sigma', shape=[self.L], dtype=tf.float32, initializer=w_init())
        log_sigma1 = tf.matmul(encoded1, W_sigma) + b_sigma
        self.sigma1 = tf.nn.elu(log_sigma1) + 1 + 1e-14
        #self.sigma1 = tf.nn.sigmoid(log_sigma1) + 1 + 1e-14

        log_sigma2 = tf.matmul(encoded2, W_sigma) + b_sigma
        self.sigma2 = tf.nn.elu(log_sigma2) + 1 + 1e-14
        #self.sigma2 = tf.nn.sigmoid(log_sigma2) + 1 + 1e-14

        log_sigma3 = tf.matmul(encoded3, W_sigma) + b_sigma
        self.sigma3 = tf.nn.elu(log_sigma3) + 1 + 1e-14
        #self.sigma3 = tf.nn.sigmoid(log_sigma3) + 1 + 1e-14

        log_sigma4 = tf.matmul(encoded4, W_sigma) + b_sigma
        self.sigma4 = tf.nn.elu(log_sigma4) + 1 + 1e-14
        #self.sigma4 = tf.nn.sigmoid(log_sigma4) + 1 + 1e-14

        with tf.name_scope("shared"):
          with tf.name_scope("sigma"):
            tf.summary.histogram('Weight', W_sigma)
            tf.summary.histogram('bias', b_sigma)
            sigma_embed_activations = [self.sigma1, self.sigma2, self.sigma3, self.sigma4]
            tf.summary.histogram('activations', tf.concat(sigma_embed_activations, 0)) """ 

        ##############EXPERIMENTAL FEATURES. PLEASE REMOVE IF DOESN'T WORK############################################

        # W_mu1 = tf.get_variable(name='W_mu1', shape=[sizes1[-1], 40], dtype=tf.float32, initializer=w_init())
        # b_mu1 = tf.get_variable(name='b_mu1', shape=[40], dtype=tf.float32, initializer=w_init())
        # #---after-replace this line of code           
        W_mu1 = tf.Variable( name='W_mu1' , initial_value = w_init( shape=[sizes1[-1], 40], dtype=tf.float32 ) )
        b_mu1 = tf.Variable( name='b_mu1' , initial_value = w_init( shape=[40], dtype=tf.float32 )  )



        # W_mu2 = tf.get_variable(name='W_mu2', shape=[40, self.L], dtype=tf.float32, initializer=w_init())
        # b_mu2 = tf.get_variable(name='b_mu2', shape=[self.L], dtype=tf.float32, initializer=w_init())
        # #---after-replace this line of code           
        W_mu2 = tf.Variable( name='W_mu2' , initial_value = w_init( shape=[40, self.L], dtype=tf.float32 ) )
        b_mu2 = tf.Variable( name='b_mu2' , initial_value = w_init( shape=[self.L], dtype=tf.float32 )  )

        embedding1_t = tf.nn.relu(tf.matmul(encoded1, W_mu1) + b_mu1)
        self.embedding1 = tf.matmul(embedding1_t, W_mu2) + b_mu2

        embedding2_t = tf.nn.relu(tf.matmul(encoded2, W_mu1) + b_mu1)
        self.embedding2 = tf.matmul(embedding2_t, W_mu2) + b_mu2

        embedding3_t = tf.nn.relu(tf.matmul(encoded3, W_mu1) + b_mu1)
        self.embedding3 = tf.matmul(embedding3_t, W_mu2) + b_mu2

        embedding4_t = tf.nn.relu(tf.matmul(encoded4, W_mu1) + b_mu1)
        self.embedding4 = tf.matmul(embedding4_t, W_mu2) + b_mu2

        # W_sigma1 = tf.get_variable(name='W_sigma1', shape=[sizes1[-1], 40], dtype=tf.float32, initializer=w_init())
        # W_sigma2 = tf.get_variable(name='W_sigma2', shape=[40, self.L], dtype=tf.float32, initializer=w_init())
        # #---after-replace this line of code           
        W_sigma1 = tf.Variable( name='W_sigma1' , initial_value = w_init( shape=[sizes1[-1], 40], dtype=tf.float32 ) )
        W_sigma2 = tf.Variable( name='W_sigma2' , initial_value = w_init( shape=[40, self.L], dtype=tf.float32 )  )

        # b_sigma1 = tf.get_variable(name='b_sigma1', shape=[40], dtype=tf.float32, initializer=w_init())
        # b_sigma2 = tf.get_variable(name='b_sigma2', shape=[self.L], dtype=tf.float32, initializer=w_init())
        # #---after-replace this line of code           
        b_sigma1 = tf.Variable( name='b_sigma1' , initial_value = w_init( shape=[40], dtype=tf.float32 ) )
        b_sigma2 = tf.Variable( name='b_sigma2' , initial_value = w_init( shape=[self.L], dtype=tf.float32 )  )
        

        log_sigma1t = tf.nn.relu(tf.matmul(encoded1, W_sigma1) + b_sigma1)
        log_sigma1 = tf.matmul(log_sigma1t, W_sigma2) + b_sigma2
        self.sigma1 = tf.nn.elu(log_sigma1) + 1 + 1e-14
        #self.sigma1 = tf.nn.sigmoid(log_sigma1) + 1 + 1e-14

        log_sigma2t = tf.nn.relu(tf.matmul(encoded2, W_sigma1) + b_sigma1)
        log_sigma2 = tf.matmul(log_sigma2t, W_sigma2) + b_sigma2
        self.sigma2 = tf.nn.elu(log_sigma2) + 1 + 1e-14
        #self.sigma2 = tf.nn.sigmoid(log_sigma2) + 1 + 1e-14

        log_sigma3t = tf.nn.relu(tf.matmul(encoded3, W_sigma1) + b_sigma1)
        log_sigma3 = tf.matmul(log_sigma3t, W_sigma2) + b_sigma2
        self.sigma3 = tf.nn.elu(log_sigma3) + 1 + 1e-14
        #self.sigma3 = tf.nn.sigmoid(log_sigma3) + 1 + 1e-14

        log_sigma4t = tf.nn.relu(tf.matmul(encoded4, W_sigma1) + b_sigma1)
        log_sigma4 = tf.matmul(log_sigma4t, W_sigma2) + b_sigma2
        self.sigma4 = tf.nn.elu(log_sigma4) + 1 + 1e-14 
        #self.sigma4 = tf.nn.sigmoid(log_sigma4) + 1 + 1e-14

        ########################################################################################################################
        #####################END OF EXPERIMENTAL, DELETE IF DOESN'T WORK########################################################
        #######################################################################################################################
        


        if proximity == 'second-order':
            #feature 1

            for i in range(1, len(sizes1)):
                # W = tf.get_variable(name='W_ctx1{}'.format(i), shape=[sizes1[i - 1], sizes1[i]], dtype=tf.float32,
                #                     initializer=w_init())
                # b = tf.get_variable(name='b_ctx1{}'.format(i), shape=[sizes1[i]], dtype=tf.float32, initializer=w_init())
                #---after-replace this line of code           
                W = tf.Variable( name='W_ctx1{}'.format(i) , initial_value = w_init( shape=[sizes1[i - 1] , sizes1[i]] , dtype=tf.float32) )
                b = tf.Variable( name='b_ctx1{}'.format(i) , initial_value = w_init( shape=[sizes1[i]], dtype=tf.float32)  )

                if i == 1:
                    #encoded1 = tf.sparse_tensor_dense_matmul(self.X1, W) + b
                    #---after-replace this line of code
                    encoded1 = tf.sparse.sparse_dense_matmul(self.X1, W) + b
                else:
                    encoded1 = tf.matmul(encoded1, W) + b

                encoded1 = tf.nn.relu(encoded1)

            #feature 2

            for i in range(1, len(sizes2)):
                # W = tf.get_variable(name='W_ctx2{}'.format(i), shape=[sizes2[i - 1], sizes2[i]], dtype=tf.float32,
                #                     initializer=w_init())
                # b = tf.get_variable(name='b_ctx2{}'.format(i), shape=[sizes2[i]], dtype=tf.float32, initializer=w_init())
                #---after-replace this line of code           
                W = tf.Variable( name='W_ctx2{}'.format(i) , initial_value = w_init( shape=[sizes2[i - 1] , sizes2[i]] , dtype=tf.float32) )
                b = tf.Variable( name='b_ctx2{}'.format(i) , initial_value = w_init( shape=[sizes2[i]], dtype=tf.float32)  )

                if i == 1:
                    #encoded2 = tf.sparse_tensor_dense_matmul(self.X2, W) + b
                    #---after-replace this line of code
                    encoded2 = tf.sparse.sparse_dense_matmul(self.X2, W) + b
                else:
                    encoded2 = tf.matmul(encoded2, W) + b

                encoded2 = tf.nn.relu(encoded2)

            #feature 3

            for i in range(1, len(sizes3)):
                # W = tf.get_variable(name='W_ctx3{}'.format(i), shape=[sizes3[i - 1], sizes3[i]], dtype=tf.float32,
                #                     initializer=w_init())
                # b = tf.get_variable(name='b_ctx3{}'.format(i), shape=[sizes3[i]], dtype=tf.float32, initializer=w_init())
                #---after-replace this line of code           
                W = tf.Variable( name='W_ctx3{}'.format(i) , initial_value = w_init( shape=[sizes3[i - 1] , sizes3[i]] , dtype=tf.float32) )
                b = tf.Variable( name='b_ctx3{}'.format(i) , initial_value = w_init( shape=[sizes3[i]], dtype=tf.float32)  )

                if i == 1:
                    #encoded3 = tf.sparse_tensor_dense_matmul(self.X3, W) + b
                    #---after-replace this line of code
                    encoded3 = tf.sparse.sparse_dense_matmul(self.X3, W) + b             
                else:
                    encoded3 = tf.matmul(encoded3, W) + b

                encoded3 = tf.nn.relu(encoded3)

            #feature 4

            for i in range(1, len(sizes4)):
                # W = tf.get_variable(name='W_ctx4{}'.format(i), shape=[sizes4[i - 1], sizes4[i]], dtype=tf.float32,
                #                     initializer=w_init())
                # b = tf.get_variable(name='b_ctx4{}'.format(i), shape=[sizes4[i]], dtype=tf.float32, initializer=w_init())
                #---after-replace this line of code           
                W = tf.Variable( name='W_ctx4{}'.format(i) , initial_value = w_init( shape=[sizes4[i - 1] , sizes4[i]] , dtype=tf.float32) )
                b = tf.Variable( name='b_ctx4{}'.format(i) , initial_value = w_init( shape=[sizes4[i]], dtype=tf.float32)  )

                if i == 1:
                    #encoded4 = tf.sparse_tensor_dense_matmul(self.X4, W) + b
                    #---after-replace this line of code
                    encoded4 = tf.sparse.sparse_dense_matmul(self.X4, W) + b
                else:
                    encoded4 = tf.matmul(encoded4, W) + b

                encoded4 = tf.nn.relu(encoded4)
            ################ USE INTERCHANGABLY WITH THE HIGHER DIMENSION#####################################################
            """ W_mu = tf.get_variable(name='W_mu_ctx', shape=[sizes1[-1], self.L], dtype=tf.float32, initializer=w_init())
            b_mu = tf.get_variable(name='b_mu_ctx', shape=[self.L], dtype=tf.float32, initializer=w_init())
            
            self.ctx_mu1 = tf.matmul(encoded1, W_mu) + b_mu
            self.ctx_mu2 = tf.matmul(encoded2, W_mu) + b_mu
            self.ctx_mu3 = tf.matmul(encoded3, W_mu) + b_mu
            self.ctx_mu4 = tf.matmul(encoded4, W_mu) + b_mu

            ''' self.ctx_mu1 = tf.nn.sigmoid(tf.matmul(encoded1, W_mu) + b_mu) + 1 + 1e-14
            self.ctx_mu2 = tf.nn.sigmoid(tf.matmul(encoded2, W_mu) + b_mu) + 1 + 1e-14
            self.ctx_mu3 = tf.nn.sigmoid(tf.matmul(encoded3, W_mu) + b_mu) + 1 + 1e-14 

            self.ctx_mu4 = tf.nn.sigmoid(tf.matmul(encoded4, W_mu) + b_mu) + 1 + 1e-14 '''

            W_sigma = tf.get_variable(name='W_sigma_ctx', shape=[sizes1[-1], self.L], dtype=tf.float32,
                                      initializer=w_init())
            b_sigma = tf.get_variable(name='b_sigma_ctx', shape=[self.L], dtype=tf.float32, initializer=w_init())
            
            log_sigma1 = tf.matmul(encoded1, W_sigma) + b_sigma
            self.ctx_sigma1 = tf.nn.elu(log_sigma1) + 1 + 1e-14
            #self.ctx_sigma1 = tf.nn.sigmoid(log_sigma1) + 1 + 1e-14

            log_sigma2 = tf.matmul(encoded2, W_sigma) + b_sigma
            self.ctx_sigma2 = tf.nn.elu(log_sigma2) + 1 + 1e-14
            #self.ctx_sigma2 = tf.nn.sigmoid(log_sigma2) + 1 + 1e-14

            log_sigma3 = tf.matmul(encoded3, W_sigma) + b_sigma
            self.ctx_sigma3 = tf.nn.elu(log_sigma3) + 1 + 1e-14
            #self.ctx_sigma3 = tf.nn.sigmoid(log_sigma3) + 1 + 1e-14


            log_sigma4 = tf.matmul(encoded4, W_sigma) + b_sigma
            self.ctx_sigma4 = tf.nn.elu(log_sigma4) + 1 + 1e-14
            #self.ctx_sigma4 = tf.nn.sigmoid(log_sigma4) + 1 + 1e-14 """

            #############HIGHER DIMENSION VERSION ##############################################
            # W_mu1 = tf.get_variable(name='W_mu_ctx1', shape=[sizes1[-1], 40], dtype=tf.float32, initializer=w_init())
            # b_mu1 = tf.get_variable(name='b_mu_ctx1', shape=[40], dtype=tf.float32, initializer=w_init())
            # #---after-replace this line of code           
            W_mu1 = tf.Variable( name='W_mu1_ctx1' , initial_value = w_init( shape=[sizes1[-1], 40], dtype=tf.float32 ) )
            b_mu1 = tf.Variable( name='b_mu1_ctx1' , initial_value = w_init( shape=[40], dtype=tf.float32 )  )
  
            
            # W_mu2 = tf.get_variable(name='W_mu_ctx2', shape=[40, self.L], dtype=tf.float32, initializer=w_init())
            # b_mu2 = tf.get_variable(name='b_mu_ctx2', shape=[self.L], dtype=tf.float32, initializer=w_init())
            # #---after-replace this line of code           
            W_mu2 = tf.Variable( name='W_mu2_ctx2' , initial_value = w_init( shape=[40, self.L], dtype=tf.float32 ) )
            b_mu2 = tf.Variable( name='b_mu2_ctx2' , initial_value = w_init( shape=[self.L], dtype=tf.float32 )  )
                
            ctx_mu1_t = tf.nn.relu(tf.matmul(encoded1, W_mu1) + b_mu1)
            self.ctx_mu1 = tf.matmul(ctx_mu1_t, W_mu2) + b_mu2
            
            ctx_mu2_t = tf.nn.relu(tf.matmul(encoded2, W_mu1) + b_mu1)
            self.ctx_mu2 = tf.matmul(ctx_mu2_t, W_mu2) + b_mu2
            
            ctx_mu3_t = tf.nn.relu(tf.matmul(encoded3, W_mu1) + b_mu1)
            self.ctx_mu3 = tf.matmul(ctx_mu3_t, W_mu2) + b_mu2
            
            ctx_mu4_t = tf.nn.relu(tf.matmul(encoded4, W_mu1) + b_mu1)
            self.ctx_mu4 = tf.matmul(ctx_mu4_t, W_mu2) + b_mu2

            # W_sigma1 = tf.get_variable(name='W_sigma_ctx1', shape=[sizes1[-1], 40], dtype=tf.float32, initializer=w_init())
            # W_sigma2 = tf.get_variable(name='W_sigma_ctx2', shape=[40, self.L], dtype=tf.float32, initializer=w_init())
            # #---after-replace this line of code           
            W_sigma1 = tf.Variable( name='W_sigma1_ctx1' , initial_value = w_init( shape=[sizes1[-1], 40], dtype=tf.float32 ) )
            W_sigma2 = tf.Variable( name='W_sigma2_ctx2' , initial_value = w_init( shape=[40, self.L], dtype=tf.float32 )  )
                
                
            # b_sigma1 = tf.get_variable(name='b_sigma_ctx1', shape=[40], dtype=tf.float32, initializer=w_init())
            # b_sigma2 = tf.get_variable(name='b_sigma_ctx2', shape=[self.L], dtype=tf.float32, initializer=w_init())
            # #---after-replace this line of code           
            b_sigma1 = tf.Variable( name='b_sigma_ctx1' , initial_value = w_init( shape=[40], dtype=tf.float32 ) )
            b_sigma2 = tf.Variable( name='b_sigma_ctx2' , initial_value = w_init( shape=[self.L], dtype=tf.float32 )  )
            

            log_sigma1t = tf.nn.relu(tf.matmul(encoded1, W_sigma1) + b_sigma1)
            log_sigma1 = tf.matmul(log_sigma1t, W_sigma2) + b_sigma2
            self.ctx_sigma1 = tf.nn.elu(log_sigma1) + 1 + 1e-14
            #self.ctx_sigma1 = tf.nn.sigmoid(log_sigma1) + 1 + 1e-14
			
            log_sigma2t = tf.nn.relu(tf.matmul(encoded2, W_sigma1) + b_sigma1)
            log_sigma2 = tf.matmul(log_sigma2t, W_sigma2) + b_sigma2
            self.ctx_sigma2 = tf.nn.elu(log_sigma2) + 1 + 1e-14
            #self.ctx_sigma2 = tf.nn.sigmoid(log_sigma2) + 1 + 1e-14

            log_sigma3t = tf.nn.relu(tf.matmul(encoded3, W_sigma1) + b_sigma1)
            log_sigma3 = tf.matmul(log_sigma3t, W_sigma2) + b_sigma2
            self.ctx_sigma3 = tf.nn.elu(log_sigma3) + 1 + 1e-14
            #self.ctx_sigma3 = tf.nn.sigmoid(log_sigma3) + 1 + 1e-14

            log_sigma4t = tf.nn.relu(tf.matmul(encoded4, W_sigma1) + b_sigma1)
            log_sigma4 = tf.matmul(log_sigma4t, W_sigma2) + b_sigma2
            self.ctx_sigma4 = tf.nn.elu(log_sigma4) + 1 + 1e-14
            #self.ctx_sigma4 = tf.nn.sigmoid(log_sigma4) + 1 + 1e-14
            #########################################DEEPER MODEL END##################################


    def energy_kl(self, u_i, u_j, proximity, node_type1, node_type2):
        def f1():
          print("f1") 
          return tf.gather(self.embedding1, u_i), tf.gather(self.sigma1, u_i)
        def f2(): 
          print("f2")
          return tf.gather(self.embedding2, u_i), tf.gather(self.sigma2, u_i)
        def f3(): 
          print("f3")
          return tf.gather(self.embedding3, u_i), tf.gather(self.sigma3, u_i)
        def f4(): 
          print("f4")
          return tf.gather(self.embedding4, u_i), tf.gather(self.sigma4, u_i)

        def f5(): 
          print("f5")
          return tf.gather(self.ctx_mu1, u_j), tf.gather(self.ctx_sigma1, u_j)
        def f6(): 
          print("f6")
          return tf.gather(self.ctx_mu2, u_j), tf.gather(self.ctx_sigma2, u_j)
        def f7(): 
          print("f7")
          return tf.gather(self.ctx_mu3, u_j), tf.gather(self.ctx_sigma3, u_j)
        def f8(): 
          print("f8")
          return tf.gather(self.ctx_mu4, u_j), tf.gather(self.ctx_sigma4, u_j)

        def f9():
          print("f9") 
          return tf.gather(self.embedding1, u_j), tf.gather(self.sigma1, u_j)
        def f10(): 
          print("f10")
          return tf.gather(self.embedding2, u_j), tf.gather(self.sigma2, u_j)
        def f11(): 
          print("f11")
          return tf.gather(self.embedding3, u_j), tf.gather(self.sigma3, u_j)
        def f12(): 
          print("f12")
          return tf.gather(self.embedding4, u_j), tf.gather(self.sigma4, u_j)

        mu_i, sigma_i = tf.case([(tf.equal(node_type1, 0), f1), (tf.equal(node_type1, 1), f2),
                              (tf.equal(node_type1, 2), f3), (tf.equal(node_type1, 3), f4)],
         default=None, exclusive=True)
        
        mu_j, sigma_j = tf.case([(tf.equal(node_type2, 0), f5), (tf.equal(node_type2, 1), f6),
                              (tf.equal(node_type2, 2), f7), (tf.equal(node_type2, 3), f8)],
         default=None, exclusive=True)

        sigma_ratio = sigma_j / sigma_i
        trace_fac = tf.reduce_sum(sigma_ratio, 1)
        log_det = tf.reduce_sum(tf.compat.v1.log(sigma_ratio + 1e-11), 1)

        mu_diff_sq = tf.reduce_sum(tf.square(mu_i - mu_j) / sigma_i, 1)

        ij_kl = 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

        sigma_ratio = sigma_i / sigma_j
        trace_fac = tf.reduce_sum(sigma_ratio, 1)
        log_det = tf.reduce_sum(tf.compat.v1.log(sigma_ratio + 1e-11), 1)

        mu_diff_sq = tf.reduce_sum(tf.square(mu_j - mu_i) / sigma_j, 1)

        ji_kl = 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

        kl_distance = 0.5 * (ij_kl + ji_kl)

        return kl_distance
