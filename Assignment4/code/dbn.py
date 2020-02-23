from util import *
from rbm import RestrictedBoltzmannMachine
import numpy as np
from tqdm import tqdm


class DeepBeliefNet:
    """
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis]
                               `-> [lbl]
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    """

    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {

            'vis--hid': RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                   is_bottom=True, image_size=image_size, batch_size=batch_size),

            'hid--pen': RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"],
                                                   batch_size=batch_size),

            'pen+lbl--top': RestrictedBoltzmannMachine(ndim_visible=sizes["pen"] + sizes["lbl"],
                                                       ndim_hidden=sizes["top"],
                                                       is_top=True, n_labels=n_labels, batch_size=batch_size)
        }

        self.sizes = sizes
        self.image_size = image_size
        self.batch_size = batch_size
        self.n_gibbs_recog = 15
        self.n_gibbs_gener = 600
        self.n_gibbs_wakesleep = 5
        self.print_period = 2000

        return

    def recognize(self, true_img, true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_img: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used
          only for calculating accuracy, not driving the net
        """

        n_samples = true_img.shape[0]

        vis = true_img  # visible layer gets the image data

        lbl = np.ones(true_lbl.shape) / 10.  # start the net by telling you know nothing about labels

        # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top.
        #  In the top RBM, run alternating Gibbs sampling \
        # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
        # NOTE : inferring entire train/test set may require too much compute memory (depends on
        # your system). In that case, divide into mini-batches.

        hidden_activation = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)[1]
        pen_actiavtion = self.rbm_stack["hid--pen"].get_h_given_v_dir(hidden_activation)[1]
        pen_lbl_activation = np.concatenate((pen_actiavtion, lbl), axis=1)
        for _ in tqdm(range(self.n_gibbs_recog)):
            top_activation = self.rbm_stack["pen+lbl--top"].get_h_given_v(pen_lbl_activation)[1]
            pen_lbl_activation = self.rbm_stack["pen+lbl--top"].get_v_given_h(top_activation)[1]

        predicted_lbl = pen_lbl_activation[:, -true_lbl.shape[1]:]
        print("accuracy = %.2f%%" % (100. * np.mean(np.argmax(predicted_lbl, axis=1) == np.argmax(true_lbl, axis=1))))

        return

    def generate(self, true_lbl, name):

        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """

        n_sample = true_lbl.shape[0]

        records = []
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([])
        ax.set_yticks([])

        lbl = true_lbl

        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling
        #  in the top RBM. From the top RBM, drive the network \
        # top to the bottom visible layer (replace 'vis' from random to your generated visible layer).

        vis_ = np.random.choice([0, 1], self.sizes['vis']).reshape(-1, self.sizes['vis'])
        # vis_ = np.random.randn(n_sample, self.sizes['vis'])
        # vis_[vis_ > 1] = 1
        # vis_[vis_ < 0] = 0

        hidden_activation = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_)[1]
        pen_activation = self.rbm_stack["hid--pen"].get_h_given_v_dir(hidden_activation)[1]
        pen_lbl_activation = np.concatenate((pen_activation, lbl), axis=1)

        for _ in tqdm(range(self.n_gibbs_gener)):
            top_activation = self.rbm_stack["pen+lbl--top"].get_h_given_v(pen_lbl_activation)[1]
            pen_lbl_activation = self.rbm_stack["pen+lbl--top"].get_v_given_h(top_activation)[1]
            pen_lbl_activation[:, -lbl.shape[1]:] = lbl[:, :]
            pen_activation_top_bottom = pen_lbl_activation[:, :-lbl.shape[1]]
            hidden_activation_top_bottom = self.rbm_stack["hid--pen"].get_v_given_h_dir(pen_activation_top_bottom)[1]
            vis = self.rbm_stack["vis--hid"].get_v_given_h_dir(hidden_activation_top_bottom)[1]

            records.append([ax.imshow(vis.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True,
                                      interpolation=None)])

        stitch_video(fig, records).save("%s.generate%d.mp4" % ("Videos/" + name, np.argmax(true_lbl)))

        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous
        saved parameters of the entire RBM stack.
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try:

            self.loadfromfile_rbm(loc="trained_rbm", name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm", name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm", name="pen+lbl--top")

        except IOError:
            # DONE
            # [TODO TASK 4.2] use CD-1 to train all RBMs greedily

            print("training vis--hid")
            """ 
            CD-1 training for vis--hid 
            """
            self.rbm_stack["vis--hid"].cd1(vis_trainset, n_iterations)  ##### DONE
            self.savetofile_rbm(loc="trained_rbm", name="vis--hid")

            print("training hid--pen")
            self.rbm_stack["vis--hid"].untwine_weights()
            """ 
            CD-1 training for hid--pen 
            """
            h_ = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_trainset)[1]  ##### DONE
            self.rbm_stack["hid--pen"].cd1(h_, n_iterations)  ##### DONE
            self.savetofile_rbm(loc="trained_rbm", name="hid--pen")

            print("training pen+lbl--top")
            self.rbm_stack["hid--pen"].untwine_weights()
            """ 
            CD-1 training for pen+lbl--top 
            """
            h_2 = self.rbm_stack["hid--pen"].get_h_given_v_dir(h_)[1]  ##### DONE
            h_concatenate = np.concatenate((h_2, lbl_trainset), axis=1)  #### DONE
            self.rbm_stack["pen+lbl--top"].cd1(h_concatenate, n_iterations)  #### DONE
            self.savetofile_rbm(loc="trained_rbm", name="pen+lbl--top")

        return

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("\ntraining wake-sleep..")

        try:

            self.loadfromfile_dbn(loc="trained_dbn", name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn", name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn", name="pen+lbl--top")

        except IOError:

            self.n_samples = vis_trainset.shape[0]
            n_labels = lbl_trainset.shape[1]
            elements = int(self.n_samples / self.batch_size)
            
            for epoch in range(n_iterations):
                for it in tqdm(range(elements)):
                    
                    index_init = int(it % elements)
                    index_stop = int((index_init + 1) * self.batch_size)
                    index_init *= self.batch_size
                    visible_minibatch = vis_trainset[index_init:index_stop, :]
                    label_minibatch = lbl_trainset[index_init:index_stop, :]

                    # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.
                    lbl_hid_wake = self.rbm_stack['vis--hid'].get_h_given_v_dir(visible_minibatch)[1]
                    lbl_pen_wake = self.rbm_stack['hid--pen'].get_h_given_v_dir(lbl_hid_wake)[1]
                    lbl_pen = np.concatenate((lbl_pen_wake, label_minibatch), axis=1)
                    lbl_wake = self.rbm_stack['pen+lbl--top'].get_h_given_v(lbl_pen)[1]
                    
                    # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for
                    #  k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.
                    lbl_neg = lbl_wake
                    lbl_pen_neg = 0
                    for _ in range(self.n_gibbs_wakesleep):
                        lbl_pen_neg = self.rbm_stack['pen+lbl--top'].get_v_given_h(lbl_neg)[1]
                        lbl_neg = self.rbm_stack['pen+lbl--top'].get_h_given_v(lbl_pen_neg)[1]
                    
                    # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.
                    lbl_pen_sleep = lbl_pen_neg[:, :-n_labels]
                    p_hid_sleep, lbl_hid_sleep = self.rbm_stack['hid--pen'].get_v_given_h_dir(lbl_pen_sleep) 
                    p_vis_sleep, lbl_vis_sleep = self.rbm_stack['vis--hid'].get_v_given_h_dir(lbl_hid_sleep)
                    
                    # [TODO TASK 4.3] compute predictions : compute generative predictions
                    #  from wake-phase activations, and recognize predictions from sleep-phase activations.
                    # Note that these predictions will not alter the network activations, we use them
                    # only to learn the directed connections.
                    
                    pred_lbl_pen_sleep = self.rbm_stack['hid--pen'].get_h_given_v_dir(lbl_hid_sleep)[1]
                    pred_lbl_hid_sleep = self.rbm_stack['vis--hid'].get_h_given_v_dir(p_vis_sleep)[1]
                    pred_p_vis, pred_lbl_vis = self.rbm_stack['vis--hid'].get_v_given_h_dir(lbl_hid_wake)
                    pred_p_hid, pred_lbl_hid = self.rbm_stack['hid--pen'].get_v_given_h_dir(lbl_pen_wake)
                    
                    # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params'
                    # method from rbm class.
                    self.rbm_stack['vis--hid'].update_generate_params(lbl_hid_wake, visible_minibatch, pred_p_vis)
                    self.rbm_stack['hid--pen'].update_generate_params(lbl_pen_wake, lbl_hid_wake, pred_p_hid)
                    
                    # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params'
                    #  method from rbm class.
                    lbl_pen = np.concatenate((lbl_pen_wake, label_minibatch), axis=1)
                    self.rbm_stack['pen+lbl--top'].update_params(lbl_pen, lbl_wake, lbl_pen_neg, lbl_neg)
                    
                    # [TODO TASK 4.3] update generative parameters : here you will only use 'update_recognize_params'
                    #  method from rbm class.
                    self.rbm_stack['hid--pen'].update_recognize_params(lbl_hid_sleep, lbl_pen_sleep, pred_lbl_pen_sleep)
                    self.rbm_stack['vis--hid'].update_recognize_params(p_vis_sleep, lbl_hid_sleep, pred_lbl_hid_sleep)

                    if it % self.print_period == 0:
                        print("4.3 iteration=%7d" % it)



            self.savetofile_dbn(loc="trained_dbn", name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn", name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn", name="pen+lbl--top")

        return

    def loadfromfile_rbm(self, loc, name):

        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy" % (loc, name))
        self.rbm_stack[name].bias_v = np.load("%s/rbm.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/rbm.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_rbm(self, loc, name):

        np.save("%s/rbm.%s.weight_vh" % (loc, name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
        return

    def loadfromfile_dbn(self, loc, name):

        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy" % (loc, name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy" % (loc, name))
        self.rbm_stack[name].bias_v = np.load("%s/dbn.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/dbn.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_dbn(self, loc, name):

        np.save("%s/dbn.%s.weight_v_to_h" % (loc, name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v" % (loc, name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
        return
