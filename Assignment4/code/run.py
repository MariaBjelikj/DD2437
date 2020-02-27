from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet
import os
import glob
import matplotlib.pyplot as plt

# files = glob.glob('trained_rbm/*')
# for f in files:
#     os.remove(f)
#
# files = glob.glob('trained_dbn/*')
# for f in files:
#     os.remove(f)

np.random.seed(21)
ITERATIONS = 20
PLOTTING = True

if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''
    # print("\nStarting a Restricted Boltzmann Machine..")
    #
    # # Iterating over the epochs
    # averages = []
    # for hidden in [500]:
    #     print("\nNumber of hidden units: " + str(hidden))
    #     rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
    #                                      ndim_hidden=hidden,
    #                                      is_bottom=True,
    #                                      image_size=image_size,
    #                                      is_top=False,
    #                                      n_labels=10,
    #                                      batch_size=20
    #                                      )
    #     averages.append(rbm.cd1(visible_trainset=train_imgs, n_iterations=ITERATIONS, plotting=PLOTTING))
    # if PLOTTING:
    #     for i, hidden in enumerate([200, 500]):
    #         plt.plot(range(10, 21), averages[i], label=str(hidden) + " hidden units")
    #     plt.xticks(range(10, 21))
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Average Loss rate")
    #     plt.legend()
    #     plt.show()
    
    ''' deep-belief net '''

    print("\nStarting a Deep Belief Net..")

    dbn = DeepBeliefNet(sizes={"vis": image_size[0] * image_size[1], "hid": 500, "pen": 500, "top": 2000, "lbl": 10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=20
                        )

    ''' greedy layer-wise training '''

    aux = dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=ITERATIONS)
    # plt.plot(range(10, 21), averages[-1], label="Original RBM")
    # plt.plot(range(10, 21), aux, label="Greedy RBM")
    # plt.xticks(range(10, 21))
    # plt.xlabel("Epoch")
    # plt.ylabel("Average Loss rate")
    # plt.legend()
    # plt.show()

    dbn.recognize(train_imgs, train_lbls)

    dbn.recognize(test_imgs, test_lbls)
    #
    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1, 10))
    #     digit_1hot[0, digit] = 1
    #     dbn.generate(digit_1hot, name="rbms")

    ''' fine-tune wake-sleep training '''

    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=ITERATIONS)

    dbn.recognize(train_imgs, train_lbls)

    dbn.recognize(test_imgs, test_lbls)
    #
    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1, 10))
    #     digit_1hot[0, digit] = 1
    #     dbn.generate(digit_1hot, name="dbn")
