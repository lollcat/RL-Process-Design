import tensorflow as tf
from Env.Simulator_New import Simulator
from tester import Tester
from tensorflow.keras.models import load_model
from P_actor import ParameterAgent
from DQN import DQN_Agent
from tensorflow.keras.utils import plot_model
from utils import Visualiser
import matplotlib.pyplot as plt
import matplotlib

with tf.device('/CPU:0'):
    param_model = load_model("param_model.h5")
    dqn_model = load_model("dqn_model.h5")

    #plot_model(dqn_model, to_file='DQNmodel.png', show_shapes=True)
    tester = Tester(param_model, dqn_model, Simulator())
    env = tester.test()

    BFD = Visualiser(env).visualise()
    matplotlib.rcParams['figure.dpi']= 800
    fig, ax = plt.subplots()
    ax.imshow(BFD)
    fig.savefig("BFD")
