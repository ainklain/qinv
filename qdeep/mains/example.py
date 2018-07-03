import tensorflow as tf

from qdeep.data_loader.data_generator import DataGenerator
from qdeep.models.example_model import ExampleModel
from qdeep.models.mom_model import AEModel, AE_DNNModel
from qdeep.trainers.example_trainer import ExampleTrainer
from qdeep.utils.config import process_config
from qdeep.utils.dirs import create_dirs
from qdeep.utils.logger import Logger
from qdeep.utils.utils import get_args

class Config:
    def __init__(self):
        self.learning_rate = 0.0001
        self.batch_size = 400
        self.n_input = 33
        self.n_hidden1 = 40
        self.n_hidden2 = 4
        self.n_hidden_dnn = 40
        self.n_output = 2
        self.max_to_keep = 5
        self.summary_dir = "./summary"
        self.checkpoint_dir = "./ckpt"
        self.num_epochs = 5
        self.num_epochs_all = 20
        self.num_iter_per_epoch = 400

config = Config()

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)
    
    # create an instance of the model you want
    # model = ExampleModel(config)
    model_ae = AEModel(config)
    model_dnn = AE_DNNModel(config)
    models = {'model_ae': model_ae, 'model_dnn': model_dnn}


    # create tensorboard logger
    # logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = ExampleTrainer(sess, models, data, config, logger)
    #load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
