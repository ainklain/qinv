import tensorflow as tf
import os
from qdeep.data_loader.data_generator import DataGenerator
from qdeep.models.funda_model import FundaModel
from qdeep.trainers.funda_trainer import FundaTrainer

from qdeep.utils.config import process_config
from qdeep.utils.dirs import create_dirs
from qdeep.utils.logger import Logger
from qdeep.utils.utils import get_args

ROOT_DIR = os.getcwd()

class Config:
    def __init__(self):
        self.learning_rate = 0.0001
        self.batch_size = 400
        self.n_input = 105
        self.n_hidden1 = 512
        self.n_hidden2 = 512
        self.n_output = 16
        self.max_to_keep = 5
        self.summary_dir = ""
        self.checkpoint_dir = ""
        self.data_file_dir = ""
        self.model_name = "funda-model"
        self.data_file_name = "fundamentals.pkl"
        self.num_epochs = 500
        self.patience = 10

        self.process_config()

    def process_config(self):
        self.summary_dir = os.path.join(ROOT_DIR, "experiments", self.model_name, "summary")
        self.checkpoint_dir = os.path.join(ROOT_DIR, "experiments", self.model_name, "checkpoint")
        self.data_file_dir = os.path.join(ROOT_DIR, "data", self.data_file_name)


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    # try:
    #     args = get_args()
    #     config = process_config(args.config)
    #
    # except:
    #     print("missing or invalid arguments")
    #     exit(0)

    config = Config()

    # create tensorflow session
    sess = tf.Session()

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    latest_checkpoint = tf.train.latest_checkpoint(config.checkpoint_dir)
    if latest_checkpoint:
        saver = tf.train.import_meta_graph(os.path.join(config.checkpoint_dir, config.model_name))
        saver.restore(sess, tf.train.latest_checkpoint(config.checkpoint_dir))
    else:
        # create your data generator
        data = DataGenerator(config)

        # create an instance of the model you want
        # model = ExampleModel(config)

        model = FundaModel(config)

        # create tensorboard logger
        logger = Logger(sess, config)
        # create trainer and pass all the previous components to it
        trainer = FundaTrainer(sess, model, data, config, logger)

    # load model if exists
    # model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()

    # tensorboard --logdir="d:\deepinvest\code\deepinv\experiments\funda-model\summary\train"