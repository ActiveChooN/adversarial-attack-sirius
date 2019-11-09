from comet_ml import Experiment
import logging
import os


class BaseLogger:
    def __init__(self, log_interval):
        self.log_interval = log_interval
        self.step = 0
        self._final_accuracy = 0.
        self._final_loss = 0.
        self.__cometml_api_key = os.environ.get("COMETML_API_KEY")
        if self.__cometml_api_key:
            self.experiment = Experiment(self.__cometml_api_key,
                                         project_name="sirius-adversarial-attack")

    def log_test(self, logs_dict):
        if not ("accuracy" in logs_dict.keys() and "loss" in logs_dict.keys()):
            raise Exception("Loss and Accuracy are necessary parameters")
        self._final_accuracy = logs_dict["accuracy"]
        self._final_loss = logs_dict["loss"]
        logging.info(f'----------------------------------------\n'
                     f'TEST: Avg. loss: {logs_dict["loss"]:.5f}, '
                     f'Accuracy: {logs_dict["accuracy"]:.5f}\n'
                     f'----------------------------------------')
        if self.__cometml_api_key:
            self.experiment.log_metrics(logs_dict, prefix='Test')

    def log_train(self, logs_dict):
        if not ("loss" in logs_dict.keys() and "epoch" in logs_dict.keys() and
                "progress" in logs_dict.keys()):
            raise Exception("Loss, epoch and progress are necessary parameters")
        if self.step % self.log_interval == 0:
            progress = logs_dict.pop("progress")
            epoch = logs_dict.pop("epoch")
            logging.info(f'Epoch: {epoch}, progress: {progress:.0f}%, '
                         f'loss {logs_dict["loss"]:.5f}')
            if self.__cometml_api_key:
                self.experiment.log_metrics(logs_dict, step=self.step,
                                            epoch=epoch, prefix='Train')
        self.step += 1

    def log_hparams(self, hparams_dict):
        hparams_dict.update({'Loss': self._final_loss,
                             'Accuracy': self._final_accuracy})
        if self.__cometml_api_key:
            self.experiment.log_parameters(hparams_dict)
