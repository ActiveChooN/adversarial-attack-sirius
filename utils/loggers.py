from comet_ml import Experiment
import logging
import os


class BaseLogger:
    def __init__(self, log_interval, train_len):
        self.log_interval = log_interval
        self.train_len = train_len
        self.metrics = {}
        self.__cometml_api_key = os.environ.get("COMETML_API_KEY")
        if self.__cometml_api_key:
            self.experiment = Experiment(self.__cometml_api_key,
                                         project_name="sirius-adversarial-attack")

    def log_test(self, engine):
        self.metrics = engine.state.metrics
        logging.info('----------------------------------------')
        logging.info(f'TEST: Epoch:[{engine.state.epoch}]')
        logging.info(f', '.join([
            f'{name}: {self.metrics[name]:.5f}' for name in self.metrics
        ]))
        logging.info(f'----------------------------------------')
        if self.__cometml_api_key:
            self.experiment.log_metrics(self.metrics, prefix='Test')

    def log_train(self, engine):
        if engine.state.iteration % self.train_len % self.log_interval == 0:
            logging.info("Epoch[{}] Iteration[{}/{}] Loss: {:.5f}"
                         .format(engine.state.epoch, engine.state.iteration %
                                 self.train_len if engine.state.iteration
                                 % self.train_len else self.train_len,
                                 self.train_len, engine.state.output))
            if self.__cometml_api_key:
                logs_dict = {"loss": engine.state.output}
                self.experiment.log_metrics(logs_dict,
                                            step=engine.state.iteration,
                                            epoch=engine.state.epoch,
                                            prefix='Train')

    def log_hparams(self, hparams_dict):
        hparams_dict.update(self.metrics)
        if self.__cometml_api_key:
            self.experiment.log_parameters(hparams_dict)
