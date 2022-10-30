import sys
import unittest
import random
from helpers.wrappers import load_data, TFClassifier
import numpy as np
import efficientnet.tfkeras
from scipy.stats import bootstrap

class RandomTest(unittest.TestCase):
    DATA_PATH = 'cifar100'
    MODEL_PATH = 'models/efficientnet.h5'

    def setUp(self):
        self.base_test_data = load_data(self.DATA_PATH)
        self.model_under_test = TFClassifier(self.MODEL_PATH)
        for batch in self.base_test_data:
            preds, probas = self.model_under_test.predict(batch.inputs)
            batch.set_predictions(preds)
            batch.set_probabilities(probas)
        self.repeats = 3
        self.samples = 2
        self.delta = 0.01
        self.mutation_prob = 0.1
        self.max_tolerance = 0.2 

    def test_robustness_against_white_noise(self):
        failure_rates = []
        for _ in range(self.samples):
            batch = random.choice(self.base_test_data)
            shape = batch.inputs[0].shape
            batch_size = batch.inputs.shape[0]
            for _ in range(self.repeats):
                U = np.random.uniform(size=shape)*2*self.delta - self.delta
                mask = np.random.binomial(1, self.mutation_prob, size=shape)
                mutation = np.tile(mask * U, (batch_size,1,1,1))
                mutated_inputs = batch.inputs + mutation
                synth_predictions, _ = self.model_under_test.predict(mutated_inputs)
                failure_rates.append((batch.predictions != synth_predictions).mean())
        #convert array to sequence
        failure_rates = (failure_rates,)
        #calculate 95% bootstrapped confidence interval for mean
        bootstrap_ci = bootstrap(failure_rates, np.mean, confidence_level=0.95, random_state=1, method='percentile')
        self.assertLess(bootstrap_ci.confidence_interval.high, self.max_tolerance)

    def tearDown(self):
        self.base_test_data = []
        self.model_under_test = None

if __name__ == '__main__':
    if len(sys.argv) > 1:
        RandomTest.DATA_PATH = sys.argv.pop()
        RandomTest.MODEL_PATH = sys.argv.pop()
    unittest.main()