import sys
import unittest
import random
from helpers.wrappers import load_data, TFClassifier
import numpy as np
import efficientnet.tfkeras
from scipy.stats import bootstrap
from scipy.spatial import distance
from pyswarm import pso

class SearchBasedTest(unittest.TestCase):
    DATA_PATH = 'cifar100'
    MODEL_PATH = 'models/efficientnet.h5'

    def setUp(self):
        self.base_test_data = load_data(self.DATA_PATH)
        self.model_under_test = TFClassifier(self.MODEL_PATH)
        for batch in self.base_test_data:
            preds, probas = self.model_under_test.predict(batch.inputs)
            batch.set_predictions(preds)
            batch.set_probabilities(probas)
        self.swarmsize = 5
        self.maxiter = 3
        self.samples = 2
        self.delta = 0.01
        self.mutation_prob = 0.1
        self.max_tolerance = 0.2 

    def test_robustness_against_white_noise(self):
        def _fitness(Delta, *args):
            model_under_test, batch, failure_rates = args
            batch_size = batch.inputs.shape[0]
            synth_inputs = batch.inputs + np.tile(Delta.reshape(batch.inputs.shape[1:]), (batch_size,1,1,1)) 
            synth_preds, synth_probas = model_under_test.predict(synth_inputs)
            failure_rates.append((orig_preds != synth_preds).mean())
            js_dists = distance.jensenshannon(p=batch.probabilities, q=synth_probas, axis=1)
            fitness = np.mean(np.nan_to_num(js_dists))
            return -fitness
        failure_rates = []
        for _ in range(self.samples):
            batch = random.choice(self.base_test_data)
            orig_preds, orig_probas = self.model_under_test.predict(batch.inputs)
            shape = batch.inputs[0].shape
            mask = np.random.binomial(1, self.mutation_prob, size=np.prod(shape))
            lb =  -self.delta * mask 
            ub = self.delta * mask + 1e-8
            args = (self.model_under_test, batch, failure_rates)
            xopt, fopt = pso(_fitness, lb, ub, swarmsize=self.swarmsize, maxiter=self.maxiter, args=args)
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