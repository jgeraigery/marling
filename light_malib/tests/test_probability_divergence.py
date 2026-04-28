# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import unittest
import torch
import math
import torch.nn.functional as F
from light_malib.utils.probability_divergence import logprobs_to_probs, KL_divergence, JS_divergence

class TestProbabilityDivergence(unittest.TestCase):
    
    def setUp(self):
        self.logits = torch.randn(3, 4)
        self.probs = F.softmax(self.logits, dim=1)
        self.logprobs = torch.log(self.probs)
    
    def test_logprobs_to_probs(self):
        probs = logprobs_to_probs(self.logprobs)
        self.assertTrue(torch.allclose(probs, self.probs))
    
    def test_KL_divergence(self):
        p = F.softmax(torch.randn(1, 5), dim=1)
        q = F.softmax(torch.randn(1, 5), dim=1)
        kl_div = KL_divergence(p, q)
        self.assertTrue(kl_div.shape == (1,))

    # test KL divergence going to zero
    def test_KL_divergence_zero(self):
        p = F.softmax(torch.tensor([[0.3, 0.3, 0.3, 0.1]]), dim=1)
        q = F.softmax(torch.tensor([[0.3, 0.3, 0.3, 0.1]]), dim=1)
        kl_div = KL_divergence(p, q)
        self.assertTrue(kl_div == 0.0)
    
    def test_JS_divergence(self):
        prob_dists = F.softmax(torch.randn(10, 20), dim=1)
        js_divergence = JS_divergence(prob_dists)
        assert js_divergence >= 0.0 and js_divergence <= math.log(prob_dists.shape[0], 2), f'JS divergence: {js_divergence} and upper bound: {math.log(prob_dists.shape[0], 2)}'

    # Test four probability distributions which are close to each other and have the highest relative probability for the same index
    def test_similar_probabilities(self):
        prob_dists = F.softmax(torch.tensor([[0.3, 0.3, 0.3, 0.1], [0.3, 0.3, 0.3, 0.1], [0.3, 0.3, 0.3, 0.1]]), dim=1)
        js_divergence = JS_divergence(prob_dists)
        self.assertTrue(torch.allclose(js_divergence, torch.tensor([0.0])))

    def test_close_probabilities(self):
        prob_dists = F.softmax(torch.tensor([[0.3, 0.1, 0.5, 0.1], [0.1, 0.3, 0.5, 0.1], [0.2, 0.2, 0.5, 0.1]]), dim=1)
        js_divergence = JS_divergence(prob_dists)