# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import unittest
import torch
from light_malib.model.gr_football.tizero.positional_encoding import PositionalEncoding

class TestPositionalEncoding(unittest.TestCase):

    def setUp(self):
        self.d_model = 512
        self.max_len = 5000
        self.pos_enc = PositionalEncoding(self.d_model, self.max_len)

    def test_init(self):
        self.assertEqual(self.pos_enc.pe.size(), (self.max_len, self.d_model))

    def test_forward(self):
        x = torch.randint(0, self.max_len, (1,))
        output = self.pos_enc(x)
        self.assertEqual(output.size(), (1, self.d_model))

    # A test where numbers from 0 to 10 are passed to the positional encoding and the results are checked to be different from each other with the right shape
    def test_forward_different_numbers(self):
        x = torch.arange(11)
        output = self.pos_enc(x)
        self.assertEqual(output.size(), (11, self.d_model))
        self.assertFalse(torch.allclose(output[0], output[1]))
        self.assertFalse(torch.allclose(output[1], output[2]))
        self.assertFalse(torch.allclose(output[2], output[3]))
        self.assertFalse(torch.allclose(output[3], output[4]))
        self.assertFalse(torch.allclose(output[4], output[5]))
        self.assertFalse(torch.allclose(output[5], output[6]))
        self.assertFalse(torch.allclose(output[6], output[7]))
        self.assertFalse(torch.allclose(output[7], output[8]))
        self.assertFalse(torch.allclose(output[8], output[9]))
        self.assertFalse(torch.allclose(output[9], output[10]))


if __name__ == '__main__':
    unittest.main()