import unittest
import tensorflow

from truthyfalseyclassifier import TruthyFalseyClassifier

class TestTruthyFalseyTester(unittest.TestCase):
    def test_falsey_values(self):
        falsey_values = ['', 0, [], 0.0, False, {}, None]
        map(lambda falsey_value: self.assertFalse(falsey_value), falsey_values)

    def test_truthy_values(self):
        truthy_values = ['a', 1, [1], 1.0, True, {'x': '5'}]
        map(lambda truthy_value: self.assertTrue(truthy_value), truthy_values)

if __name__ == '__main__':
    unittest.main()