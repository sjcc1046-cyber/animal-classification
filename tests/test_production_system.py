import unittest

from src.production_system import ANIMALS, EXAMPLES, create_engine


class ProductionSystemTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine()

    def test_all_reference_cases(self):
        self.assertEqual(set(EXAMPLES), set(ANIMALS))
        for expected, facts in EXAMPLES.items():
            with self.subTest(animal=expected):
                result = self.engine.infer(facts)
                self.assertEqual(result.status, "recognized")
                self.assertEqual(result.animal, expected)
                self.assertGreaterEqual(len(result.fired_rules), 2)

    def test_incomplete_facts_are_unknown(self):
        result = self.engine.infer({"有毛发", "家养"})
        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.animal)

    def test_conflicting_facts_are_reported_as_ambiguous(self):
        result = self.engine.infer({"有毛发", "吃肉", "家养", "会喵叫", "会汪叫"})
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(set(result.candidates), {"猫", "狗"})

    def test_repeated_facts_do_not_repeat_rules(self):
        result = self.engine.infer(["有毛发", "有毛发", "吃肉", "家养", "会喵叫"])
        ids = [item.rule.rule_id for item in result.fired_rules]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
