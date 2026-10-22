import unittest

from shared.normalize import amounts_equal, normalize_invoice_number, normalize_order_id, normalize_po_number, parse_amount


class NormalizeTests(unittest.TestCase):
    def test_order_ids(self):
        for raw in ["ORD-10023", "ord-10023", "order #10023", "order number 10023", "#10023", "ORD 10023", " 10023 "]:
            self.assertEqual(normalize_order_id(raw), "ORD-10023", raw)
        for empty in [None, "", "null", "None", "N/A"]:
            self.assertIsNone(normalize_order_id(empty))
        self.assertEqual(normalize_order_id("abc"), "ABC")
        # Order numbers longer than five digits must be kept whole, not truncated.
        for raw in ["123456", "order #123456", "ORD-123456"]:
            self.assertEqual(normalize_order_id(raw), "ORD-123456", raw)

    def test_po_numbers(self):
        for raw in ["PO-5013", "po 5013", "P.O. 5013", "5013", "PO#5013"]:
            self.assertEqual(normalize_po_number(raw), "PO-5013", raw)
        self.assertIsNone(normalize_po_number(None))
        self.assertIsNone(normalize_po_number("null"))

    def test_invoice_numbers(self):
        self.assertEqual(normalize_invoice_number(" grf-14930 "), "GRF-14930")
        self.assertEqual(normalize_invoice_number("LLC-INV- 0420"), "LLC-INV-0420")
        self.assertIsNone(normalize_invoice_number(None))
        self.assertIsNone(normalize_invoice_number("null"))

    def test_amounts(self):
        self.assertEqual(parse_amount("$1,234.50"), 1234.5)
        self.assertEqual(parse_amount("1234.5 CAD"), 1234.5)
        self.assertEqual(parse_amount(99), 99.0)
        self.assertIsNone(parse_amount("lots"))
        self.assertIsNone(parse_amount(None))
        self.assertIsNone(parse_amount(True))
        self.assertTrue(amounts_equal("$1,746.28", 1746.28))
        self.assertTrue(amounts_equal(0.1 + 0.2, 0.3))
        self.assertFalse(amounts_equal(1746.28, 1746.29))
        self.assertFalse(amounts_equal(None, 1.0))


if __name__ == "__main__":
    unittest.main()
