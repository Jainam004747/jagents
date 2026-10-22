CREATE TABLE menu (id INTEGER PRIMARY KEY, name TEXT NOT NULL, price_cents INTEGER, category TEXT, calories INTEGER);
INSERT INTO menu (name, price_cents, category, calories) VALUES
  ('coffee', 199, 'drink', 5),
  ('tea', 159, 'drink', 2),
  ('latte', 449, 'drink', 190),
  ('cappuccino', 429, 'drink', 120),
  ('espresso', 249, 'drink', 3),
  ('hot chocolate', 299, 'drink', 270),
  ('donut', 129, 'food', 260),
  ('bagel', 249, 'food', 280),
  ('muffin', 219, 'food', 410),
  ('cookie', 149, 'food', 220),
  ('croissant', 279, 'food', 300),
  ('timbits', 99, 'food', 70);
