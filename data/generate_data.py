"""Generate all the fake data for the project.

Creates (inside the data/ folder by default):
  business.db      SQLite database: customers, orders, purchase orders
  items.jsonl      The inbox: customer emails and supplier invoices (what the agent sees)
  labels.jsonl     The correct answer for every item (what the eval compares against)
  invoices/*.pdf   Supplier invoice PDFs attached to some emails

Everything is generated from a fixed random seed, so running it again
produces exactly the same data.

Run:  python -m data.generate_data
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

DATA_DIR = Path(__file__).resolve().parent
SEED = 42
STORE_NAME = "Cedar Lane Home Goods"
STORE_INBOX = "inbox@cedarlane.example"

FIRST_NAMES = [
    "Olivia", "Liam", "Emma", "Noah", "Ava", "Ethan", "Sophia", "Lucas", "Mia", "Mason",
    "Isabella", "Logan", "Amelia", "James", "Harper", "Benjamin", "Evelyn", "Jacob", "Abigail",
    "Daniel", "Chloe", "Henry", "Grace", "Owen", "Zoe", "Samuel", "Leah", "Wyatt", "Nora", "Arjun",
]
LAST_NAMES = [
    "Tremblay", "Gagnon", "Roy", "Cote", "Bouchard", "Gauthier", "Morin", "Lavoie", "Fortin",
    "Gagne", "Ouellet", "Pelletier", "Belanger", "Levesque", "Bergeron", "Leblanc", "Paquette",
    "Girard", "Simard", "Boucher", "Caron", "Beaulieu", "Cloutier", "Dube", "Poirier", "Fournier",
    "Lapointe", "Leclerc", "Lefebvre", "Singh",
]
CITIES = ["Kamloops", "Kelowna", "Vancouver", "Victoria", "Calgary", "Edmonton", "Prince George", "Nanaimo"]

PRODUCTS = [
    ("oak dining table", 899.00), ("velvet armchair", 449.00), ("queen bed frame", 649.00),
    ("standing desk", 529.00), ("floor lamp", 129.00), ("bookshelf", 219.00),
    ("area rug", 299.00), ("office chair", 349.00), ("nightstand", 159.00),
    ("sectional sofa", 1799.00), ("coffee table", 279.00), ("bar stool set", 239.00),
    ("wall mirror", 189.00), ("TV stand", 399.00), ("patio chair set", 589.00),
]

# Each supplier formats its invoices differently, like in real life.
SUPPLIERS = [
    {"name": "Birchwood Supply Ltd.", "domain": "birchwoodsupply.example",
     "inv_label": "Invoice No.", "po_label": "PO Number", "total_label": "TOTAL DUE",
     "goods": ["Oak lumber (kiln dried)", "Walnut veneer sheets", "Maple table legs", "Wood finish (4L)"]},
    {"name": "Granite Ridge Fabrics", "domain": "graniteridgefabrics.example",
     "inv_label": "Invoice #", "po_label": "Purchase Order", "total_label": "Total",
     "goods": ["Velvet upholstery fabric", "Linen blend fabric", "Foam cushion inserts", "Thread spools"]},
    {"name": "Summit Hardware Wholesale", "domain": "summithardware.example",
     "inv_label": "Invoice Number", "po_label": "Your PO", "total_label": "Balance Due", "po_style": "dotted",
     "goods": ["Drawer slides (pairs)", "Cabinet hinges", "Assorted screws (box)", "Steel brackets"]},
    {"name": "Lakeshore Lighting Co.", "domain": "lakeshorelighting.example",
     "inv_label": "Inv. No.", "po_label": "Customer PO #", "total_label": "Invoice Total", "po_style": "plain",
     "goods": ["LED bulbs (case)", "Lamp shades", "Brass lamp bases", "Dimmer switches"]},
    {"name": "Pinecrest Packaging Inc.", "domain": "pinecrestpackaging.example",
     "inv_label": "Invoice Number", "po_label": "PO Number", "total_label": "Total Due",
     "goods": ["Shipping boxes (large)", "Foam corner protectors", "Stretch wrap rolls", "Packing tape (case)"]},
]

ORDER_COUNT = 80
PO_COUNT = 25


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def money(x: float) -> str:
    return f"${x:,.2f}"


def order_ref(rng: random.Random, order_id: str) -> str:
    """Customers write order numbers in many different ways."""
    digits = order_id.split("-")[1]
    return rng.choice([order_id, order_id, f"order #{digits}", f"order number {digits}", order_id.lower(), f"#{digits}"])


def make_invoice_number(supplier_index: int, rng: random.Random) -> str:
    n = rng.randint(100, 9999)
    formats = [
        f"BW-2026-{n:04d}",
        f"GRF-{rng.randint(10000, 19999)}",
        f"SHW-26-{n:04d}",
        f"LLC-INV-{n:04d}",
        f"PP26-{n:04d}",
    ]
    return formats[supplier_index]


# ---------------------------------------------------------------------------
# Business records
# ---------------------------------------------------------------------------

def build_customers(rng: random.Random) -> list[dict]:
    customers = []
    for i in range(30):
        first, last = FIRST_NAMES[i], LAST_NAMES[i]
        customers.append({
            "id": f"C{i + 1:03d}",
            "name": f"{first} {last}",
            "first": first,
            "email": f"{first.lower()}.{last.lower()}@example.com",
            "city": rng.choice(CITIES),
        })
    return customers


def build_orders(rng: random.Random, customers: list[dict]) -> list[dict]:
    orders = []
    start = date(2026, 8, 1)
    for i in range(ORDER_COUNT):
        product, price = rng.choice(PRODUCTS)
        # First 40 delivered, next 25 shipped, last 15 processing.
        status = "delivered" if i < 40 else ("shipped" if i < 65 else "processing")
        orders.append({
            "id": f"ORD-{10001 + i}",
            "customer_id": rng.choice(customers)["id"],
            "product": product,
            "amount": price,
            "status": status,
            "order_date": (start + timedelta(days=rng.randint(0, 60))).isoformat(),
        })
    return orders


def build_purchase_orders(rng: random.Random) -> list[dict]:
    pos = []
    for i in range(PO_COUNT):
        s = i % len(SUPPLIERS)
        pos.append({
            "po_number": f"PO-{5001 + i}",
            "supplier_index": s,
            "supplier": SUPPLIERS[s]["name"],
            "amount": round(rng.randint(40000, 900000) / 100, 2),
            "issued_date": (date(2026, 8, 1) + timedelta(days=rng.randint(0, 30))).isoformat(),
            "status": "open",
        })
    return pos


def write_database(path: Path, customers, orders, pos) -> None:
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE customers (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE, city TEXT
        );
        CREATE TABLE orders (
            id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(id),
            product TEXT NOT NULL, amount REAL NOT NULL, status TEXT NOT NULL, order_date TEXT NOT NULL
        );
        CREATE TABLE purchase_orders (
            po_number TEXT PRIMARY KEY, supplier TEXT NOT NULL, amount REAL NOT NULL,
            issued_date TEXT NOT NULL, status TEXT NOT NULL
        );
        """
    )
    con.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)",
                    [(c["id"], c["name"], c["email"], c["city"]) for c in customers])
    con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)",
                    [(o["id"], o["customer_id"], o["product"], o["amount"], o["status"], o["order_date"]) for o in orders])
    con.executemany("INSERT INTO purchase_orders VALUES (?, ?, ?, ?, ?)",
                    [(p["po_number"], p["supplier"], p["amount"], p["issued_date"], p["status"]) for p in pos])
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Invoice PDFs
# ---------------------------------------------------------------------------

def split_into_lines(rng: random.Random, subtotal_cents: int, goods: list[str]) -> list[tuple[str, int, int, int]]:
    """Split a subtotal into 2-3 invoice lines: (description, qty, unit_cents, line_cents).
    The line totals always add up exactly to subtotal_cents."""
    n = rng.choice([2, 3])
    descriptions = rng.sample(goods, n)
    while True:  # pick cut points so that no line is smaller than 10% of the subtotal
        cuts = sorted(rng.sample(range(1, subtotal_cents), n - 1))
        parts = [b - a for a, b in zip([0] + cuts, cuts + [subtotal_cents])]
        if min(parts) >= subtotal_cents * 0.10:
            break
    lines = []
    for desc, line_cents in zip(descriptions, parts):
        qty = next((q for q in (10, 5, 4, 2) if line_cents % q == 0 and line_cents // q >= 100), 1)
        lines.append((desc, qty, line_cents // qty, line_cents))
    return lines


def write_invoice_pdf(path: Path, rng: random.Random, supplier: dict, invoice_number: str,
                      po_number: str, total: float, invoice_date: date) -> None:
    total_cents = round(total * 100)
    subtotal_cents = round(total_cents / 1.05)  # 5% GST
    tax_cents = total_cents - subtotal_cents
    lines = split_into_lines(rng, subtotal_cents, supplier["goods"])

    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER
    y = height - 60
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, supplier["name"])
    c.setFont("Helvetica", 9)
    c.drawString(50, y - 14, f"accounts@{supplier['domain']}  |  GST Reg. 8{rng.randint(10000000, 99999999)}")

    c.setFont("Helvetica-Bold", 22)
    c.drawRightString(width - 50, y, "INVOICE")

    y -= 60
    c.setFont("Helvetica", 10)
    po_digits = po_number.split("-")[1]
    po_display = {"plain": po_digits, "dotted": f"P.O. {po_digits}"}.get(supplier.get("po_style"), po_number)
    meta = [
        (supplier["inv_label"], invoice_number),
        ("Date", invoice_date.strftime("%B %d, %Y")),
        (supplier["po_label"], po_display),
        ("Terms", "Net 30"),
    ]
    for label, value in meta:
        c.drawString(width - 250, y, f"{label}:")
        c.drawString(width - 140, y, value)
        y -= 15

    c.drawString(50, y + 45, "Bill To:")
    c.drawString(50, y + 30, STORE_NAME)
    c.drawString(50, y + 15, "Accounts Payable")

    y -= 30
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Description")
    c.drawRightString(370, y, "Qty")
    c.drawRightString(460, y, "Unit Price")
    c.drawRightString(width - 50, y, "Amount")
    c.line(50, y - 5, width - 50, y - 5)
    c.setFont("Helvetica", 10)
    for desc, qty, unit_cents, line_cents in lines:
        y -= 20
        c.drawString(50, y, desc)
        c.drawRightString(370, y, str(qty))
        c.drawRightString(460, y, money(unit_cents / 100))
        c.drawRightString(width - 50, y, money(line_cents / 100))

    y -= 30
    c.line(330, y + 12, width - 50, y + 12)
    for label, cents, bold in (("Subtotal", subtotal_cents, False), ("GST (5%)", tax_cents, False),
                               (supplier["total_label"], total_cents, True)):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 11 if bold else 10)
        c.drawString(340, y, label)
        c.drawRightString(width - 50, y, money(cents / 100))
        y -= 18

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 60, "Thank you for your business. Please reference the invoice number with your payment.")
    c.save()


# ---------------------------------------------------------------------------
# Email templates
# ---------------------------------------------------------------------------

REFUND_TEMPLATES = [
    ("Refund request for {ref}",
     "Hi,\n\nI received my {product} last week but it isn't what I expected. I'd like to return it and get a refund. My order is {ref}.\n\nThanks,\n{name}"),
    ("Damaged item - want my money back",
     "Hello Cedar Lane team,\n\nThe {product} from {ref} arrived with a cracked leg. I don't want a replacement, I just want a refund please.\n\n{first}"),
    ("return",
     "hey so the {product} doesnt fit in my space at all. can i send it back and get refunded?? its {ref}\n\n- {first}"),
    ("Requesting a refund",
     "Good afternoon,\n\nI am writing to request a full refund for {ref}. The colour of the {product} is quite different from the photos on your website.\n\nKind regards,\n{name}"),
    ("{ref} - changed my mind",
     "Hi there, I ordered a {product} but I've changed my mind and would like to return it for a refund. It's still in the box. Thanks! {name}"),
    ("Wrong item received",
     "Hello,\n\nI ordered a {product} ({ref}) but the box contained something completely different. Please refund me, I no longer need it.\n\n{name}"),
    ("Refund for {ref}",
     "Please process a refund, the product stopped working after two days. Thanks, {first}"),
]

REFUND_NO_ORDER_TEMPLATES = [
    ("Falling apart already",
     "Hi, I bought a {product} from you a couple of weeks ago and it's falling apart already. I want a refund. I can't find my order confirmation email though.\n\n{name}"),
    ("money back please",
     "I want my money back for the lamp I bought. It flickers constantly. {first}"),
    ("Refund",
     "Hello, please refund my last purchase, I'm not happy with the quality. Thanks, {name}"),
]

SHIPPING_TEMPLATES = [
    ("Where is my order?",
     "Hi, I placed {ref} almost three weeks ago and it still hasn't arrived. Can you tell me what's going on?\n\n{name}"),
    ("Delivery delay {ref}",
     "Hello, my tracking for {ref} hasn't updated in 8 days. Is my {product} lost?\n\n{first}"),
    ("still waiting",
     "my {product} was supposed to come last monday. its {ref}. nobody has contacted me. please help - {first}"),
    ("Late delivery",
     "Good morning,\n\nI need the {product} from {ref} before my move on Saturday, but the delivery date keeps getting pushed back. Can you give me an update?\n\nRegards,\n{name}"),
    ("Order not received - {ref}",
     "Hi, the website says my order shipped but I haven't received anything yet. Could you look into it? Thanks, {name}"),
]

SHIPPING_NO_ORDER_TEMPLATES = [
    ("Package is late",
     "Hi, my package is very late, it was supposed to arrive last week. When will it come?\n\n{name}"),
    ("where is my stuff",
     "where is my stuff?? ordered ages ago and still nothing - {first}"),
]

BILLING_ORDER_TEMPLATES = [
    ("Charged twice",
     "Hello, I was charged twice for {ref}. I can see two identical payments on my credit card statement. Can you check?\n\n{name}"),
    ("Question about my receipt",
     "Hi, the total on my receipt for {ref} doesn't match what I saw at checkout. Can you explain the difference? Thanks, {first}"),
    ("Discount code not applied",
     "Hi, I used a 10% off code for {ref} but it doesn't look like it was applied to the total. Can someone check?\n\n{name}"),
    ("Need a receipt",
     "Hello, could you send me an official receipt for {ref}? I need it for my company's expense report.\n\n{name}"),
]

BILLING_NO_ORDER_TEMPLATES = [
    ("Update payment method",
     "Hi, how do I update the credit card saved on my account? My old one expired.\n\n{name}"),
    ("Charge I don't recognise",
     "Hello, I see a charge from Cedar Lane Home Goods on my statement that I don't recognise. Can you tell me what it was for?\n\n{name}"),
    ("Tax question",
     "Hi, quick question - are prices on your site shown before or after tax? I'm trying to figure out my bill. {first}"),
    ("Gift card balance",
     "Hello, can you tell me the remaining balance on my gift card? I'm not sure how much I have left. Thanks, {name}"),
]

# Harder messages: no obvious keywords, unusual order number formats, word traps.
REFUND_HARD_TEMPLATES = [
    ("Not happy with my purchase",
     "Hi, the {product} I got ({ref}) wobbles no matter what I do. I'd like to send it back and be reimbursed.\n\n{name}"),
    ("Re: {ref}",
     "Can I get my money back for this? The {product} smells strongly of chemicals and I can't keep it in the house. - {first}"),
    ("{product}",
     "Hello, my {product} was delivered yesterday but one of the drawers is broken. I don't want a replacement, please reverse the payment. Order ORD {digits}.\n\n{name}"),
    ("Return label please",
     "Hi, I'd like to return my {product} for a full refund. Could you send me a return label? It was order {digits}.\n\n{name}"),
    ("Unhappy customer",
     "I've been a loyal customer for years but this {product} is terrible quality. I expect a full reimbursement for {ref}.\n\n{name}"),
]

SHIPPING_HARD_TEMPLATES = [
    ("Any update?",
     "Hi, any news on {ref}? The status on your site hasn't changed in two weeks. {first}"),
    ("Sofa still isn't here",
     "Hello, it has been almost a month and my {product} still isn't here. Order {ref}. I'm starting to get worried.\n\n{name}"),
    ("Missing package",
     "The tracking number you sent me doesn't work. Can you check where {ref} is? Thanks, {name}"),
    ("When will my order ship?",
     "Hi, I ordered a {product} ({ref}) and I haven't gotten a shipping confirmation yet. When will it go out?\n\n{name}"),
    ("Still waiting on {ref}",
     "Hi, my {product} hasn't come yet. I don't want a refund, I still want it! I just need to know when it will arrive.\n\n{name}"),
]

BILLING_HARD_TEMPLATES = [
    ("Price dropped",
     "Hi, I bought the {product} ({ref}) last week and now it's on sale for $50 less. Can you give me the difference as a price adjustment?\n\n{name}"),
    ("Payment declined?",
     "My card was declined at checkout but my bank says the money was taken. Did my payment go through?\n\n{name}"),
    ("Gift card not working",
     "Hello, I received a Cedar Lane gift card as a present but it says invalid when I enter it at checkout. Can you help?\n\n{name}"),
    ("Copy of invoice for {ref}",
     "Hi, could you email me a copy of the invoice for {ref}? I need it for my records. Thanks, {first}"),
]

INVOICE_TEMPLATES = [
    ("Invoice {inv} from {supplier}",
     "Hello,\n\nPlease find attached invoice {inv} for your recent order. Payment terms are net 30.\n\nBest regards,\nAccounts Receivable\n{supplier}"),
    ("{supplier} - invoice attached",
     "Hi Cedar Lane accounts team,\n\nAttached is our invoice for {po}. Let us know if you have any questions.\n\nThanks,\n{supplier}"),
    ("New invoice",
     "Good day,\n\nYour invoice is attached. Kindly process at your earliest convenience.\n\n{supplier} Billing Department"),
    ("Invoice for recent deliveries",
     "Hello,\n\nAttached please find our invoice covering the latest delivery. Reference details are in the PDF.\n\nRegards,\n{supplier}"),
]

SPAM_MESSAGES = [
    ("promo@deals-blast.example", "90% OFF everything - today only!!!",
     "Don't miss out! Click here to claim your exclusive discount on luxury watches. Unsubscribe anytime."),
    ("security@paypa1-alerts.example", "Refund pending: verify your account",
     "Dear customer, you have a pending refund of $349.00. To receive it, confirm your banking details within 24 hours at the link below or your refund will be cancelled."),
    ("growth@rankfast-seo.example", "Rank #1 on Google",
     "Hello, I noticed your website is not ranking on page one. Our SEO team can guarantee top rankings for just $99/month. Reply YES to get started."),
    ("claims@intl-lottery-board.example", "Congratulations! You have won",
     "Your email was selected in our international lottery. To claim your $2,500,000 prize, send your full name, address and a small processing fee."),
    ("invest@ai-trader-pro.example", "Turn $250 into $25,000",
     "Our AI trading bot made investors 100x returns this month. Limited spots remaining. Join now before it's too late."),
    ("billing@quick-payments.example", "Invoice overdue - action required",
     "Your invoice #99812 is overdue. Download the file from the link below within 12 hours to avoid service interruption."),
    ("hello@starcollab-agency.example", "Partnership opportunity",
     "Hi! We're an influencer agency with 2M followers. Let's collab - send us 10 free sofas and we'll post about your brand."),
    ("admin@mail-storage-alert.example", "Your mailbox is full",
     "Your email storage is 99% full. Log in at the link below to upgrade or you will start losing messages."),
    ("jobs@easy-cash-home.example", "Work from home - $5000/week",
     "Earn $5,000 a week from home with no experience needed. Click to learn the secret banks don't want you to know."),
    ("delivery@post-track-notice.example", "Your package could not be delivered",
     "We attempted to deliver your package today. Pay the $1.99 redelivery fee at the link below to reschedule."),
    ("rewards@giftcard-winners.example", "You've been chosen!",
     "You've been chosen to receive a $500 Cedar Lane gift card! Just complete this short survey and enter your credit card details to cover shipping."),
    ("offers@fastbiz-funding.example", "Business loan pre-approved",
     "Your business has been pre-approved for a $150,000 loan. No credit check required. Reply now to claim your funds."),
    ("mark@webfixers.example", "Quick question about your website",
     "Hi there, I was browsing your website and found several errors that may be costing you customers. Want me to send over a free audit?"),
    ("orders@cedarlane-support.example", "Re: Order confirmation",
     "We were unable to process your recent order. Please update your payment information within 24 hours using the secure portal below."),
    ("sales@globalimport-deals.example", "Furniture at factory prices",
     "Dear purchasing manager, we sell furniture at factory prices with worldwide shipping. Contact us today for an exclusive quote."),
]


# ---------------------------------------------------------------------------
# Building the inbox and the answer key
# ---------------------------------------------------------------------------

def generate(out_dir: Path = DATA_DIR, seed: int = SEED) -> dict:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    invoice_dir = out_dir / "invoices"
    invoice_dir.mkdir(exist_ok=True)
    for old in invoice_dir.glob("*.pdf"):
        old.unlink()

    customers = build_customers(rng)
    orders = build_orders(rng, customers)
    pos = build_purchase_orders(rng)
    write_database(out_dir / "business.db", customers, orders, pos)

    cust_by_id = {c["id"]: c for c in customers}
    delivered = [o for o in orders if o["status"] == "delivered"]
    in_transit = [o for o in orders if o["status"] in ("shipped", "processing")]
    rng.shuffle(delivered)
    rng.shuffle(in_transit)
    used_orders: set[str] = set()

    def take(pool: list[dict]) -> dict:
        for o in pool:
            if o["id"] not in used_orders:
                used_orders.add(o["id"])
                return o
        raise RuntimeError("ran out of orders")

    def other_customer(not_id: str) -> dict:
        return rng.choice([c for c in customers if c["id"] != not_id])

    pairs: list[tuple[dict, dict]] = []  # (item, label)

    def add(sender: str, subject: str, body: str, label: dict, attachments: list[str] | None = None) -> None:
        item = {"from": sender, "to": STORE_INBOX, "subject": subject, "body": body,
                "attachments": attachments or []}
        full_label = {"category": None, "action": None, "order_id": None, "invoice_number": None,
                      "po_number": None, "amount": None, "note": ""}
        full_label.update(label)
        pairs.append((item, full_label))

    def fill(template: tuple[str, str], **kw) -> tuple[str, str]:
        return template[0].format(**kw), template[1].format(**kw)

    # --- Refund requests (25) ---
    for i in range(14):
        o = take(delivered)
        c = cust_by_id[o["customer_id"]]
        subj, body = fill(REFUND_TEMPLATES[i % len(REFUND_TEMPLATES)], ref=order_ref(rng, o["id"]),
                          product=o["product"], name=c["name"], first=c["first"])
        add(c["email"], subj, body, {"category": "refund_request", "action": "create_ticket:refund",
                                     "order_id": o["id"], "note": "valid order owned by sender"})
    for t in REFUND_HARD_TEMPLATES:
        o = take(delivered)
        c = cust_by_id[o["customer_id"]]
        subj, body = fill(t, ref=order_ref(rng, o["id"]), digits=o["id"][4:], product=o["product"],
                          name=c["name"], first=c["first"])
        add(c["email"], subj, body, {"category": "refund_request", "action": "create_ticket:refund",
                                     "order_id": o["id"], "note": "valid order owned by sender (harder wording)"})
    for t in REFUND_NO_ORDER_TEMPLATES:
        c = rng.choice(customers)
        subj, body = fill(t, product=rng.choice(PRODUCTS)[0], name=c["name"], first=c["first"])
        add(c["email"], subj, body, {"category": "refund_request", "action": "flag_for_human",
                                     "note": "no order number given"})
    for i in range(2):
        c = rng.choice(customers)
        fake_id = f"ORD-{19000 + rng.randint(100, 999)}"
        subj, body = fill(REFUND_TEMPLATES[i], ref=fake_id, product=rng.choice(PRODUCTS)[0],
                          name=c["name"], first=c["first"])
        add(c["email"], subj, body, {"category": "refund_request", "action": "flag_for_human",
                                     "order_id": fake_id, "note": "order number does not exist"})
    o = take(delivered)
    c = other_customer(o["customer_id"])
    subj, body = fill(REFUND_TEMPLATES[3], ref=o["id"], product=o["product"], name=c["name"], first=c["first"])
    add(c["email"], subj, body, {"category": "refund_request", "action": "flag_for_human",
                                 "order_id": o["id"], "note": "order belongs to a different customer"})

    # --- Shipping delays (21) ---
    for i in range(11):
        o = take(in_transit)
        c = cust_by_id[o["customer_id"]]
        subj, body = fill(SHIPPING_TEMPLATES[i % len(SHIPPING_TEMPLATES)], ref=order_ref(rng, o["id"]),
                          product=o["product"], name=c["name"], first=c["first"])
        add(c["email"], subj, body, {"category": "shipping_delay", "action": "create_ticket:shipping",
                                     "order_id": o["id"], "note": "valid order owned by sender"})
    for t in SHIPPING_HARD_TEMPLATES:
        o = take(in_transit)
        c = cust_by_id[o["customer_id"]]
        subj, body = fill(t, ref=order_ref(rng, o["id"]), digits=o["id"][4:], product=o["product"],
                          name=c["name"], first=c["first"])
        add(c["email"], subj, body, {"category": "shipping_delay", "action": "create_ticket:shipping",
                                     "order_id": o["id"], "note": "valid order owned by sender (harder wording)"})
    # The real customer, but writing from an address we don't have on file.
    o = take(in_transit)
    c = cust_by_id[o["customer_id"]]
    add(f"{c['first'].lower()}.{c['name'].split()[1].lower()}@workmail.example", f"Where is {o['id']}?",
        f"Hi, writing from my work email. My {o['product']} ({o['id']}) was supposed to arrive last week "
        f"and I haven't heard anything. Can you check?\n\n{c['name']}",
        {"category": "shipping_delay", "action": "flag_for_human", "order_id": o["id"],
         "note": "sender email does not match the customer on the order"})
    for t in SHIPPING_NO_ORDER_TEMPLATES:
        c = rng.choice(customers)
        subj, body = fill(t, name=c["name"], first=c["first"])
        add(c["email"], subj, body, {"category": "shipping_delay", "action": "flag_for_human",
                                     "note": "no order number given"})
    c = rng.choice(customers)
    fake_id = f"ORD-{19000 + rng.randint(100, 999)}"
    subj, body = fill(SHIPPING_TEMPLATES[0], ref=fake_id, product="order", name=c["name"], first=c["first"])
    add(c["email"], subj, body, {"category": "shipping_delay", "action": "flag_for_human",
                                 "order_id": fake_id, "note": "order number does not exist"})
    o = take(in_transit)
    c = other_customer(o["customer_id"])
    subj, body = fill(SHIPPING_TEMPLATES[1], ref=o["id"], product=o["product"], name=c["name"], first=c["first"])
    add(c["email"], subj, body, {"category": "shipping_delay", "action": "flag_for_human",
                                 "order_id": o["id"], "note": "order belongs to a different customer"})

    # --- Billing questions (19) ---
    for i in range(8):
        o = take(orders)
        c = cust_by_id[o["customer_id"]]
        subj, body = fill(BILLING_ORDER_TEMPLATES[i % len(BILLING_ORDER_TEMPLATES)], ref=order_ref(rng, o["id"]),
                          name=c["name"], first=c["first"])
        add(c["email"], subj, body, {"category": "billing_question", "action": "create_ticket:billing",
                                     "order_id": o["id"], "note": "billing question about an order"})
    for i in range(7):
        c = rng.choice(customers)
        subj, body = fill(BILLING_NO_ORDER_TEMPLATES[i % len(BILLING_NO_ORDER_TEMPLATES)],
                          name=c["name"], first=c["first"])
        add(c["email"], subj, body, {"category": "billing_question", "action": "create_ticket:billing",
                                     "note": "general billing question"})
    for t in BILLING_HARD_TEMPLATES:
        if "{ref}" in t[0] + t[1]:
            o = take(orders)
            c = cust_by_id[o["customer_id"]]
            subj, body = fill(t, ref=order_ref(rng, o["id"]), product=o["product"], name=c["name"], first=c["first"])
            label = {"order_id": o["id"], "note": "billing question about an order (harder wording)"}
        else:
            c = rng.choice(customers)
            subj, body = fill(t, name=c["name"], first=c["first"])
            label = {"note": "general billing question (harder wording)"}
        add(c["email"], subj, body, {"category": "billing_question", "action": "create_ticket:billing", **label})

    # --- Supplier invoices (25): 17 match, 5 wrong amount, 3 unknown PO ---
    po_pool = pos[:]
    rng.shuffle(po_pool)
    used_invoice_numbers: set[str] = set()
    invoice_cases = ["match"] * 17 + ["mismatch"] * 5 + ["unknown_po"] * 3
    for i, case in enumerate(invoice_cases):
        if case == "unknown_po":
            s_idx = rng.randrange(len(SUPPLIERS))
            po_number = f"PO-{5900 + rng.randint(10, 99)}"
            total = round(rng.randint(40000, 900000) / 100, 2)
            note = "PO number does not exist"
        else:
            po = po_pool.pop()
            s_idx, po_number = po["supplier_index"], po["po_number"]
            if case == "match":
                total, note = po["amount"], "invoice total matches the PO"
            else:
                delta = rng.choice([150.00, 89.50, 1200.00, 12.75, 430.00])
                total = round(po["amount"] + delta, 2)
                note = f"invoice total is {money(delta)} more than the PO amount {money(po['amount'])}"
        supplier = SUPPLIERS[s_idx]
        inv = make_invoice_number(s_idx, rng)
        while inv in used_invoice_numbers:
            inv = make_invoice_number(s_idx, rng)
        used_invoice_numbers.add(inv)
        filename = f"invoices/{inv}.pdf"
        inv_date = date(2026, 9, 10) + timedelta(days=rng.randint(0, 20))  # after POs, before emails arrive
        write_invoice_pdf(out_dir / filename, rng, supplier, inv, po_number, total, inv_date)
        subj, body = fill(INVOICE_TEMPLATES[i % len(INVOICE_TEMPLATES)], inv=inv, po=po_number,
                          supplier=supplier["name"])
        add(f"accounts@{supplier['domain']}", subj, body,
            {"category": "invoice_received", "action": "record_invoice" if case == "match" else "flag_for_human",
             "invoice_number": inv, "po_number": po_number, "amount": total, "note": note},
            attachments=[filename])

    # --- Spam (15) ---
    for sender, subj, body in SPAM_MESSAGES:
        add(sender, subj, body, {"category": "spam", "action": "mark_spam", "note": "spam / scam"})

    # Shuffle so categories are mixed like a real inbox, then number them.
    rng.shuffle(pairs)
    base_time = datetime(2026, 10, 1, 8, 0)
    items, labels = [], []
    for n, (item, label) in enumerate(pairs, start=1):
        item_id = f"item-{n:03d}"
        received = base_time + timedelta(minutes=rng.randint(0, 14 * 24 * 60))
        items.append({"id": item_id, "received_at": received.isoformat(timespec="minutes"), **item})
        labels.append({"id": item_id, **label})

    with open(out_dir / "items.jsonl", "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(out_dir / "labels.jsonl", "w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")

    return {"customers": len(customers), "orders": len(orders), "purchase_orders": len(pos),
            "items": len(items), "invoices": len(invoice_cases)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the fake inbox, database and invoices.")
    parser.add_argument("--out", type=Path, default=DATA_DIR, help="output folder (default: data/)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    stats = generate(args.out, args.seed)
    print("Generated:", ", ".join(f"{v} {k}" for k, v in stats.items()))
    print(f"Files written to {args.out}")


if __name__ == "__main__":
    main()
