import os

from src import embeddings

TEST_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_files", "shakespeare.txt"
)


def test_embeddings_init():

    e = embeddings.Embeddings("This is a test")
    assert e.chunk_size == 8191
    assert e.chunk_overlap == 0

    e = embeddings.Embeddings("This is a test", chunk_size=1000)
    assert e.chunk_size == 1000
    assert e.chunk_overlap == 0

    e = embeddings.Embeddings(
        "This is a test", chunk_size=1000, chunk_overlap=100
    )
    assert e.chunk_size == 1000
    assert e.chunk_overlap == 100

    try:
        e = embeddings.Embeddings("This is a test", chunk_size=10000)
    except ValueError as e:
        assert str(e).startswith("Chunk size must be less than")


def test_chunkenize_text_short():

    e = embeddings.Embeddings("This is a test")
    chunks = e._chunkenize_text()
    assert type(chunks) == list
    assert len(chunks) == 1
    assert chunks[0] == "This is a test"


def test_chunkenize_text_long():

    text = """
Tiger Woods, widely regarded as one of the greatest golfers of all time,
has left an indelible mark on the sport. Born Eldrick Tont Woods on
December 30, 1975, he became a prodigy at a young age, making waves in
the golfing world with his incredible talent and determination. Woods
turned professional in 1996 and quickly dominated the sport, earning
numerous accolades, including 15 major championships and a
record-tying 82 PGA Tour wins. His impact extended beyond the course,
as he became a global icon, inspiring a new generation of golfers and
breaking down barriers in a sport long associated with exclusivity.
Despite facing significant challenges, including injuries and
personal setbacks, Woods made an astonishing comeback by winning the
2019 Masters, a moment that solidified his legacy. His influence on
golf and sports, in general, remains unparalleled, making him a true
legend.
    """

    text = text.strip()

    e = embeddings.Embeddings(text, chunk_size=40)
    chunks = e._chunkenize_text()

    assert type(chunks) == list
    assert chunks[0].startswith("Tiger Woods, widely regarded as one of the")
    assert chunks[-1].endswith("remains unparalleled, making him a true\nlegend.")

    # Check overlap
    e = embeddings.Embeddings(text, chunk_size=40, chunk_overlap=10)
    chunks = e._chunkenize_text()

    c1 = chunks[0]
    c2 = chunks[1]

    assert c1.endswith('rick Tont Woods on\nDecember 30,')
    assert c2.startswith('rick Tont Woods on\nDecember 30,')


def test_chunkenize_text_very_long():

    with open(TEST_FILE, "r") as f:
        text = f.read()

    e = embeddings.Embeddings(text, chunk_size=8000)
    chunks = e._chunkenize_text()

    assert type(chunks) == list
    assert len(chunks) == 3


def test_get_embeddings():

    text = """
Tiger Woods, widely regarded as one of the greatest golfers of all time,
has left an indelible mark on the sport. Born Eldrick Tont Woods on
December 30, 1975, he became a prodigy at a young age, making waves in
the golfing world with his incredible talent and determination. Woods
turned professional in 1996 and quickly dominated the sport, earning
numerous accolades, including 15 major championships and a
record-tying 82 PGA Tour wins. His impact extended beyond the course,
as he became a global icon, inspiring a new generation of golfers and
breaking down barriers in a sport long associated with exclusivity.
Despite facing significant challenges, including injuries and
personal setbacks, Woods made an astonishing comeback by winning the
2019 Masters, a moment that solidified his legacy. His influence on
golf and sports, in general, remains unparalleled, making him a true
legend.
    """

    text = text.strip()
    e = embeddings.Embeddings(text)
    vectors = e.get_embeddings()
    assert len(vectors) == 1
    assert len(vectors[0]) == 2
    assert type(vectors[0][0]) == str
    assert type(vectors[0][1]) == list
    assert len(vectors[0][1]) == 1536

def test_get_embeddings_long():

    with open(TEST_FILE, "r") as f:
        text = f.read()

    e = embeddings.Embeddings(text, chunk_size=8000)
    vectors = e.get_embeddings()
    assert len(vectors) == 3
    assert len(vectors[0]) == 2
    assert type(vectors[0][0]) == str
    assert type(vectors[0][1]) == list
    assert len(vectors[0][1]) == 1536


def test_get_embeddings_long_diff_size():

    with open(TEST_FILE, "r") as f:
        text = f.read()

    e = embeddings.Embeddings(text, chunk_size=500)
    vectors = e.get_embeddings()

    assert len(vectors) == 44
    assert len(vectors[0]) == 2
    assert type(vectors[0][0]) == str
    assert type(vectors[0][1]) == list
    assert len(vectors[0][1]) == 1536