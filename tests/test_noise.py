from src.retrieval.noise import classify_noise


def test_contents_list_is_toc():
    text = "\n".join(
        [
            "Contents",
            "1. Summary",
            "2. Background",
            "2.1. A global treaty to end plastic pollution",
            "3. The big picture",
            "4. Results from the workshop",
            "4.3.4. Lacking data on large waste sources",
            "5. Recommendations",
            "References",
        ]
    )
    assert classify_noise(text) == "toc"


def test_contents_followed_by_prose_is_kept():
    text = "\n".join(
        [
            "Contents",
            "1. Introduction",
            "2. Methods",
            "3. Results",
            "4. Discussion",
            "",
            "Summary",
            (
                "This report evaluates waste prevention measures in the Nordic "
                "countries and explains why regulatory instruments remain uncommon."
            ),
        ]
    )
    assert classify_noise(text) is None


def test_imprint_page_is_flagged():
    text = "\n".join(
        [
            "About this publication",
            "TemaNord 2024:513",
            "ISBN 978-92-893-7793-5 (PDF)",
            "ISBN 978-92-893-7794-2 (ONLINE)",
            "Published: 24/2/2024",
            "Disclaimer",
            (
                "This publication was funded by the Nordic Council of Ministers. "
                "However, the content does not necessarily reflect its views."
            ),
            "Creative Commons Attribution 4.0",
        ]
    )
    assert classify_noise(text) == "imprint"


def test_prose_that_ends_with_isbn_is_kept():
    text = (
        "Replacing copper and nickel coatings on ammunition with wax, plastic, "
        "or steel would reduce demand for those metals in the Nordic countries. "
        "The report treats this as one waste-prevention option among several.\n\n"
        "About this publication\n"
        "ISBN 978-92-893-8267-0 (PDF)\n"
        "Disclaimer\n"
        "This publication was funded by the Nordic Council of Ministers."
    )
    assert classify_noise(text) is None


def test_bibliography_is_references():
    text = "\n".join(
        [
            "Swedish Government. (2022a). Ekonomiska styrmedel. https://example.org/a",
            "Swedish Government. (2022b). Uppdrag att utveckla. https://example.org/b",
            "Soderholm, P. (2011). Taxing virgin natural resources. https://example.org/c",
            "World Bank. (2022). Squaring the Circle. https://example.org/d",
            "United Nations Statistics Division. (2006). Trade. https://example.org/e",
        ]
    )
    assert classify_noise(text) == "references"


def test_blank_chunk_is_blank():
    assert classify_noise("   \n  ") == "blank"
