from pathlib import Path

import pytest

from slam.data.kitti import KittiSequence
from slam.features.orb import extract_orb
from slam.features.vocab import Vocabulary, score


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary("datasets/vocab/ORBvoc.txt")


def test_bow_separates_places(vocab):
    seq = KittiSequence(Path("datasets/kitti"), "00")
    bow = {}
    for i in (0, 1, 1200):
        _, desc, _, _ = extract_orb(seq[i].image_left)
        words, node = vocab.transform(desc)
        assert (words >= 0).all() and node.max() < len(vocab.desc)
        bow[i] = vocab.bow(words)
        assert bow[i][1].sum() == pytest.approx(1.0)
    assert score(bow[0], bow[0]) == pytest.approx(1.0)
    assert score(bow[0], bow[1]) > 2 * score(bow[0], bow[1200])
