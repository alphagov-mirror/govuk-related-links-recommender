import random
import pytest
import numpy as np

from src.models.train_node2vec_model import N2VModel

# set seed for deterministic tests etc
random.seed(2)
np.random.seed(2)


@pytest.mark.skip(reason="not staying deterministic")
def test_train_node2_vec_model(all_network_fixture):
    # TODO if this needs to be deterministic use the same random walks for training the model (probably)
    n2v = N2VModel()

    n2v.create_graph(all_network_fixture)

    n2v.generate_walks(workers=1)

    n2v.fit_model(workers=1)

    # test we get the same most similar nodes
    print(n2v.model.wv.most_similar(
        "8ccf089c-a512-452f-b027-dee409feb1f3", topn=10))
    assert n2v.model.wv.most_similar(
        "8ccf089c-a512-452f-b027-dee409feb1f3", topn=10) == [
        ('2', 0.9990991353988647), ('501', 0.9988441467285156), ('502', 0.9988089799880981),
        ('500', 0.9987748265266418), ('505', 0.998684287071228), ('504', 0.9986145496368408),
        ('499', 0.9983056783676147), ('1', 0.9978947639465332), ('0', 0.9414731860160828), ('380', 0.6794115304946899)]

    # test an embedding vector
    print(n2v.model.wv['d6204ab9-99fd-4265-a081-5e4c6fdfe11d'])
    np.testing.assert_array_equal(
        n2v.model.wv['d6204ab9-99fd-4265-a081-5e4c6fdfe11d'],
        np.array(
            [-0.88379043, -0.30888686, -0.1813301, -0.067976356, -0.4609355, -0.35824582, -0.08936877, 0.08270162,
             -0.6352391, 0.18851767, -0.14360602, -0.6236442, -0.10684205, 0.16706303, -0.017764626, -0.19982526,
             -0.08983809, 0.23044425, -0.60631204, 0.21698123, -0.05427989, 0.24906242, 0.6751661, -0.6701944,
             -0.08946387, 1.2561842, -0.3632976, -0.33041343, -0.42487434, 1.3301302, -0.1845399, -1.1003605,
             0.23692428, -1.5703396, 0.69791293, -0.14096017, -0.3939386, -0.7056384, 0.20997848, -0.013814987,
             0.3107433, -0.059977658, -0.06617668, 0.20900504, -0.7393868, 0.8214224, -0.10128406, 0.3389172,
             -0.89528805, -0.41821027, 0.3999012, -0.06859911, -0.70715356, -0.45099786, -0.16714878, -0.3983585,
             0.5686123, -0.37777436, -0.6602222, 0.58181953, -1.0959847, -0.09768479, 0.38792232, 0.32195967],
            dtype='float32'))
