import math
from collections import Counter

import numpy as np
import pytest
import torch

from tiktorch import tiktypes as types
from tiktorch.server import datasets


class TestDynamicDataset:
    @pytest.fixture
    def dataset(self):
        return datasets.DynamicDataset()

    @pytest.fixture
    def simple_dataset(self, dataset):
        labels = types.TikTensorBatch(
            [types.TikTensor(np.ones(shape=(3, 3)), id_=(0, 0)), types.TikTensor(np.ones(shape=(3, 3)), id_=(1, 0))]
        )
        data = types.TikTensorBatch(
            [types.TikTensor(np.ones(shape=(3, 3)), id_=(0, 0)), types.TikTensor(np.ones(shape=(3, 3)), id_=(1, 0))]
        )
        dataset.update(data, labels)
        return dataset

    def test_empty_dataset_has_lenght_of_0(self, dataset):
        assert 0 == len(dataset)

    def test_updating_dataset_increases_its_size(self, dataset):
        labels = types.TikTensorBatch([types.TikTensor(np.ones(shape=(3, 3)), id_=(0, 0))])
        data = types.TikTensorBatch([types.TikTensor(np.ones(shape=(3, 3)), id_=(0, 0))])

        dataset.update(data, labels)
        assert 1 == len(dataset)

    def test_removing_entries_from_dataset(self, dataset):
        labels = types.TikTensorBatch(
            [types.TikTensor(np.ones(shape=(3, 3)), id_=(0, 0)), types.TikTensor(np.ones(shape=(3, 3)), id_=(1, 0))]
        )
        data = types.TikTensorBatch(
            [types.TikTensor(np.ones(shape=(3, 3)), id_=(0, 0)), types.TikTensor(np.ones(shape=(3, 3)), id_=(1, 0))]
        )
        dataset.update(data, labels)

        assert 2 == len(dataset)

        dataset.remove((0, 0))

        assert 1 == len(dataset)

    def test_updating_removed_entries_recovers_them(self, dataset):
        labels = types.TikTensorBatch(
            [types.TikTensor(np.ones(shape=(3, 3)), id_=(0, 0)), types.TikTensor(np.ones(shape=(3, 3)), id_=(1, 0))]
        )
        data = types.TikTensorBatch(
            [types.TikTensor(np.ones(shape=(3, 3)), id_=(0, 0)), types.TikTensor(np.ones(shape=(3, 3)), id_=(1, 0))]
        )
        dataset.update(data, labels)

        assert 2 == len(dataset)

        dataset.remove((0, 0))
        dataset.update(data, labels)

        assert 2 == len(dataset)

    def test_access_by_index(self, dataset):
        first_label = torch.Tensor(np.arange(9).reshape(3, 3))
        first_data = torch.Tensor(np.arange(1, 10).reshape(3, 3))

        labels = types.TikTensorBatch(
            [types.TikTensor(first_label, id_=(0, 0)), types.TikTensor(np.ones(shape=(3, 3)), id_=(1, 0))]
        )
        data = types.TikTensorBatch(
            [types.TikTensor(first_data, id_=(0, 0)), types.TikTensor(np.ones(shape=(3, 3)), id_=(1, 0))]
        )
        dataset.update(data, labels)
        ret_data, ret_label = dataset[0]

        assert torch.equal(first_label, ret_label)
        assert torch.equal(first_data, ret_data)

    def test_access_by_index_to_deleted_element_raises_entry_deleted(self, dataset):
        labels = types.TikTensorBatch([types.TikTensor(np.ones(shape=(3, 3)), id_=(0, 0))])
        data = types.TikTensorBatch([types.TikTensor(np.ones(shape=(3, 3)), id_=(0, 0))])
        dataset.update(data, labels)

        dataset.remove((0, 0))

        with pytest.raises(datasets.EntryRemoved):
            assert dataset[0]

    def test_initial_get_weights(self, simple_dataset):
        expected = torch.DoubleTensor([1.0, 1.0])
        assert torch.equal(expected, simple_dataset.get_weights())

    def test_access_by_index_changes_weight(self, simple_dataset):
        _ = simple_dataset[0]

        weights = simple_dataset.get_weights().tolist()
        assert [0.9, 1.0] == weights

        _ = simple_dataset[0]

        weights = simple_dataset.get_weights().tolist()
        assert [0.81, 1.0] == weights


class TestDynamicWeightedRandomSampler:
    class DatasetStub:
        def __init__(self, weights):
            self.weights = weights

        def get_weights(self):
            return self.weights

        def __len__(self):
            return len(self.weights)

    def test_should_return(self):
        ds = self.DatasetStub(torch.Tensor([0.0, 0.0, 1.0]))

        sampler = datasets.DynamicWeightedRandomSampler(ds)
        sample_idx = next(iter(sampler))
        assert sample_idx == 2

        sample_idx = next(iter(sampler))
        assert sample_idx == 2

    def test_distribution(self):
        ds = self.DatasetStub(torch.Tensor([0.1, 0.2, 0.7]))
        num_samples = 10_000

        sampler = datasets.DynamicWeightedRandomSampler(ds)
        sampler_iter = iter(sampler)

        samples = Counter(next(sampler_iter) for _ in range(num_samples))
        norm_2 = samples[2] / num_samples
        norm_1 = samples[1] / num_samples
        norm_0 = samples[0] / num_samples

        assert math.isclose(0.7, norm_2, abs_tol=0.01)
        assert math.isclose(0.2, norm_1, abs_tol=0.01)
        assert math.isclose(0.1, norm_0, abs_tol=0.01)
