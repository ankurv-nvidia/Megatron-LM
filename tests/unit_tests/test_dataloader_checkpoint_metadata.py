# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Loading external dataloader shards from distributed-checkpoint metadata."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.training import checkpointing
from megatron.training.state import TrainState

pytestmark = pytest.mark.usefixtures("run_config")


def _metadata(*loader_keys):
    # Model metadata must not be converted into ShardedTensors just to discover the loader.
    return SimpleNamespace(
        state_dict_metadata={
            'model.weight': object(),
            'rng_state/shard_0.0.0_1.1.1': object(),
            **{key: object() for key in loader_keys},
        }
    )


def test_old_checkpoint_without_dataloader_state_needs_no_data_groups():
    with (
        mock.patch.object(checkpointing, '_get_filesystem_reader') as reader,
        mock.patch.object(checkpointing, 'get_pg_rank') as get_rank,
        mock.patch.object(checkpointing, 'get_pg_size') as get_size,
    ):
        reader.return_value.read_metadata.return_value = _metadata()
        assert checkpointing._get_dataloader_state_load_request('old-checkpoint') == {}

    reader.assert_called_once_with('old-checkpoint')
    get_rank.assert_not_called()
    get_size.assert_not_called()


@pytest.mark.parametrize('data_rank,data_size,view_rank', [(0, 1, 0), (3, 4, 2), (7, 8, 5)])
def test_shared_cursor_loads_one_shard_in_each_checkpoint_view(data_rank, data_size, view_rank):
    """A shared cursor follows resharded DP layouts and role-local checkpoint groups."""
    data_group = SimpleNamespace(rank=data_rank, size=data_size)
    checkpoint_group = object()
    with (
        mock.patch.object(checkpointing, '_get_filesystem_reader') as reader,
        mock.patch.object(checkpointing, 'get_pg_rank', side_effect=lambda group: group.rank),
        mock.patch.object(checkpointing, 'get_pg_size', side_effect=lambda group: group.size),
        mock.patch.object(
            checkpointing.torch.distributed, 'get_rank', return_value=view_rank
        ) as get_rank,
    ):
        reader.return_value.read_metadata.return_value = _metadata(
            'dataloader_state_dict/shard_0_1'
        )
        request = checkpointing._get_dataloader_state_load_request(
            'checkpoint', dp_group=data_group, checkpoint_group=checkpoint_group
        )['dataloader_state_dict']

    assert request.key == 'dataloader_state_dict'
    assert request.global_shape == (1,)
    assert request.global_offset == (0,)
    assert request.replica_id == view_rank
    assert request.data is None
    get_rank.assert_called_once_with(group=checkpoint_group)


@pytest.mark.parametrize('data_rank,tp_rank,pp_rank,cp_rank', [(0, 0, 0, 0), (3, 1, 2, 1)])
def test_rank_dependent_cursor_uses_full_data_axis_and_marks_model_replicas(
    data_rank, tp_rank, pp_rank, cp_rank
):
    """Distinct data ranks restore distinct state; TP/PP/CP peers are replicas."""
    groups = {
        'tp': SimpleNamespace(rank=tp_rank, size=2),
        'pp': SimpleNamespace(rank=pp_rank, size=3),
        'cp': SimpleNamespace(rank=cp_rank, size=2),
        # This represents the full dp x gtp_remat axis, rather than replicate DP.
        'data': SimpleNamespace(rank=data_rank, size=4),
    }
    with (
        mock.patch.object(checkpointing, '_get_filesystem_reader') as reader,
        mock.patch.object(checkpointing, 'get_pg_rank', side_effect=lambda group: group.rank),
        mock.patch.object(checkpointing, 'get_pg_size', side_effect=lambda group: group.size),
        mock.patch.object(checkpointing.mpu, 'get_context_parallel_group', return_value=object()),
        mock.patch.object(checkpointing.mpu, 'get_context_parallel_rank', return_value=cp_rank),
        mock.patch.object(checkpointing.mpu, 'get_data_parallel_rank') as global_data_rank,
        mock.patch.object(checkpointing.mpu, 'get_data_parallel_world_size') as global_data_size,
    ):
        reader.return_value.read_metadata.return_value = _metadata(
            *(f'dataloader_state_dict/shard_{rank}_4' for rank in range(4))
        )
        request = checkpointing._get_dataloader_state_load_request(
            'checkpoint',
            tp_group=groups['tp'],
            pp_group=groups['pp'],
            dp_group=groups['data'],
            cp_group=groups['cp'],
        )['dataloader_state_dict']

    assert request.global_shape == (4,)
    assert request.global_offset == (data_rank,)
    assert request.replica_id == (tp_rank, pp_rank, cp_rank)
    global_data_rank.assert_not_called()
    global_data_size.assert_not_called()


@pytest.mark.parametrize('current_size', [2, 8])
def test_rank_dependent_cursor_rejects_data_parallel_resize(current_size):
    with (
        mock.patch.object(checkpointing, '_get_filesystem_reader') as reader,
        mock.patch.object(checkpointing, 'get_pg_rank', side_effect=lambda group: group.rank),
        mock.patch.object(checkpointing, 'get_pg_size', side_effect=lambda group: group.size),
    ):
        reader.return_value.read_metadata.return_value = _metadata(
            'dataloader_state_dict/shard_0_4'
        )
        with pytest.raises(RuntimeError, match=f'checkpoint 4, current {current_size}'):
            checkpointing._get_dataloader_state_load_request(
                'checkpoint', dp_group=SimpleNamespace(rank=0, size=current_size)
            )


@pytest.mark.parametrize('restore_training_state', [True, False])
def test_only_primary_checkpoint_load_invalidates_pending_dataloader_state(
    monkeypatch, run_config, restore_training_state
):
    """An auxiliary teacher load before provider construction preserves the training cursor."""
    args = SimpleNamespace(dataloader_type='external')
    train_state = TrainState(iteration=7)
    run_config.checkpoint.load = 'training-checkpoint'
    run_config.checkpoint.ckpt_format = 'torch_dist'
    cursor = {'global_sequence_id': 16}
    monkeypatch.setattr(checkpointing, '_LOADED_DATALOADER_STATE', (args, train_state, 7, cursor))
    monkeypatch.setattr(checkpointing, 'get_train_state', lambda: train_state)
    monkeypatch.setattr(checkpointing, 'get_args', lambda: args)
    monkeypatch.setattr(
        checkpointing, '_load_base_checkpoint', mock.Mock(return_value=(None, '', False, None))
    )

    assert checkpointing.load_checkpoint(
        [], None, None, restore_training_state=restore_training_state
    ) == (0, 0)
    assert checkpointing.get_loaded_dataloader_state(args) == (
        None if restore_training_state else cursor
    )


def test_pending_dataloader_state_is_scoped_to_args_and_iteration(monkeypatch):
    args = SimpleNamespace()
    train_state = TrainState(iteration=7)
    monkeypatch.setattr(checkpointing, 'get_train_state', lambda: train_state)
    monkeypatch.setattr(checkpointing, '_LOADED_DATALOADER_STATE', (args, train_state, 7, {}))
    assert checkpointing.get_loaded_dataloader_state(args) == {}
    assert checkpointing.get_loaded_dataloader_state(SimpleNamespace()) is None
    train_state.iteration = 8
    assert checkpointing.get_loaded_dataloader_state(args) is None
    replacement = TrainState(iteration=7)
    monkeypatch.setattr(checkpointing, 'get_train_state', lambda: replacement)
    assert checkpointing.get_loaded_dataloader_state(args) is None


@pytest.mark.parametrize('dataloader_type', [None, 'single', 'external'])
@pytest.mark.parametrize('restore_training_state', [False, True])
def test_only_external_loader_resumes_request_embedded_state(
    monkeypatch, run_config, dataloader_type, restore_training_state
):
    """Weights-only inference loads do not impose the saved dataloader's DP layout."""
    args = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        world_size=1,
        data_parallel_size=1,
    )
    if dataloader_type is not None:
        args.dataloader_type = dataloader_type
    run_config.checkpoint.load = 'checkpoint'
    run_config.checkpoint.ckpt_format = 'torch_dist'
    run_config.checkpoint.load_rng = False
    run_config.checkpoint.load_optim = False
    probe = {'args': args, 'iteration': 7, 'checkpoint_version': 3.1}
    monkeypatch.setattr(checkpointing, 'get_args', lambda: args)
    monkeypatch.setattr(
        checkpointing,
        '_load_base_checkpoint',
        mock.Mock(
            side_effect=[
                (probe, 'checkpoint', False, checkpointing.CheckpointType.GLOBAL),
                (None, '', False, None),
            ]
        ),
    )
    monkeypatch.setattr(
        checkpointing, '_maybe_setup_gpt_to_hybrid_load', lambda *a, **kw: (None, False)
    )
    monkeypatch.setattr(checkpointing, 'generate_state_dict', lambda *a, **kw: {'model': {}})
    monkeypatch.setattr(checkpointing.dist_checkpointing, 'load_content_metadata', lambda **kw: {})
    request = mock.Mock(return_value={})
    monkeypatch.setattr(checkpointing, '_get_dataloader_state_load_request', request)

    assert checkpointing.load_checkpoint(
        [object()], None, None, dp_cp_group=object(), restore_training_state=restore_training_state
    ) == (0, 0)
    assert request.call_count == int(restore_training_state and dataloader_type == 'external')


def test_raw_dataloader_snapshot_preserves_state_without_backend_wrapping():
    """Capture an owned payload that checkpoint-format adapters can reuse."""
    group = SimpleNamespace(rank=0, size=1)
    data_group = SimpleNamespace(rank=3, size=4)
    raw_state = {'cursor': 16, 'workers': [{'offsets': [2, 7], 'buffer': b'tokens'}]}
    iterator = SimpleNamespace(iterable=SimpleNamespace(save_state=lambda: raw_state))
    with (
        mock.patch.object(checkpointing, 'get_pg_rank', side_effect=lambda group: group.rank),
        mock.patch.object(checkpointing, 'get_pg_size', side_effect=lambda group: group.size),
        mock.patch.object(checkpointing, 'ShardedObject') as backend_wrapper,
        mock.patch.object(checkpointing.torch, 'save') as save,
        mock.patch.object(checkpointing.torch.distributed, 'barrier') as barrier,
    ):
        payload = checkpointing._get_dataloader_state(
            iterator, 'enabled', tp_group=group, pp_group=group, dp_group=data_group
        )

    raw_state['cursor'] = 17
    raw_state['workers'][0]['offsets'][0] = 100
    assert payload == {
        'dataloader_state_dict': {
            'cursor': 16,
            'workers': [{'offsets': [2, 7], 'buffer': b'tokens'}],
        },
        'is_rank_independent': False,
        'data_parallel_size': 4,
    }
    backend_wrapper.assert_not_called()
    save.assert_not_called()
    barrier.assert_not_called()


@pytest.mark.parametrize('ckpt_format', ['torch_dist', 'torch', 'future-format'])
@pytest.mark.parametrize('payload', [None, {}, {'dataloader_state_dict': {'cursor': 16}}])
def test_generate_state_dict_accepts_dataloader_state_independently_of_format(ckpt_format, payload):
    args = SimpleNamespace(ckpt_format=ckpt_format, no_save_optim=True, no_save_rng=True)

    state = checkpointing.generate_state_dict(
        args, [], None, None, None, iteration=7, ckpt_format=ckpt_format, dataloader_state=payload
    )

    assert state['iteration'] == 7
    if payload is None:
        assert 'dataloader_state_dict' not in state
    else:
        assert state['dataloader_state_dict'] is payload


def test_model_extra_state_cleanup_does_not_transform_opaque_dataloader_state(run_config):
    args = SimpleNamespace(
        ckpt_format='torch_dist',
        no_save_optim=True,
        no_save_rng=True,
        ckpt_drop_redundant_extra_state=True,
    )
    run_config.checkpoint.ckpt_drop_redundant_extra_state = True
    opaque = checkpointing.ShardedObject('loader._extra_state', None, (1,), (0,))
    payload = {'opaque': opaque}

    state = checkpointing.generate_state_dict(args, [], None, None, None, dataloader_state=payload)

    assert state['dataloader_state_dict']['opaque'] is opaque


@pytest.mark.parametrize('rank_independent', [False, True])
@pytest.mark.parametrize('data_rank', [0, 3])
def test_native_dataloader_state_round_trip_preserves_opaque_values(rank_independent, data_rank):
    """Native DCP stores each opaque state as one byte entry without flattening its contents."""
    state_rank = 0 if rank_independent else data_rank
    cursor = 16 + state_rank
    payload = {
        'dataloader_state_dict': {
            'cursor': cursor,
            'workers': [{'rng': torch.tensor([1234, state_rank]), 'buffer': b'token-bytes'}],
            'pending': (None, [True, 1.25]),
        },
        'is_rank_independent': rank_independent,
        'data_parallel_size': 4,
    }
    group = SimpleNamespace(rank=data_rank, size=4)
    with mock.patch.object(checkpointing, 'get_pg_rank', side_effect=lambda group: group.rank):
        encoded = checkpointing._get_native_dataloader_state(payload, dp_group=group)

    key = 'shared' if rank_independent else f'rank_{data_rank:05d}'
    assert set(encoded) == {key}
    assert isinstance(encoded[key], bytes)
    payload['dataloader_state_dict']['cursor'] = 1000
    payload['dataloader_state_dict']['workers'][0]['rng'].zero_()
    decoded = checkpointing._decode_dataloader_state(encoded)

    assert decoded['is_rank_independent'] is rank_independent
    assert decoded['data_parallel_size'] == 4
    assert decoded['dataloader_state_dict']['cursor'] == cursor
    assert torch.equal(
        decoded['dataloader_state_dict']['workers'][0]['rng'], torch.tensor([1234, state_rank])
    )
    assert decoded['dataloader_state_dict']['workers'][0]['buffer'] == b'token-bytes'
    assert decoded['dataloader_state_dict']['pending'] == (None, [True, 1.25])


def test_native_dataloader_state_distinguishes_absent_and_empty_payload():
    assert checkpointing._get_native_dataloader_state(None) is None
    assert checkpointing._decode_dataloader_state(None) is None
    payload = {'dataloader_state_dict': {}, 'is_rank_independent': True, 'data_parallel_size': 1}

    encoded = checkpointing._get_native_dataloader_state(payload)

    assert set(encoded) == {'shared'}
    assert checkpointing._decode_dataloader_state(encoded) == payload
    # The torch_dist adapter already supplies the decoded payload.
    assert checkpointing._decode_dataloader_state(payload) is payload


@pytest.mark.parametrize('ckpt_format', ['torch_dcp', 'fsdp_dtensor'])
def test_old_native_checkpoint_without_loader_state_needs_no_data_groups(ckpt_format):
    with (
        mock.patch.object(checkpointing, '_get_filesystem_reader') as reader,
        mock.patch.object(checkpointing, 'get_pg_rank') as get_rank,
        mock.patch.object(checkpointing, 'get_pg_size') as get_size,
    ):
        reader.return_value.read_metadata.return_value = _metadata()
        assert (
            checkpointing._get_dataloader_state_load_request(
                'old-checkpoint', ckpt_format=ckpt_format
            )
            == {}
        )

    get_rank.assert_not_called()
    get_size.assert_not_called()


@pytest.mark.parametrize('ckpt_format', ['torch_dcp', 'fsdp_dtensor'])
def test_native_shared_cursor_loads_on_resharded_data_rank(ckpt_format):
    group = SimpleNamespace(rank=7, size=8)
    with (
        mock.patch.object(checkpointing, '_get_filesystem_reader') as reader,
        mock.patch.object(checkpointing, 'get_pg_rank', side_effect=lambda group: group.rank),
        mock.patch.object(checkpointing, 'get_pg_size', side_effect=lambda group: group.size),
    ):
        reader.return_value.read_metadata.return_value = _metadata('dataloader_state_dict.shared')
        request = checkpointing._get_dataloader_state_load_request(
            'checkpoint', dp_group=group, ckpt_format=ckpt_format
        )

    assert request == {'dataloader_state_dict': {'shared': None}}


@pytest.mark.parametrize('ckpt_format', ['torch_dcp', 'fsdp_dtensor'])
@pytest.mark.parametrize('data_rank', [0, 3])
def test_native_rank_dependent_cursor_requests_only_its_rank(ckpt_format, data_rank):
    group = SimpleNamespace(rank=data_rank, size=4)
    with (
        mock.patch.object(checkpointing, '_get_filesystem_reader') as reader,
        mock.patch.object(checkpointing, 'get_pg_rank', side_effect=lambda group: group.rank),
        mock.patch.object(checkpointing, 'get_pg_size', side_effect=lambda group: group.size),
    ):
        reader.return_value.read_metadata.return_value = _metadata(
            *(f'dataloader_state_dict.rank_{rank:05d}' for rank in range(4))
        )
        request = checkpointing._get_dataloader_state_load_request(
            'checkpoint', dp_group=group, ckpt_format=ckpt_format
        )

    assert request == {'dataloader_state_dict': {f'rank_{data_rank:05d}': None}}


@pytest.mark.parametrize('ckpt_format', ['torch_dcp', 'fsdp_dtensor'])
@pytest.mark.parametrize('current_size', [2, 8])
def test_native_rank_dependent_cursor_rejects_data_parallel_resize(ckpt_format, current_size):
    group = SimpleNamespace(rank=0, size=current_size)
    with (
        mock.patch.object(checkpointing, '_get_filesystem_reader') as reader,
        mock.patch.object(checkpointing, 'get_pg_rank', side_effect=lambda group: group.rank),
        mock.patch.object(checkpointing, 'get_pg_size', side_effect=lambda group: group.size),
    ):
        reader.return_value.read_metadata.return_value = _metadata(
            *(f'dataloader_state_dict.rank_{rank:05d}' for rank in range(4))
        )
        with pytest.raises(RuntimeError, match=f'checkpoint 4, current {current_size}'):
            checkpointing._get_dataloader_state_load_request(
                'checkpoint', dp_group=group, ckpt_format=ckpt_format
            )


@pytest.mark.parametrize('ckpt_format', ['torch_dcp', 'fsdp_dtensor'])
def test_native_checkpoint_with_missing_rank_cannot_fall_back_to_sidecar(ckpt_format):
    """An incomplete embedded layout is an error even under FSDP's partial model load."""
    group = SimpleNamespace(rank=3, size=4)
    with (
        mock.patch.object(checkpointing, '_get_filesystem_reader') as reader,
        mock.patch.object(checkpointing, 'get_pg_rank', side_effect=lambda group: group.rank),
        mock.patch.object(checkpointing, 'get_pg_size', side_effect=lambda group: group.size),
    ):
        reader.return_value.read_metadata.return_value = _metadata(
            *(f'dataloader_state_dict.rank_{rank:05d}' for rank in (0, 1, 2, 4))
        )
        with pytest.raises(RuntimeError, match='rank'):
            checkpointing._get_dataloader_state_load_request(
                'checkpoint', dp_group=group, ckpt_format=ckpt_format
            )
