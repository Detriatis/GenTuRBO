"""Focused sokoban regression tests for the reproducibility release."""
from __future__ import annotations

from collections import deque
import numpy as np
import pytest
from sokoban.boards import structural_errors
from sokoban.solver import SolverConfig, replay_solution, solve
import json
from dataclasses import replace
from pathlib import Path
import torch
import torch.nn.functional as F
from processes.sokoban_masked import sample_one_cell, weighted_masked_loss
from runners.common import restore_process_from_checkpoint
from runners.sample import SampleConfig, sample
from sokoban.train_transformer import TrainingState, TransformerTrainConfig, run

def puzzle(*, boxes=((2, 2),), goals=((2, 4),), player=(2, 1), room=(1, 5, 1, 6)):
    board = np.zeros((10, 10), dtype=np.uint8)
    r0, r1, c0, c1 = room
    board[r0:r1, c0:c1] = 1
    for cell in ((7, 2), (7, 4), (7, 6))[:4 - len(boxes)]:
        board[cell] = 5
    for cell in goals:
        board[cell] = 4
    for cell in boxes:
        board[cell] = 5 if board[cell] == 4 else 3
    board[player] = 6 if board[player] == 4 else 2
    assert not structural_errors(board)
    return board.flatten()

def reference_minimum_pushes(tokens):
    """Independent 0-1 BFS on literal worker steps, without any deadlock pruning."""
    board = tokens.reshape(10, 10)
    walls = {tuple(cell) for cell in np.argwhere(board == 0)}
    boxes = frozenset((tuple(cell) for cell in np.argwhere(np.isin(board, (3, 5)))))
    goals = frozenset((tuple(cell) for cell in np.argwhere(np.isin(board, (4, 5, 6)))))
    player = tuple(np.argwhere(np.isin(board, (2, 6)))[0])
    initial = (player, boxes)
    costs = {initial: 0}
    queue = deque([(initial, 0)])
    while queue:
        (player, boxes), cost = queue.popleft()
        if cost != costs[player, boxes]:
            continue
        if boxes == goals:
            return cost
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            target = (player[0] + dr, player[1] + dc)
            if target in walls:
                continue
            pushing = target in boxes
            new_boxes = boxes
            if pushing:
                destination = (target[0] + dr, target[1] + dc)
                if destination in walls or destination in boxes:
                    continue
                new_boxes = boxes - {target} | {destination}
            state = (target, new_boxes)
            next_cost = cost + int(pushing)
            if next_cost < costs.get(state, float('inf')):
                costs[state] = next_cost
                if pushing:
                    queue.append((state, next_cost))
                else:
                    queue.appendleft((state, next_cost))
    return None

def test_known_push_counts_and_independent_solution_replay():
    for goal, expected in [((2, 2), 0), ((2, 3), 1), ((2, 4), 2)]:
        board = puzzle(goals=(goal,))
        result = solve(board)
        assert result.status == 'solved'
        assert result.minimum_pushes == expected
        assert replay_solution(board, result.solution) == expected
        assert result.to_dict()['solution_verified']
        if expected < 2:
            assert result.difficulty == expected
    assert solve(puzzle()).solution == solve(puzzle()).solution

def test_matches_unpruned_worker_step_oracle_on_small_puzzles():
    rng = np.random.default_rng(313)
    cells = [(r, c) for r in range(1, 4) for c in range(1, 5)]
    outcomes = set()
    for count in (1, 2):
        for _ in range(50):
            selected = rng.choice(len(cells), size=2 * count + 1, replace=False)
            board = puzzle(boxes=tuple((cells[i] for i in selected[:count])), goals=tuple((cells[i] for i in selected[count:2 * count])), player=cells[selected[-1]], room=(1, 4, 1, 5))
            expected = reference_minimum_pushes(board)
            result = solve(board, SolverConfig(max_seconds=None))
            assert result.minimum_pushes == expected
            assert result.status == ('solved' if expected is not None else 'proven_unsolvable')
            outcomes.add(result.status)
            if expected is not None:
                assert replay_solution(board, result.solution) == expected
    assert outcomes == {'solved', 'proven_unsolvable'}

def test_dead_cell_and_exhausted_dynamic_deadlock_are_proofs():
    dead = solve(puzzle(boxes=((1, 2),)))
    assert (dead.status, dead.reason, dead.expanded_states) == ('proven_unsolvable', 'static_dead_cell', 0)
    locked = puzzle(boxes=((3, 3), (3, 4), (4, 3), (4, 4)), goals=((2, 2), (2, 5), (5, 2), (5, 5)), room=(1, 7, 1, 7))
    exhausted = solve(locked)
    assert (exhausted.status, exhausted.reason, exhausted.expanded_states) == ('proven_unsolvable', 'exhausted_search', 1)
    assert dead.difficulty is None and exhausted.difficulty is None

def test_limits_are_unknown_and_have_no_difficulty(monkeypatch):
    board = puzzle()
    for config, reason in [(SolverConfig(max_states=1), 'state_limit'), (SolverConfig(max_discovered_states=1), 'discovered_state_limit')]:
        result = solve(board, config)
        assert (result.status, result.reason) == ('unknown', reason)
        assert result.difficulty is None and result.minimum_pushes is None and (result.solution is None)
    times = iter([0.0, 2.0, 2.0])
    monkeypatch.setattr('sokoban.solver.time.monotonic', lambda: next(times))
    timed = solve(board, SolverConfig(max_seconds=1.0))
    assert (timed.status, timed.reason) == ('unknown', 'time_limit')

@pytest.mark.parametrize('change', ['no_boxes', 'two_players', 'mask', 'open_border'])
def test_invalid_boards_never_vacuously_solve(change):
    board = puzzle()
    if change == 'no_boxes':
        board[board == 3] = 1
        board[board == 5] = 4
    elif change == 'two_players':
        board[11] = 2
    elif change == 'mask':
        board[11] = 7
    else:
        board[0] = 1
    result = solve(board)
    assert result.status == 'structurally_invalid' and result.difficulty is None
    assert result.structural_errors and (not result.to_dict()['solution_verified'])

@pytest.mark.parametrize('moves', ['', 'rr', 'LL', 'l', 'Q'])
def test_replay_rejects_incomplete_or_illegal_certificates(moves):
    with pytest.raises(ValueError):
        replay_solution(puzzle(), moves)
REPO = Path(__file__).parents[1]
BASE = REPO / 'configs/train/sokoban_mask_transformer.json'

def config(tmp_path, name='test'):
    data = tmp_path / 'data'
    data.mkdir(exist_ok=True)
    rng = np.random.default_rng(5)
    np.save(data / 'train.npy', rng.integers(0, 7, (11, 100), dtype=np.uint8))
    np.save(data / 'valid.npy', rng.integers(0, 7, (3, 100), dtype=np.uint8))
    values = json.loads(BASE.read_text())
    values['model_cfg'].update(model_dim=16, num_layers=2, num_heads=2, ff_dim=32)
    values['dataset_cfg'].update(root=str(data), expected_count=11, batch_size=4)
    values.update(device='cpu', precision='float32', steps=6, warmup_steps=2, micro_batch_size=2, run_root=str(tmp_path / 'runs'), run_name=name, validation_every=0, evaluation_every=0, validation_size=3, evaluation_samples=2, evaluation_batch_size=2, checkpoint_every=2, solver_workers=1, log_every=1)
    return TransformerTrainConfig(**values)

def test_loss_is_weighted_per_board_not_pooled_per_token():
    torch.manual_seed(7)
    logits = torch.randn(3, 4, 7, requires_grad=True)
    clean = torch.tensor([[0, 1, 2, 3]] * 3)
    masked = torch.tensor([[True, False, False, False], [True, True, True, True], [False] * 4])
    steps = torch.tensor([1, 100, 10])
    expected = (10 * F.cross_entropy(logits[0, :1], clean[0, :1]) + F.cross_entropy(logits[1], clean[1])) / 3
    actual, _ = weighted_masked_loss(logits, clean, masked, steps)
    assert actual.item() == pytest.approx(expected.item())
    actual.backward()
    assert logits.grad[0, 1:].count_nonzero() == 0
    assert logits.grad[2].count_nonzero() == 0

def test_single_cell_sampling_and_time_mapping():

    class Recorder(torch.nn.Module):

        def forward(self, board, *, timesteps):
            assert torch.allclose(timesteps, 1 - (board == 7).sum(1).float() / 100)
            return torch.zeros((*board.shape, 7))
    model = Recorder().train()
    initial = torch.full((3, 100), 7, dtype=torch.long)
    initial[1, :97] = 2
    initial[2] = 4
    previous = initial.clone()
    calls = 0

    def check(board, time):
        nonlocal previous, calls
        visible = previous != 7
        assert torch.equal(board[visible], previous[visible])
        assert torch.equal((board != previous).sum(1), (previous == 7).any(1).long())
        previous = board
        calls += 1
    result = sample_one_cell(model, initial, generator=torch.Generator().manual_seed(99), on_step=check)
    assert calls == 100 and result.max() < 7 and model.training
    assert torch.equal(initial[1, :97], result[1, :97])
    assert torch.equal(result, sample_one_cell(model, initial, generator=torch.Generator().manual_seed(99)))

def test_full_resume_reproduces_uninterrupted_training(tmp_path):
    cfg = config(tmp_path, 'continuous')
    run(cfg)
    interrupted = replace(cfg, run_name='interrupted')
    result = run(interrupted, stop_after=3)
    assert result['status'] == 'interrupted'
    resumed = run(interrupted, resume=result['resume_checkpoint'])
    assert resumed['status'] == 'completed' and resumed['examples'] == 24
    expected = torch.load(Path(cfg.run_root) / cfg.run_name / 'checkpoints/training_state.pt', weights_only=True)
    actual = torch.load(resumed['resume_checkpoint'], weights_only=True)
    for name, parameter in expected['model_state'].items():
        assert torch.equal(parameter, actual['model_state'][name]), name
    assert expected['training_state']['cursor'] == actual['training_state']['cursor']
    assert torch.equal(expected['training_state']['rng']['torch'], actual['training_state']['rng']['torch'])
    restored = restore_process_from_checkpoint(Path(cfg.run_root) / cfg.run_name / 'checkpoints/model.pt', device_override='cpu')
    assert restored.process.sample(2, steps=100).shape == (2, 100)
    sampled = sample(SampleConfig(model=cfg.model, model_cfg=cfg.model_cfg, process=cfg.process, process_cfg=cfg.process_cfg, checkpoint_path=str(restored.checkpoint_path), num_samples=2, batch_size=1, device='cpu'))
    assert sampled.shape == (2, 100) and sampled.max() < 7
    with pytest.raises(ValueError, match='configuration'):
        run(replace(interrupted, steps=8), resume=resumed['resume_checkpoint'])
    with pytest.raises(FileExistsError):
        run(cfg)

def test_microbatch_accumulation_is_mean_of_board_losses(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    first = TrainingState(replace(cfg, micro_batch_size=4))
    second = TrainingState(replace(cfg, micro_batch_size=2))

    def deterministic(process):

        def step(batch):
            clean = batch['x']
            prediction = process.model.token_emb(clean).float().mean((1, 2))
            loss = prediction.square().mean()
            return (loss, {})
        return step
    monkeypatch.setattr(first.process, 'training_step', deterministic(first.process))
    monkeypatch.setattr(second.process, 'training_step', deterministic(second.process))
    a, b = (first.update(), second.update())
    assert a['loss'] == pytest.approx(b['loss'])
    for p, q in zip(first.model.parameters(), second.model.parameters(), strict=True):
        assert torch.allclose(p, q, atol=1e-07)
