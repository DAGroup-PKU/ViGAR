"""Caption CFG training, worker-independent resume and original optimizer groups."""

import copy
import json
import random
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from recipes.GoalWAM.data.data_loader import CaptionDropoutDataset, StatefulWindowLoader, WindowSampler
from recipes.GoalWAM.data.dataset import LeRobot0824SFTDataset
from recipes.GoalWAM.tests.test_checkpoint_bundle import make_bundle
from recipes.GoalWAM.trainer import goalwam_trainer
from recipes.GoalWAM.trainer.arguments import GoalWAMDataArguments, VeOmniGoalWAMArguments
from veomni.arguments.parser import _instantiate_recursive


RECIPE = Path(__file__).resolve().parents[1]


class TinyTokenizer:
    def tokenize_text(self, text, **kwargs):
        return [100, *text.encode(), 101]


@pytest.mark.parametrize("text_conditioning", ["episode", "segment", "episode_segment"])
def test_training_dropout_retains_goal_state_and_eval_caption(monkeypatch, text_conditioning):
    from cosmos_framework.data.vfm.augmentors import text_tokenizer

    monkeypatch.setattr(text_tokenizer, "lazy_instantiate", lambda _: TinyTokenizer())
    raw = SimpleNamespace(img_size=[240, 320], resolution="384x320", normalizers={"robot": None})

    def build_raw(*args, **kwargs):
        assert kwargs["text_conditioning"] == text_conditioning
        return raw

    monkeypatch.setattr(goalwam_trainer, "LeRobot0824Dataset", build_raw)
    trainer = object.__new__(goalwam_trainer.GoalWAMTrainer)
    trainer.args = SimpleNamespace(
        data=GoalWAMDataArguments(
            train_path="manifest.yaml", caption_dropout_rate=1.0, text_conditioning=text_conditioning
        ),
        train=SimpleNamespace(seed=42),
    )
    trainer.native_config = OmegaConf.create({"model": {"config": {"vlm_config": {"tokenizer": {}}}}})
    train = trainer._dataset("train.yaml", "all", training=True)
    evaluation = trainer._dataset("test.yaml", "all")
    sample = dict(
        ai_caption="tidy the table\nCurrent subtask: pick up bottle"
        if text_conditioning == "episode_segment"
        else "pick up bottle",
        video=torch.ones(3, 13, 32, 32, dtype=torch.uint8),
        goal_frame=torch.full((3, 1, 32, 32), 7, dtype=torch.uint8),
        action=torch.arange(49 * 49, dtype=torch.float32).reshape(49, 49),
        action_valid_mask=torch.ones(49, 49, dtype=torch.bool),
        mode="policy",
        domain_id=torch.tensor(0),
        fps=torch.tensor(7.5),
        action_fps=torch.tensor(30.0),
    )
    dropped = train.prepare_sample(copy.deepcopy(sample), "robot")
    retained = evaluation.prepare_sample(copy.deepcopy(sample), "robot")
    assert dropped["ai_caption"] == ""
    assert retained["ai_caption"] == sample["ai_caption"]
    assert dropped["text_token_ids"].tolist() == TinyTokenizer().tokenize_text("")
    assert retained["text_token_ids"].tolist() == TinyTokenizer().tokenize_text(sample["ai_caption"])
    for key in ("action", "action_valid_mask"):
        torch.testing.assert_close(dropped[key], retained[key])
    for actual, expected in zip(dropped["video"], retained["video"]):
        torch.testing.assert_close(actual, expected)
    assert dropped["video"][0].eq(7).all()
    assert dropped["sequence_plan"].vision_item_roles == ["goal", "default"]
    assert dropped["sequence_plan"].condition_frame_indexes_action == [0]


class RandomCaptionDataset(Dataset):
    cfg_dropout_rate = 0.1

    def __len__(self):
        return 11

    def __getitem__(self, index):
        return index, random.random() < self.cfg_dropout_rate


def decisions(workers, cursor, count):
    raw = RandomCaptionDataset()
    sampler = WindowSampler(len(raw), 1, 4, seed=42, include_position=True)
    sampler.cursor = cursor
    loader = DataLoader(
        CaptionDropoutDataset(raw, seed=42),
        sampler=list(islice(iter(sampler), count)),
        batch_size=None,
        num_workers=workers,
        **({"multiprocessing_context": "spawn", "prefetch_factor": 4} if workers else {}),
    )
    return list(loader)


def test_dropout_reproduces_with_workers_prefetch_resume_and_repeated_windows():
    rng_before = random.getstate()
    reference = decisions(0, 0, 160)
    assert random.getstate() == rng_before
    assert decisions(2, 0, 160) == reference
    assert decisions(2, 13, 147) == reference[13:]
    assert 0 < sum(dropped for _, dropped in reference) < len(reference)
    # Repeated epochs must not permanently mark a physical window as dropped.
    assert any({flag for j, flag in reference if j == i} == {False, True} for i in range(11))


def test_loader_preserves_legacy_state_and_rejects_changed_dropout():
    raw = RandomCaptionDataset()
    loader = StatefulWindowLoader(raw, num_workers=0)
    state = loader.state_dict()
    state["cursor"] = 13
    loader.load_state_dict(state)
    assert next(iter(loader.sampler))[1] == 13
    assert state["caption_dropout"] == {"rate": 0.1, "seed_version": 1}
    raw.cfg_dropout_rate = 0.0
    legacy = StatefulWindowLoader(raw, num_workers=0)
    assert "caption_dropout" not in legacy.state_dict()
    legacy.load_state_dict(legacy.state_dict())
    with pytest.raises(ValueError, match="Caption dropout changed"):
        loader.load_state_dict(legacy.state_dict())
    with pytest.raises(ValueError, match="Caption dropout changed"):
        legacy.load_state_dict(state)


@pytest.mark.parametrize("evaluation_only", [False, True])
def test_standalone_eval_skips_training_resume_but_keeps_checkpoint_step(tmp_path, monkeypatch, evaluation_only):
    (tmp_path / "complete.json").write_text(json.dumps({"iteration": 2000}))
    restored = []

    def load(*args, resume):
        restored.append(resume)
        return 2000 if resume else 0

    monkeypatch.setattr(goalwam_trainer, "load_checkpoint", load)
    monkeypatch.setattr(goalwam_trainer, "seed_all", lambda _: None)
    monkeypatch.setattr(goalwam_trainer, "parameter_manifest", lambda _: {})
    monkeypatch.setattr(goalwam_trainer.dist, "get_rank", lambda: 1)
    trainer = object.__new__(goalwam_trainer.GoalWAMTrainer)
    trainer.args = SimpleNamespace(
        model=SimpleNamespace(native_checkpoint=str(tmp_path), model_path=None, resume_training=True),
        train=SimpleNamespace(evaluation_only=evaluation_only),
        evaluation_budget=SimpleNamespace(windows_per_rank=0),
    )
    trainer.output = tmp_path
    trainer.state = SimpleNamespace(global_step=0)
    trainer.core = trainer.optimizers = trainer.lr_schedulers = trainer.train_dataloader = None
    assert trainer._initialize_runtime() == 2000
    assert trainer.state.global_step == 2000
    assert restored == [not evaluation_only]


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_caption_dropout(rate):
    with pytest.raises(ValueError, match="caption_dropout_rate"):
        GoalWAMDataArguments(train_path="manifest.yaml", caption_dropout_rate=rate)
    with pytest.raises(ValueError, match="cfg_dropout_rate"):
        LeRobot0824SFTDataset(None, cfg_dropout_rate=rate)


@pytest.mark.parametrize("maintained", [False, True])
def test_original_optimizer_overrides_and_historical_defaults(tmp_path, monkeypatch, maintained):
    from cosmos_framework.utils.vfm.optimizer import _build_params_with_metadata

    values = yaml.safe_load((RECIPE / "configs/robotwin.yaml").read_text())
    if not maintained:
        values["data"].pop("caption_dropout_rate")
        values["train"]["optimizer"].pop("goal_lr_multiplier")
        values["train"]["optimizer"].pop("disable_weight_decay_for_1d_params")
    values["model"]["model_path"] = str(make_bundle(tmp_path / "bundle"))
    args = _instantiate_recursive(VeOmniGoalWAMArguments, values)
    native = OmegaConf.create(
        json.loads((RECIPE / "configs/migration/initialization_config.json").read_text())["cosmos"]
    )
    config = SimpleNamespace(runtime_config=lambda **kwargs: native, bundle_assets=None)
    monkeypatch.setattr(goalwam_trainer, "get_model_config", lambda _: config)
    monkeypatch.setattr(
        goalwam_trainer, "get_model_class", lambda _: lambda *args, **kwargs: SimpleNamespace(core=SimpleNamespace())
    )
    trainer = object.__new__(goalwam_trainer.GoalWAMTrainer)
    trainer.args = args
    trainer._build_model()
    opt = native.optimizer
    assert args.data.caption_dropout_rate == (0.1 if maintained else 0.0)
    assert opt.lr_multipliers.get("goal_vision_embed", 1.0) == (5.0 if maintained else 1.0)
    assert opt.disable_weight_decay_for_1d_params == maintained
    model = torch.nn.Module()
    model.net = torch.nn.Module()
    model.net.goal_vision_embed = torch.nn.Parameter(torch.ones(8))
    model.net.action2llm = torch.nn.Linear(8, 8)
    model.net.reasoner = torch.nn.Linear(8, 8)
    groups = _build_params_with_metadata(
        model, list(opt.keys_to_select), dict(opt.lr_multipliers), opt.lr, opt.disable_weight_decay_for_1d_params
    )
    metadata = {id(param): meta for param, meta in groups}
    goal = metadata[id(model.net.goal_vision_embed)]
    assert goal.lr == pytest.approx(5e-5 if maintained else 1e-5)
    assert goal.enable_weight_decay == (not maintained)
    assert metadata[id(model.net.action2llm.weight)].enable_weight_decay
    assert metadata[id(model.net.action2llm.bias)].enable_weight_decay == (not maintained)
    assert not model.net.reasoner.weight.requires_grad
