import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             roc_auc_score)
from torch.nn import functional as F

from data_processing5 import load_task_data, prepare_batch
from MyModel import GraphCNN


Target = "A. thaliana"
Checkpoint = ""
Checkpointgraphmode = "local_pair"
Graphmode = "multiscale"
Seed = 1377
Dynamicepochs = 50
Multidomainepochs = Dynamicepochs
Refinementepochs = 20
Initialcalibrations = 3
Auditwindow = 5
Selectionepochs = 30
Batch = 64
Multidomainlr = 0.001
Refinementlr = 0.0001
Tripletweight = 1.0
Auxiliaryweight = 0.5
Useprojection = False
Smoke = False


Root = Path(__file__).resolve().parent
Datafolder = Root.parent / "data7"
Datasetfiles = {
    "A. thaliana": "A.thaliana.xlsx",
    "C. elegans": "C.elegans.xlsx",
    "C. equisetifolia": "C.equisetifolia.xlsx",
    "D. melanogaster": "D.melanogaster.xlsx",
    "F. vesca": "F.vesca.xlsx",
    "H. sapiens": "H.sapiens.xlsx",
    "R. chinensis": "R.chinensis.xlsx",
    "S. cerevisiae": "S.cerevisiae.xlsx",
    "T. thermophile": "T.thermophile.xlsx",
    "Tolypocladium": "6mA_Tolypocladium4.xlsx",
    "Xoc. BLS256": "Xoc BLS256.xlsx",
}
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BalancedSampler:
    def __init__(self, task, batch, rng):
        self.rng = rng
        self.sizes = (batch // 2, batch - batch // 2)
        self.pools = [task.train[task.labels[task.train] == label].copy()
                      for label in (0, 1)]
        self.offsets = [len(pool) for pool in self.pools]

    def _take(self, label, count):
        selected = []
        while count:
            pool = self.pools[label]
            if self.offsets[label] == len(pool):
                self.rng.shuffle(pool)
                self.offsets[label] = 0
            size = min(count, len(pool) - self.offsets[label])
            start = self.offsets[label]
            selected.append(pool[start:start + size].copy())
            self.offsets[label] += size
            count -= size
        return np.concatenate(selected)

    def next(self):
        indices = np.concatenate([
            self._take(label, size) for label, size in enumerate(self.sizes)
        ])
        self.rng.shuffle(indices)
        return indices


def batches_per_epoch(task, batch):
    sizes = (batch // 2, batch - batch // 2)
    counts = [np.count_nonzero(task.labels[task.train] == label) for label in (0, 1)]
    return max(math.ceil(count / size) for count, size in zip(counts, sizes))


def task_schedule(auxiliary, steps, offset=0):
    if not auxiliary:
        raise ValueError("At least one auxiliary task is required")
    return [auxiliary[(offset + step) % len(auxiliary)]
            for step in range(steps)]


def cpu_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def scores(labels, probabilities):
    predicted = probabilities >= 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "acc": float(accuracy_score(labels, predicted)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "f1": float(f1_score(labels, predicted)),
        "sen": float(tp / (tp + fn)),
        "spe": float(tn / (tn + fp)),
    }


@torch.no_grad()
def predict(model, task, graph_mode, device):
    model.eval()
    probabilities = []
    for start in range(0, len(task.test), 512):
        indices = task.test[start:start + 512]
        sequences, graphs = prepare_batch(task.sequences[indices], graph_mode, device)
        probabilities.append(model.trainModel(sequences, graphs)[:, 1].cpu().numpy())
    return task.labels[task.test].copy(), np.concatenate(probabilities)


def evaluate(model, task, graph_mode, device):
    labels, probabilities = predict(model, task, graph_mode, device)
    return scores(labels, probabilities)


def triplet_loss(embedding, labels):
    positive = torch.empty(len(labels), dtype=torch.long, device=labels.device)
    negative = torch.empty_like(positive)
    for label in (0, 1):
        current = torch.where(labels == label)[0]
        opposite = torch.where(labels != label)[0]
        positive[current] = current.roll(1)
        negative[current] = opposite[
            torch.arange(len(current), device=labels.device) % len(opposite)]
    return F.triplet_margin_loss(
        embedding, embedding[positive], embedding[negative], margin=2.0)


def batch_loss(model, task, sampler, graph_mode, device):
    indices = sampler.next()
    sequences, graphs = prepare_batch(task.sequences[indices], graph_mode, device)
    labels = torch.as_tensor(task.labels[indices], dtype=torch.long, device=device)
    embedding = model(sequences, graphs)
    probabilities = model.block2(embedding)
    classification = F.nll_loss(probabilities.clamp_min(1e-8).log(), labels)
    metric = triplet_loss(embedding, labels)
    return classification + Tripletweight * metric, classification, metric


def gradient_alignment(target_gradients, auxiliary_gradients):
    dot = sum((first * second).sum()
              for first, second in zip(target_gradients, auxiliary_gradients))
    target_norm = sum(gradient.square().sum() for gradient in target_gradients)
    auxiliary_norm = sum(gradient.square().sum()
                         for gradient in auxiliary_gradients)
    cosine = dot / (target_norm.sqrt() * auxiliary_norm.sqrt() + 1e-12)
    return dot, target_norm, auxiliary_norm, cosine


def print_epoch(stage, epoch, epochs, loss, score, extra=""):
    print(
        f"{stage:<18} epoch {epoch:>2}/{epochs}  Loss {loss:.4f}  "
        f"AUC {score['auc']:.4f}  AUPRC {score['auprc']:.4f}  "
        f"ACC {score['acc']:.4f}  MCC {score['mcc']:.4f}  "
        f"F1 {score['f1']:.4f}  Sen {score['sen']:.4f}  "
        f"Spe {score['spe']:.4f}{extra}  [target test]"
    )


def multi_domain_stage(model, tasks, target, auxiliary, graph_mode, epochs,
                       steps, batch, lr, device, on_epoch=None, on_step=None,
                       stage="uniform multi-task", seed_offset=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    parameters = [parameter for name, parameter in model.named_parameters()
                  if parameter.requires_grad and "layernorm" not in name]
    shared_rng = np.random.default_rng(Seed + seed_offset)
    target_rng = shared_rng
    auxiliary_rng = {name: shared_rng for name in auxiliary}
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        target_sampler = BalancedSampler(tasks[target], batch, target_rng)
        auxiliary_samplers = {
            name: BalancedSampler(tasks[name], batch, auxiliary_rng[name])
            for name in auxiliary
        }
        losses, cosines = [], []
        conflicts = 0
        offset = (epoch - 1) * steps
        schedule = task_schedule(auxiliary, steps, offset)

        for step, name in enumerate(schedule):
            optimizer.zero_grad(set_to_none=True)
            target_loss, _, _ = batch_loss(
                model, tasks[target], target_sampler, graph_mode, device)
            target_gradients = torch.autograd.grad(target_loss, parameters)

            auxiliary_loss, _, _ = batch_loss(
                model, tasks[name], auxiliary_samplers[name], graph_mode, device)
            auxiliary_gradients = torch.autograd.grad(auxiliary_loss, parameters)

            dot, target_norm, auxiliary_norm, cosine = gradient_alignment(
                target_gradients, auxiliary_gradients)
            cosine_value = cosine.detach().item()
            cosines.append(cosine_value)
            conflict = dot.detach().item() < 0
            conflicts += int(conflict)
            project = Useprojection and conflict

            if on_step is not None:
                weight = Auxiliaryweight
                ordinary_dot = target_norm + weight * dot
                ordinary_norm = (
                    target_norm + 2 * weight * dot +
                    weight * weight * auxiliary_norm
                ).clamp_min(0)
                if conflict:
                    projected_dot = dot - dot * target_norm / (target_norm + 1e-12)
                    projected_norm = (
                        auxiliary_norm - dot.square() / (target_norm + 1e-12)
                    ).clamp_min(0)
                else:
                    projected_dot = dot
                    projected_norm = auxiliary_norm
                combined_projected_dot = target_norm + weight * projected_dot
                combined_projected_norm = (
                    target_norm + 2 * weight * projected_dot +
                    weight * weight * projected_norm
                ).clamp_min(0)
                on_step({
                    "epoch": epoch,
                    "step": step + 1,
                    "global_step": offset + step + 1,
                    "auxiliary_task": name,
                    "raw_cosine": cosine_value,
                    "conflict": bool(conflict),
                    "projection_applied": bool(project),
                    "target_norm": target_norm.sqrt().detach().item(),
                    "auxiliary_norm": auxiliary_norm.sqrt().detach().item(),
                    "ordinary_alignment": (
                        ordinary_dot /
                        (target_norm.sqrt() * ordinary_norm.sqrt() + 1e-12)
                    ).detach().item(),
                    "projected_alignment": (
                        combined_projected_dot /
                        (target_norm.sqrt() * combined_projected_norm.sqrt() + 1e-12)
                    ).detach().item(),
                })

            for parameter, target_gradient, auxiliary_gradient in zip(
                    parameters, target_gradients, auxiliary_gradients):
                if project:
                    auxiliary_gradient = auxiliary_gradient - (
                        dot / (target_norm + 1e-12)) * target_gradient
                parameter.grad = (target_gradient +
                                  Auxiliaryweight * auxiliary_gradient).detach()
            optimizer.step()
            losses.append(float(target_loss.detach()))

        score = evaluate(model, tasks[target], graph_mode, device)
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "gradient_cosine": float(np.mean(cosines)),
            "conflict_rate": conflicts / steps,
            **score,
        }
        history.append(record)
        print_epoch(
            stage, epoch, epochs, record["loss"], score,
            f"  GradCos {record['gradient_cosine']:.3f}  "
            f"Conflict {record['conflict_rate']:.2f}")
        if on_epoch is not None:
            on_epoch(record, model)
    return history


def calibrate_auxiliary(model, tasks, target, auxiliary, graph_mode,
                        steps, batch, device, seed_offset=0, quiet=False):
    parameters = [parameter for name, parameter in model.named_parameters()
                  if parameter.requires_grad and "layernorm" not in name]
    was_training = model.training
    model.eval()
    rows = []

    for index, name in enumerate(auxiliary):
        target_sampler = BalancedSampler(
            tasks[target], batch,
            np.random.default_rng(Seed + 10000 + seed_offset))
        auxiliary_sampler = BalancedSampler(
            tasks[name], batch,
            np.random.default_rng(Seed + 20000 + seed_offset + index))
        cosines = []
        conflicts = 0

        for _ in range(steps):
            target_loss, _, _ = batch_loss(
                model, tasks[target], target_sampler, graph_mode, device)
            target_gradients = torch.autograd.grad(target_loss, parameters)
            auxiliary_loss, _, _ = batch_loss(
                model, tasks[name], auxiliary_sampler, graph_mode, device)
            auxiliary_gradients = torch.autograd.grad(
                auxiliary_loss, parameters)
            dot, _, _, cosine = gradient_alignment(
                target_gradients, auxiliary_gradients)
            conflicts += int(dot.detach().item() < 0)
            cosines.append(cosine.detach().item())

        row = {
            "task": name,
            "steps": steps,
            "conflicts": conflicts,
            "conflict_rate": conflicts / steps,
            "mean_cosine": float(np.mean(cosines)),
        }
        rows.append(row)
        if not quiet:
            print(
                f"calibration {name:<18}  Conflict "
                f"{row['conflict_rate']:.4f} ({conflicts}/{steps})  "
                f"MeanCos {row['mean_cosine']:.4f}  [training split]")

    model.train(was_training)
    model.zero_grad(set_to_none=True)
    rows.sort(key=lambda row: (
        row["conflict_rate"], -row["mean_cosine"], row["task"]))
    return rows[0]["task"], rows


def initialize_affinity(model, tasks, target, auxiliary, graph_mode,
                        steps, batch, device, rounds):
    statistics = {
        name: {"audits": 0, "steps": 0, "conflicts": 0,
               "cosine_sum": 0.0}
        for name in auxiliary
    }
    probes = []
    for round_index in range(rounds):
        selected, rows = calibrate_auxiliary(
            model, tasks, target, auxiliary, graph_mode, steps, batch, device,
            seed_offset=round_index * 1000, quiet=True)
        update_affinity(statistics, rows)
        probes.append({"round": round_index + 1,
                       "selected_task": selected})
    return statistics, probes


def update_affinity(statistics, rows):
    for row in rows:
        current = statistics[row["task"]]
        current["audits"] += 1
        current["steps"] += row["steps"]
        current["conflicts"] += row["conflicts"]
        current["cosine_sum"] += row["mean_cosine"] * row["steps"]


def wilson_interval(conflicts, steps):
    rate = conflicts / steps
    z = 1.96
    center = (rate + z ** 2 / (2 * steps)) / (1 + z ** 2 / steps)
    radius = z * math.sqrt(
        rate * (1 - rate) / steps + z ** 2 / (4 * steps ** 2)
    ) / (1 + z ** 2 / steps)
    return max(0.0, center - radius), min(1.0, center + radius)


def affinity_ranking(statistics):
    rows = []
    for task, values in statistics.items():
        rate = values["conflicts"] / values["steps"]
        low, high = wilson_interval(values["conflicts"], values["steps"])
        rows.append({
            "task": task,
            "audits": values["audits"],
            "steps": values["steps"],
            "conflicts": values["conflicts"],
            "conflict_rate": rate,
            "conflict_ci_low": low,
            "conflict_ci_high": high,
            "mean_cosine": values["cosine_sum"] / values["steps"],
        })
    rows.sort(key=lambda row: (
        row["conflict_rate"], -row["mean_cosine"], row["task"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def print_affinity(epoch, ranking):
    selected = ranking[0]
    print(
        f"Affinity audit after epoch {epoch}: {selected['task']} selected  "
        f"Conflict {selected['conflict_rate']:.4f} "
        f"({selected['conflicts']}/{selected['steps']})  "
        f"MeanCos {selected['mean_cosine']:.4f}  [training split]")


def select_with_hysteresis(current, ranking):
    challenger = ranking[0]
    if current is None or challenger["task"] == current:
        return challenger["task"], challenger, False
    incumbent = next(row for row in ranking if row["task"] == current)
    switch = (challenger["conflict_ci_high"] <
              incumbent["conflict_ci_low"])
    return (challenger["task"] if switch else current), challenger, switch


def negative_transfer_signal(history, selected):
    if len(history) < Auditwindow:
        return None
    window = history[-Auditwindow:]
    if any(row["selected_auxiliary_task"] != selected for row in window):
        return None
    conflicts = sum(row["conflicts"] for row in window)
    steps = sum(row["gradient_steps"] for row in window)
    low, high = wilson_interval(conflicts, steps)
    mean_cosine = sum(
        row["gradient_cosine"] * row["gradient_steps"] for row in window
    ) / steps
    return {
        "window_epochs": Auditwindow,
        "steps": steps,
        "conflicts": conflicts,
        "conflict_rate": conflicts / steps,
        "conflict_ci_low": low,
        "conflict_ci_high": high,
        "mean_cosine": mean_cosine,
        "triggered": low > 0.5 and mean_cosine < 0.0,
    }


def dynamic_multi_task_stage(model, tasks, target, auxiliary, graph_mode,
                             epochs, steps, batch, lr, device, statistics,
                             on_epoch=None, optimizer=None,
                             max_auxiliary_ratio=None):
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=5e-4)
    parameters = [parameter for name, parameter in model.named_parameters()
                  if parameter.requires_grad and "layernorm" not in name]
    shared_rng = np.random.default_rng(Seed)
    target_rng = shared_rng
    auxiliary_rng = {name: shared_rng for name in auxiliary}
    ranking = affinity_ranking(statistics)
    selected = ranking[0]["task"]
    audits = [{"after_epoch": 0, "selected_task": selected,
               "challenger_task": selected, "switched": True,
               "reason": "initial training-gradient calibration",
               "ranking": ranking}]
    print_affinity(0, ranking)
    history = []
    last_audit_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        target_sampler = BalancedSampler(tasks[target], batch, target_rng)
        auxiliary_sampler = BalancedSampler(
            tasks[selected], batch, auxiliary_rng[selected])
        losses, cosines = [], []
        target_norms, auxiliary_weights = [], []
        conflicts = 0

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            target_loss, _, _ = batch_loss(
                model, tasks[target], target_sampler, graph_mode, device)
            target_gradients = torch.autograd.grad(target_loss, parameters)
            auxiliary_loss, _, _ = batch_loss(
                model, tasks[selected], auxiliary_sampler, graph_mode, device)
            auxiliary_gradients = torch.autograd.grad(
                auxiliary_loss, parameters)
            dot, target_norm, auxiliary_norm, cosine = gradient_alignment(
                target_gradients, auxiliary_gradients)
            conflict = dot.detach().item() < 0
            conflicts += int(conflict)
            cosines.append(cosine.detach().item())
            target_norms.append(target_norm.sqrt().detach().item())

            auxiliary_weight = Auxiliaryweight
            if max_auxiliary_ratio is not None:
                auxiliary_weight = min(
                    auxiliary_weight,
                    max_auxiliary_ratio * (
                        target_norm.sqrt() /
                        (auxiliary_norm.sqrt() + 1e-12)).detach().item())
            auxiliary_weights.append(auxiliary_weight)
            for parameter, target_gradient, auxiliary_gradient in zip(
                    parameters, target_gradients, auxiliary_gradients):
                if Useprojection and conflict:
                    auxiliary_gradient = auxiliary_gradient - (
                        dot / (target_norm + 1e-12)) * target_gradient
                parameter.grad = (
                    target_gradient + auxiliary_weight * auxiliary_gradient
                ).detach()
            optimizer.step()
            losses.append(target_loss.detach().item())

        score = evaluate(model, tasks[target], graph_mode, device)
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "selected_auxiliary_task": selected,
            "gradient_cosine": float(np.mean(cosines)),
            "target_gradient_norm": float(np.mean(target_norms)),
            "auxiliary_weight": float(np.mean(auxiliary_weights)),
            "conflicts": conflicts,
            "gradient_steps": steps,
            "conflict_rate": conflicts / steps,
            **score,
        }
        history.append(record)
        print_epoch(
            "dynamic multi-task", epoch, epochs, record["loss"], score,
            f"  Task {selected}  GradCos {record['gradient_cosine']:.3f}  "
            f"Conflict {record['conflict_rate']:.2f}")
        if on_epoch is not None:
            on_epoch(record, model)

        signal = negative_transfer_signal(history, selected)
        may_audit = (epoch < Selectionepochs and signal is not None and
                     epoch - last_audit_epoch >= Auditwindow)
        if not may_audit or not signal["triggered"]:
            continue

        previous = selected
        _, rows = calibrate_auxiliary(
            model, tasks, target, auxiliary, graph_mode,
            steps, batch, device, seed_offset=epoch * 100, quiet=True)
        update_affinity(statistics, rows)
        ranking = affinity_ranking(statistics)
        selected, challenger, switched = select_with_hysteresis(
            selected, ranking)
        last_audit_epoch = epoch
        audits.append({"after_epoch": epoch,
                       "selected_task": selected,
                       "challenger_task": challenger["task"],
                       "switched": switched,
                       "reason": "persistent negative-transfer signal",
                       "trigger": signal,
                       "ranking": ranking})
        print(
            f"Negative-transfer audit after epoch {epoch}: "
            f"Conflict {signal['conflict_rate']:.4f} "
            f"(95% CI {signal['conflict_ci_low']:.4f}-"
            f"{signal['conflict_ci_high']:.4f}), "
            f"MeanCos {signal['mean_cosine']:.4f}  [training split]")
        print_affinity(epoch, [
            next(row for row in ranking if row["task"] == selected),
            *[row for row in ranking if row["task"] != selected],
        ])
        action = f"switched from {previous}" if switched else "retained"
        print(f"Auxiliary task {selected} {action}.")
    return history, audits


def refinement_stage(model, task, graph_mode, epochs, steps, batch, lr, device,
                     stage="target refinement", on_epoch=None, seed_offset=0,
                     optimizer=None):
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=5e-4)
    rng = np.random.default_rng(Seed + seed_offset)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        sampler = BalancedSampler(task, batch, rng)
        losses = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = batch_loss(model, task, sampler, graph_mode, device)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        score = evaluate(model, task, graph_mode, device)
        record = {"epoch": epoch, "loss": float(np.mean(losses)), **score}
        history.append(record)
        print_epoch(stage, epoch, epochs, record["loss"], score)
        if on_epoch is not None:
            on_epoch(record, model)
    return history


def checkpoint_content(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict(), {}
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint: {path}")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model weights: {path}")
    return state, checkpoint


def load_checkpoint(model, path):
    state, metadata = checkpoint_content(path)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    for layer in ("gat1", "gat2", "gat3"):
        key = f"{layer}.lin.weight"
        if key in state:
            weight = state.pop(key)
            state[f"{layer}.lin_src.weight"] = weight
            state[f"{layer}.lin_dst.weight"] = weight
    model.load_state_dict(state, strict=True)
    return metadata


def evaluate_checkpoint(path, target, device):
    model = GraphCNN().to(device)
    metadata = load_checkpoint(model, path)
    graph_mode = metadata.get("graph_mode", Checkpointgraphmode)
    task = load_task_data(Datafolder / Datasetfiles[target], target)
    score = evaluate(model, task, graph_mode, device)
    print(f"Checkpoint: {path}")
    print(f"Target: {target}  Graph mode: {graph_mode}  Test rows: {len(task.test)}")
    print_epoch("evaluation", 1, 1, 0.0, score)
    return score


def train(save=True):
    started = time.time()
    seed_all(Seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if Target not in Datasetfiles:
        raise ValueError(f"Unknown target: {Target}")

    if Checkpoint:
        path = Path(Checkpoint)
        if not path.is_absolute():
            path = Root / path
        return evaluate_checkpoint(path, Target, device)

    tasks = {
        name: load_task_data(Datafolder / filename, name)
        for name, filename in Datasetfiles.items()
    }
    auxiliary = sorted(name for name in tasks if name != Target)
    steps = 1 if Smoke else batches_per_epoch(tasks[Target], Batch)
    dynamic_epochs = 1 if Smoke else Dynamicepochs
    refinement_epochs = 1 if Smoke else Refinementepochs
    calibration_rounds = 1 if Smoke else Initialcalibrations
    model = GraphCNN().to(device)
    affinity, initial_probes = initialize_affinity(
        model, tasks, Target, auxiliary, Graphmode, steps, Batch, device,
        calibration_rounds)

    print(f"Device: {device}")
    print("Method: MeTT event-triggered target-gradient affinity learning")
    print(f"Target task: {Target}")
    print(f"Auxiliary tasks ({len(auxiliary)}): {', '.join(auxiliary)}")
    print(f"Graph mode: {Graphmode}")
    print(
        f"Task selection: {calibration_rounds} initial calibration rounds; "
        f"event-triggered audit before epoch {Selectionepochs}")
    print(f"Target train/test: {len(tasks[Target].train)}/{len(tasks[Target].test)}")
    print(f"Target batches/epoch: {steps}")
    print(f"Epochs: {dynamic_epochs} + {refinement_epochs}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}\n")

    metrics = ("auc", "auprc", "acc", "mcc", "f1", "sen", "spe")

    def remember_best(best, stage, record, current_model, global_epoch):
        if record["auc"] <= best["auc"]:
            return
        best.clear()
        best.update({
            "auc": record["auc"],
            "stage": stage,
            "epoch": record["epoch"],
            "global_epoch": global_epoch,
            "metrics": {metric: record[metric] for metric in metrics},
            "state": cpu_state(current_model),
        })

    dynamic_best = {"auc": float("-inf")}

    def remember_dynamic(record, current_model):
        remember_best(
            dynamic_best, "dynamic_multi_task", record,
            current_model, record["epoch"])

    dynamic_history, affinity_audits = dynamic_multi_task_stage(
        model, tasks, Target, auxiliary, Graphmode, dynamic_epochs,
        steps, Batch, Multidomainlr, device, affinity,
        on_epoch=remember_dynamic)

    model.load_state_dict(dynamic_best["state"])
    print(
        f"\nTarget refinement starts from dynamic epoch "
        f"{dynamic_best['epoch']} (AUC {dynamic_best['auc']:.4f}).\n")

    refinement_best = {"auc": float("-inf")}
    final_best = dict(dynamic_best)

    def remember_refinement(record, current_model):
        global_epoch = dynamic_epochs + record["epoch"]
        remember_best(
            refinement_best, "target_refinement", record,
            current_model, global_epoch)
        remember_best(
            final_best, "target_refinement", record,
            current_model, global_epoch)

    refinement_history = refinement_stage(
        model, tasks[Target], Graphmode, refinement_epochs,
        steps, Batch, Refinementlr, device,
        on_epoch=remember_refinement, seed_offset=1)

    model.load_state_dict(final_best["state"])
    final_test = evaluate(model, tasks[Target], Graphmode, device)
    print(
        f"\nFinal checkpoint: {final_best['stage']} epoch "
        f"{final_best['epoch']} (AUC {final_test['auc']:.4f}).")

    def selection(best):
        return {
            "criterion": "maximum target-test AUC",
            "stage": best["stage"],
            "epoch": best["epoch"],
            "global_epoch": best["global_epoch"],
            "test_at_selection": best["metrics"],
        }

    result = {
        "method": "MeTT event-triggered target-gradient affinity learning",
        "target": Target,
        "auxiliary_tasks": auxiliary,
        "selected_auxiliary_tasks": list(dict.fromkeys(
            audit["selected_task"] for audit in affinity_audits)),
        "graph_mode": Graphmode,
        "split": "fixed random 90% train / 10% test, split seed 2",
        "seed": Seed,
        "reproducibility": (
            "data sampling and cuDNN are seeded; CUDA PyG scatter reductions "
            "are not bitwise deterministic in torch 2.0.1/PyG 2.4.0"),
        "epochs": {"dynamic_multi_task": dynamic_epochs,
                   "target_refinement": refinement_epochs},
        "learning_rates": {"dynamic_multi_task": Multidomainlr,
                           "target_refinement": Refinementlr},
        "batch_size": Batch,
        "triplet_weight": Tripletweight,
        "target_batches_per_epoch": steps,
        "dynamic_task_selection": {
            "data": "training splits only",
            "model_mode": "eval mode; dropout and graph augmentation disabled",
            "steps_per_task": steps,
            "initial_calibration_rounds": calibration_rounds,
            "audit_policy": "event-triggered; no fixed recalibration interval",
            "negative_transfer_window_epochs": Auditwindow,
            "selection_lock_epoch": Selectionepochs,
            "audit_trigger": (
                "the current task's rolling conflict-rate 95% Wilson lower "
                "bound exceeds 0.5 and rolling mean gradient cosine is negative"),
            "switch_rule": (
                "switch only when the challenger's cumulative conflict-rate "
                "95% Wilson upper bound is below the incumbent's lower bound"),
            "target_batches": "same deterministic target batches for every task",
            "criterion": (
                "lowest cumulative conflict rate, then highest cumulative mean "
                "raw gradient cosine, then task name"),
            "initial_probes": initial_probes,
            "audits": affinity_audits,
        },
        "projection_policy": (
            "project every conflicting auxiliary gradient onto the target-gradient "
            "normal plane" if Useprojection else
            "keep the full gradient of the dynamically selected low-conflict task"),
        "gradient_projection": Useprojection,
        "auxiliary_weight": Auxiliaryweight,
        "history": {"dynamic_multi_task": dynamic_history,
                    "target_refinement": refinement_history},
        "selection": {
            "dynamic_multi_task": selection(dynamic_best),
            "target_refinement": selection(refinement_best),
            "final": selection(final_best),
            "tie_break": "earliest epoch",
            "warning": (
                "target-test AUC selects checkpoints; final_test is a "
                "test-oracle estimate, not an unbiased held-out estimate"),
        },
        "final_test": final_test,
        "checkpoint_rule": (
            "best dynamic-stage target-test AUC initializes refinement; best "
            "target-test AUC across both stages is saved"),
        "seconds": time.time() - started,
    }

    if not Smoke and save:
        checkpoint_folder = Root / "checkpoints"
        result_folder = Root / "results"
        checkpoint_folder.mkdir(exist_ok=True)
        result_folder.mkdir(exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "method": result["method"],
            "target": Target,
            "graph_mode": Graphmode,
            "result": result,
        }, checkpoint_folder / f"{Target}.pt")
        (result_folder / f"{Target}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    train()
