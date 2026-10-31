"""Small helpers for experiments.ipynb. Experimental decisions live in the notebook."""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def load_data(root):
    """Reuse the original splits and resolve paths from the supplied manifests."""
    root = Path(root)
    split_path = root / 'results/splits.json'
    split_document = json.loads(split_path.read_text())
    records = split_document['records']
    video_hashes = {record['sha256'] for record in records}
    assert len(video_hashes) == len(records)

    data = {}
    for task in ('direction', 'speed', 'acceleration'):
        task_records = [record for record in records if record['task'] == task]
        data_directory = root / 'vjepa-physics-takehome-4E00/data' / task
        manifest_path = data_directory / 'manifest.jsonl'
        manifest_lines = manifest_path.read_text().splitlines()

        manifest = {}
        for line in manifest_lines:
            entry = json.loads(line)
            manifest[entry['id']] = entry

        metadata = []
        video_paths = []
        for record in task_records:
            entry = manifest[record['id']]
            metadata_path = data_directory / entry['metadata']
            clip_metadata = json.loads(metadata_path.read_text())
            metadata.append(clip_metadata)
            video_paths.append(data_directory / entry['video'])

        saved_metadata = [record['metadata'] for record in task_records]
        assert metadata == saved_metadata, 'Metadata changed since the saved split'

        if task == 'direction':
            degrees = [clip['theta_degrees'] for clip in metadata]
            radians = np.deg2rad(degrees)
            sine_targets = np.sin(radians)
            cosine_targets = np.cos(radians)
            targets = np.column_stack((sine_targets, cosine_targets))
        else:
            label_field = 'speed_mps' if task == 'speed' else 'acceleration_mps2'
            magnitudes = [clip[label_field] for clip in metadata]
            targets = np.array(magnitudes)
            targets = targets[:, None]  # Keep a target dimension for scalar regression.

        video_ids = [record['id'] for record in task_records]
        split_labels = [record['split'] for record in task_records]
        assert set(split_labels) == set('ABCD')
        data[task] = {
            'ids': np.array(video_ids),
            'videos': video_paths,
            'y': targets,
            'split': np.array(split_labels),
        }
    return data


def load_features(path, ids):
    """One file per dataset: features [video, block, channel] and ordered IDs."""
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved['ids'], ids)
        features = saved['features']
    assert features.shape == (len(ids), 24, 1024)
    assert np.isfinite(features).all()
    return features


def load_model(checkpoint, device='cuda'):
    from transformers import VJEPA2Model

    model = VJEPA2Model.from_pretrained(
        checkpoint,
        local_files_only=True,
        attn_implementation='sdpa',
    )
    model.eval()
    model = model.to(device)
    model.requires_grad_(False)
    return model


def load_video(path):
    import av

    frames = []
    with av.open(str(path)) as video:
        for frame in video.decode(video=0):
            rgb_frame = frame.to_ndarray(format='rgb24')
            frames.append(rgb_frame)
    frames = np.stack(frames)
    assert frames.shape == (16, 256, 256, 3)
    return frames


def extract_features(model, videos, batch_size=2):
    """Normalize native RGB frames, then mean-pool each raw block output."""
    import torch

    first_parameter = next(model.parameters())
    device = first_parameter.device
    channel_mean = torch.tensor([.485, .456, .406], device=device)
    channel_std = torch.tensor([.229, .224, .225], device=device)
    # Broadcast normalization over batch, time, height, and width.
    channel_mean = channel_mean[None, None, :, None, None]
    channel_std = channel_std[None, None, :, None, None]
    pooled_by_layer = {}

    def make_capture_hook(layer_index):
        def capture(module, inputs, output):
            tokens = output[0] if isinstance(output, tuple) else output
            assert tokens.shape[1:] == (2048, 1024)
            tokens = tokens.float()
            # Pool space and time, retaining the 1,024 feature channels.
            pooled_by_layer[layer_index] = tokens.mean(dim=1)
        return capture

    handles = []
    for layer_index, block in enumerate(model.encoder.layer):
        capture_hook = make_capture_hook(layer_index)
        handle = block.register_forward_hook(capture_hook)
        handles.append(handle)

    feature_batches = []
    try:
        with torch.inference_mode():
            for start in range(0, len(videos), batch_size):
                batch_paths = videos[start:start + batch_size]
                decoded_videos = [load_video(path) for path in batch_paths]
                frames = np.stack(decoded_videos)
                pixels = torch.from_numpy(frames)
                # [batch, time, height, width, RGB] -> [batch, time, RGB, height, width]
                pixels = pixels.permute(0, 1, 4, 2, 3)
                pixels = pixels.to(device)
                pixels = pixels.float()
                pixels = pixels / 255
                normalized_pixels = (pixels - channel_mean) / channel_std

                # Hooks capture raw residuals before the encoder's final LayerNorm.
                model.get_vision_features(normalized_pixels)
                layer_features = [pooled_by_layer[index] for index in range(24)]
                batch_features = torch.stack(layer_features, dim=1)
                batch_features = batch_features.cpu()
                feature_batches.append(batch_features.numpy())
    finally:
        for handle in handles:
            handle.remove()

    features = np.concatenate(feature_batches)
    assert np.isfinite(features).all()
    return features


def metrics(y, prediction, task):
    prediction = np.asarray(prediction)
    prediction = prediction.reshape(y.shape)
    if task == 'direction':
        predicted_sine = prediction[:, 0]
        predicted_cosine = prediction[:, 1]
        predicted_angle = np.arctan2(predicted_sine, predicted_cosine)

        # A nearly zero predicted vector has no reliable angle; use zero consistently.
        vector_lengths = np.linalg.norm(prediction, axis=1)
        predicted_angle[vector_lengths < 1e-8] = 0
        true_angle = np.arctan2(y[:, 0], y[:, 1])
        angle_difference = predicted_angle - true_angle
        wrapped_difference = (angle_difference + np.pi) % (2 * np.pi) - np.pi
        absolute_error = np.abs(wrapped_difference)
        absolute_error = np.rad2deg(absolute_error)
    else:
        absolute_error = np.abs(y - prediction)

    r_squared = r2_score(y, prediction)
    mean_absolute_error = absolute_error.mean()
    return {'r2': float(r_squared), 'mae': float(mean_absolute_error)}


def fit_probe(x_train, y_train, x_val, y_val, task, alphas):
    """Fit on training data; select regularization using validation MAE only."""
    training_features = x_train.astype(float)
    validation_features = x_val.astype(float)
    best_probe = None
    best_score = None

    for alpha in alphas:
        probe = Ridge(alpha=alpha, solver='cholesky')
        probe.fit(training_features, y_train)
        validation_prediction = probe.predict(validation_features)
        score = metrics(y_val, validation_prediction, task)
        if best_score is None or score['mae'] < best_score['mae']:
            best_probe = probe
            best_score = score

    return best_probe, best_score


def bootstrap_interval(y, prediction, task, seed=20260929, repeats=500):
    rng = np.random.default_rng(seed)
    prediction = np.asarray(prediction)
    prediction = prediction.reshape(y.shape)
    scores = []
    for _ in range(repeats):
        sampled_indices = rng.integers(0, len(y), len(y))
        sampled_targets = y[sampled_indices]
        sampled_predictions = prediction[sampled_indices]
        score = metrics(sampled_targets, sampled_predictions, task)
        scores.append([score['r2'], score['mae']])

    bounds = np.quantile(scores, [.025, .975], axis=0)
    r_squared_interval = bounds[:, 0].tolist()
    mae_interval = bounds[:, 1].tolist()
    return {'r2': r_squared_interval, 'mae': mae_interval}


def project_out(features, removed_basis):
    """Remove components along the orthonormal columns of removed_basis."""
    coordinates = features @ removed_basis
    removed_component = coordinates @ removed_basis.T
    remaining_features = features - removed_component
    return remaining_features


def extend_removed_basis(removed_basis, probe_weights, max_dimensions):
    """Add independent probe readout directions, up to the dimension budget."""
    weights = np.atleast_2d(probe_weights)

    # A newly fitted probe can contain tiny numerical components in old directions.
    new_weights = project_out(weights, removed_basis)
    new_weights = project_out(new_weights, removed_basis)
    _, singular_values, right_vectors = np.linalg.svd(new_weights, full_matrices=False)

    weight_scale = np.linalg.norm(weights)
    rank_tolerance = max(1e-12, 1e-8 * weight_scale)
    independent_count = int(np.sum(singular_values > rank_tolerance))
    remaining_budget = max_dimensions - removed_basis.shape[1]
    added_count = min(independent_count, remaining_budget)
    if added_count == 0:
        return removed_basis

    new_directions = right_vectors[:added_count].T
    extended_basis = np.column_stack((removed_basis, new_directions))
    # Reorthogonalization prevents projection errors accumulating over many steps.
    extended_basis, _ = np.linalg.qr(extended_basis, mode='reduced')
    return extended_basis


def steering_basis(probe_weights):
    """Orthonormal basis for the row space of the selected probes."""
    feature_count = probe_weights.shape[-1]
    stacked_weights = probe_weights.reshape(-1, feature_count)
    _, singular_values, right_vectors = np.linalg.svd(stacked_weights, full_matrices=False)

    # Drop numerical dependencies rather than inventing extra subspace directions.
    largest_singular_value = singular_values[0]
    tolerance = 1e-10 * largest_singular_value
    rank = int(np.sum(singular_values > tolerance))
    basis = right_vectors[:rank].T
    return basis


def steer_features(centered_features, probe_weights, probe_biases, basis, target):
    """Least-squares target matching with the smallest change in subspace coordinates.

    Features must be centered using the construction probes' training mean.
    Probe weights are the saved effective weights in that same coordinate system.
    The target contains one scalar, or the direction's sine/cosine pair.
    """
    feature_count = centered_features.shape[1]
    probe_count = len(probe_weights)
    stacked_weights = probe_weights.reshape(-1, feature_count)
    stacked_biases = probe_biases.reshape(-1)
    stacked_target = np.tile(target, probe_count)

    source_coordinates = centered_features @ basis
    readout_in_subspace = stacked_weights @ basis
    source_readouts = source_coordinates @ readout_in_subspace.T
    source_readouts = source_readouts + stacked_biases
    required_readout_change = stacked_target - source_readouts

    # Solve for a delta, not absolute coordinates. This retains any unconstrained
    # source coordinates and chooses the minimum-norm change among LS solutions.
    coordinate_change, _, _, _ = np.linalg.lstsq(
        readout_in_subspace,
        required_readout_change.T,
        rcond=1e-10,
    )
    feature_change = coordinate_change.T @ basis.T
    steered_features = centered_features + feature_change
    return steered_features


def target_errors(predictions, target, task):
    """Per-example target error; R² is not meaningful for a constant target."""
    target = np.asarray(target)
    predictions = np.asarray(predictions)
    predictions = predictions.reshape(-1, len(target))

    if task == 'direction':
        predicted_angle = np.arctan2(predictions[:, 0], predictions[:, 1])
        vector_lengths = np.linalg.norm(predictions, axis=1)
        predicted_angle[vector_lengths < 1e-8] = 0
        target_angle = np.arctan2(target[0], target[1])
        angle_difference = predicted_angle - target_angle
        wrapped_difference = (angle_difference + np.pi) % (2 * np.pi) - np.pi
        absolute_error = np.abs(wrapped_difference)
        return np.rad2deg(absolute_error)

    absolute_error = np.abs(predictions[:, 0] - target[0])
    return absolute_error


def fit_centroid_spline(coordinates, labels, task, knot_count, bounds):
    """Average around fixed label anchors, then interpolate the PCA centroids.

    Boundary anchors represent half-width bins. Coordinates are physical labels,
    not inferred PCA angles. All fitting data must come from the training split.
    """
    from scipy.interpolate import CubicSpline

    lower, upper = bounds
    periodic = task == 'direction'
    anchors = np.linspace(lower, upper, knot_count, endpoint=not periodic)
    distances = np.abs(labels[:, None] - anchors[None, :])
    if periodic:
        period = upper - lower
        distances = distances % period
        distances = np.minimum(distances, period - distances)
    assignments = np.argmin(distances, axis=1)
    counts = np.bincount(assignments, minlength=knot_count)
    if np.any(counts == 0):
        raise ValueError('An empty label bin cannot define a centroid.')

    centroids = []
    for index in range(knot_count):
        in_bin = assignments == index
        centroid = coordinates[in_bin].mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    if periodic:
        knots = np.append(anchors, upper)
        values = np.vstack([centroids, centroids[0]])
        curve = CubicSpline(knots, values, bc_type='periodic', extrapolate='periodic')
    else:
        curve = CubicSpline(anchors, centroids, bc_type='natural', extrapolate=False)
    return curve, anchors, centroids, counts


def replace_pca_coordinates(features, pca, target_coordinates):
    """Replace PCA coordinates, retaining the original orthogonal component."""
    source_coordinates = pca.transform(features)
    coordinate_change = target_coordinates - source_coordinates
    feature_change = coordinate_change @ pca.components_
    return features + feature_change
