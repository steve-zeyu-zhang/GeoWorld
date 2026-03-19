# Models

GeoWorld uses a frozen original V-JEPA 2 encoder and a trainable action-conditioned predictor. The architecture is described in [Sections 3 and 4.3](https://arxiv.org/html/2602.23058v2) and adapts the [official V-JEPA 2-AC predictor](https://github.com/facebookresearch/vjepa2/blob/c2963a47433ecca0ad4f06ec28bcfa8cb5b5cefb/src/models/ac_predictor.py). The encoder and attention blocks come from revision `c2963a47433ecca0ad4f06ec28bcfa8cb5b5cefb`, before the V-JEPA 2.1 release. Source files, licenses, and original/vendored SHA-256 hashes are in `third_party/vjepa2`.

## Frozen perception

`FrozenVJEPA2Encoder` directly constructs the official video Vision Transformer with 16-pixel spatial patches, two-frame temporal tubelets, 3D rotary positions, and scaled dot-product attention. The original rotary-position implementation is preserved, including the upstream compatibility behavior documented in `modules.py`.

The default `vit_large` encoder has 24 blocks, width 1024, and 16 attention heads. The official [VideoMix22M pretrained checkpoint](https://dl.fbaipublicfiles.com/vjepa2/vitl.pt) supplies the frozen weights. Alternative supported original V-JEPA 2 encoders are `vit_huge` (1280 output dimensions), `vit_giant` (1408), and `vit_giant_384` (1408, intended for a 384-pixel crop). ViT-L is the default backbone. Their checkpoint URLs follow the pinned [official hub implementation](https://github.com/facebookresearch/vjepa2/blob/c2963a47433ecca0ad4f06ec28bcfa8cb5b5cefb/src/hub/backbones.py).

The loader reads the checkpoint's `encoder` dictionary, removes `module.` and `backbone.` prefixes, and checks every parameter name and shape. It allows only the obsolete `pos_embed` tensor that the upstream RoPE loader explicitly ignores. Missing layers, unexpected tensors, and shape mismatches raise errors. `encoder.load_report` records the result. Training and optimizer states in the downloaded file are not loaded into the model. Pretrained encoders require valid checkpoint weights. The `name: tiny, pretrained: false` option provides a small version of the same architecture for functional tests.

Inputs are raw RGB clips `[B,3,F,H,W]`, either uint8 or floating point in `[0,1]`. `encoder.preprocess` resizes the shorter edge to `floor(image_size * 256/224)`, center crops to `image_size`, and applies ImageNet RGB mean `(0.485,0.456,0.406)` and standard deviation `(0.229,0.224,0.225)`. These operations follow the upstream single-view evaluation recipe, using tensor bilinear interpolation with antialiasing for resizing. Temporal frame selection belongs to the dataset. The frame count must be even and at least two. `encode_tokens` returns `[B,F/2,H/16,W/16,D]`.

Default feature extraction averages all spatial patches and temporal tubelets to obtain one vector `[B,D]` per boundary clip. `temporal_pool: last` instead selects the final tubelet after spatial averaging. All encoder parameters are frozen, feature extraction runs without autograd, and calling `model.train()` keeps the encoder in evaluation mode.

## Action-conditioned hyperbolic predictor

`HyperbolicPredictor` reuses the vendored `ACBlock` and `ACRoPEAttention` directly. The paper configuration has 24 blocks, width 1024, 16 heads, a 4x feed-forward expansion, GELU, pre-normalization, and no attention dropout. Linear weights use the upstream truncated-normal initialization with standard deviation 0.02, and each block's attention/MLP output weights are divided by `sqrt(2 * layer_index)`.

The original robotics predictor interleaves continuous robot-action, robot-state, optional extrinsics, and visual patch tokens. GeoWorld's inputs provide discrete procedural actions and video features. Each time step therefore contains one learned action-ID embedding and one projected pooled visual-state token. The original attention receives `H=W=1` and `action_tokens=1`. Its block-causal mask allows both tokens in the current step to attend to each other and to earlier steps. Future actions and states are inaccessible. The action-conditioned transformer uses the dataset's discrete action vocabulary.

The positive curvature magnitude is `c = exp(clamp(log_c, log(c_min), log(c_max)))`, initialized to one; sectional curvature is `-c`. The bounds of 0.1 and 10 follow the paper's supplementary curvature ablation. Frozen Euclidean features `f` are mapped to the Poincare ball by

```text
z = Exp_0^c(s * f),     s = tangent_scale = 0.01.
```

The transformer consumes ball coordinates `z`, as in the paper's `P(z,a)`. A final linear projection produces tangent coordinates, followed by `Exp_0^c(s * projection)` to produce the next valid ball state. The same fixed scale on inputs and outputs avoids immediate boundary saturation with 1024-dimensional features and the upstream initialization. The scale is fixed during training. Geometry uses float32 (or float64 when explicitly supplied), including when transformer operations use mixed precision.

`forward_sequence(states, actions)` predicts all next states with teacher-forced causal history. `forward(state, action)` is its single-step interface. `rollout(initial, actions)` repeatedly predicts from the complete generated state/action prefix, returning `[B,T+1,D]` including the initial state. This retains sequence context during autoregressive rollout. It recomputes the prefix at each step rather than using an attention cache.

## Interfaces

```python
from geoworld.models import GeoWorldModel, HyperbolicPredictor

model = GeoWorldModel(config["model"])
features = model.encode(raw_video)                 # [B,D], frozen
state = model.predictor.encode_states(features)    # [B,D], on ball
next_state = model.predictor(state, action_ids)    # action_ids: int64 [B]
trajectory = model.predictor.rollout(state, ids)   # ids: [B,T]; output: [B,T+1,D]

# Cached-feature training constructs only the trainable predictor.
predictor = HyperbolicPredictor(**config["model"]["predictor"])
```

Supported configuration:

```yaml
model:
  encoder:
    name: vit_large
    checkpoint: checkpoints/vjepa2-vitl.pt
    pretrained: true
    frames: 16
    image_size: 256
    spatial_pool: mean
    temporal_pool: mean
  predictor:
    feature_dim: 1024
    hidden_dim: 1024
    depth: 24
    num_heads: 16
    num_actions: 105
    curvature_init: 1.0
    curvature_min: 0.1
    curvature_max: 10.0
    tangent_scale: 0.01
    mlp_ratio: 4.0
    activation_checkpointing: false
```

The action count must match the prepared dataset vocabulary; 105 is the CrossTask configuration, not a universal model constant. Change the predictor feature dimension when selecting a different encoder. Omit `checkpoint` with `pretrained: true` to download the official model through Torch Hub's cache. An explicitly provided but missing checkpoint path is an error.

## Configuration and validation

Configuration controls the backbone, frame count, crop size, feature pooling, predictor dimensions, curvature bounds, and tangent scale. The encoder loads pretrained VideoMix22M weights and stays frozen. The action-conditioned predictor is initialized and trained on the prepared procedural data.

`tests/test_models.py` checks the actual vendored encoder and AC blocks, frozen perception, checkpoint validation, video normalization/pooling, action sensitivity, causal attention, history-preserving rollout, valid ball outputs, finite trainable-curvature gradients, and activation-checkpointed gradients. Dataset evaluation runs separately through `geoworld.evaluate`.

Small test models require at least 12 channels per attention head. The preserved upstream rotary implementation has a singleton-dimension issue below that size; configuration validation rejects it without changing the checkpoint-compatible attention code.
