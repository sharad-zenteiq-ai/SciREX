import os
import sys

# Force deterministic GPU operations for reproducibility
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax import linen as nn
import time
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass, field
from typing import Tuple, List

from scirex.operators.models.gino import GINO
from scirex.operators.training import TrainState
from scirex.operators.losses import lp_loss
from scirex.operators.data.car_cfd_dataset import CarCFDDataset
from scirex.operators.training.normalizers import GaussianNormalizer
from configs.gino_carcfd_config import GINOCarCFDConfig
    
def make_schedule(config: GINOCarCFDConfig, steps_per_epoch: int):
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = min(310, total_steps // 10)
    
    if config.scheduler_type == "cosine":
        cosine_decay_steps = config.cosine_decay_epochs * steps_per_epoch - warmup_steps
        cosine_decay_steps = max(cosine_decay_steps, 1)
        
        cosine_schedule = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=cosine_decay_steps,
            alpha=0.0
        )
        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, config.learning_rate, warmup_steps),
                cosine_schedule
            ],
            boundaries=[warmup_steps]
        )
    else:
        schedule = optax.constant_schedule(config.learning_rate)
    
    return schedule

def main():
    config = GINOCarCFDConfig()
    
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    
    print(f"Loading CarCFDDataset from {config.data_root}...")
    dataset = CarCFDDataset(
        root_dir=config.data_root,
        n_train=config.n_train,
        n_test=config.n_test,
        query_res=config.query_res,
        download=config.download
    )
    
    print(f"Initializing GINO model...")
    model = GINO(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        gno_coord_dim=config.gno_coord_dim,
        in_gno_radius=config.in_gno_radius,
        out_gno_radius=config.out_gno_radius,
        in_gno_transform_type=config.in_gno_transform_type,
        out_gno_transform_type=config.out_gno_transform_type,
        fno_n_modes=config.fno_n_modes,
        fno_hidden_channels=config.fno_hidden_channels,
        fno_lifting_channel_ratio=config.fno_lifting_channel_ratio,
        fno_n_layers=config.fno_n_layers
    )
    
    try:
        dummy_dict = next(dataset.train_generator(batch_size=1, shuffle=False))
    except StopIteration:
        print("Warning: Failed to fetch data. Dataset might be empty.")
        return

    input_geom = jnp.asarray(dummy_dict["vertices"])
    latent_queries = jnp.asarray(dummy_dict["query_points"])
    output_queries = jnp.asarray(dummy_dict["vertices"])
    latent_features = jnp.asarray(dummy_dict["distance"])
    
    # In some datasets, latent features lack the trailing channel block
    if latent_features.ndim == 4:
        latent_features = jnp.expand_dims(latent_features, -1)
        
    print("Init sizes | Geom:", input_geom.shape, "| Queries:", latent_queries.shape, "| Features:", latent_features.shape)
    
    variables = model.init(
        init_rng,
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        latent_features=latent_features
    )
    params = variables["params"]
    
    steps_per_epoch = max(1, config.n_train // config.batch_size)
    schedule = make_schedule(config, steps_per_epoch)
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
    )
    
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    @jax.jit
    def train_step(state, batch_geom, batch_queries, batch_out_queries, batch_lat_features, batch_y):
        def loss_fn(params):
            pred = state.apply_fn(
                {"params": params},
                input_geom=batch_geom,
                latent_queries=batch_queries,
                output_queries=batch_out_queries,
                latent_features=batch_lat_features
            )
            return lp_loss(pred, batch_y)
            
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, {"loss": loss}

    @jax.jit
    def eval_step(state, batch_geom, batch_queries, batch_out_queries, batch_lat_features, batch_y):
        pred = state.apply_fn(
            {"params": state.params},
            input_geom=batch_geom,
            latent_queries=batch_queries,
            output_queries=batch_out_queries,
            latent_features=batch_lat_features
        )
        return {"loss": lp_loss(pred, batch_y)}

    print(f"Starting training for {config.epochs} epochs...")
    best_rel_l2 = float("inf")
    history = {"train_rel_l2": [], "test_rel_l2": []}
    
    ckpt_dir = os.path.join(project_root, "experiments/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "gino_carcfd_params.pkl")
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        for batch in dataset.train_generator(batch_size=config.batch_size, shuffle=True):
            batch_geom = jnp.asarray(batch["vertices"])
            batch_queries = jnp.asarray(batch["query_points"])
            batch_out_queries = jnp.asarray(batch["vertices"])
            batch_lat_features = jnp.asarray(batch["distance"])
            batch_y = jnp.asarray(batch["press"])
            
            if batch_y.ndim == 2:
                batch_y = jnp.expand_dims(batch_y, -1)
            if batch_lat_features.ndim == 4:
                batch_lat_features = jnp.expand_dims(batch_lat_features, -1)
                
            state, metrics = train_step(
                state, batch_geom, batch_queries, batch_out_queries, batch_lat_features, batch_y
            )
            epoch_loss += float(metrics["loss"])
            
        avg_train_loss = epoch_loss / steps_per_epoch
        
        # Test
        test_loss = 0.0
        test_steps = 0
        for batch in dataset.test_generator(batch_size=config.batch_size, shuffle=False):
            batch_geom = jnp.asarray(batch["vertices"])
            batch_queries = jnp.asarray(batch["query_points"])
            batch_out_queries = jnp.asarray(batch["vertices"])
            batch_lat_features = jnp.asarray(batch["distance"])
            batch_y = jnp.asarray(batch["press"])
            
            if batch_y.ndim == 2:
                batch_y = jnp.expand_dims(batch_y, -1)
            if batch_lat_features.ndim == 4:
                batch_lat_features = jnp.expand_dims(batch_lat_features, -1)
                
            metrics = eval_step(
                state, batch_geom, batch_queries, batch_out_queries, batch_lat_features, batch_y
            )
            test_loss += float(metrics["loss"])
            test_steps += 1
            
        avg_test_loss = test_loss / max(test_steps, 1)
        
        history["train_rel_l2"].append(avg_train_loss)
        history["test_rel_l2"].append(avg_test_loss)
        
        if avg_test_loss < best_rel_l2:
            best_rel_l2 = avg_test_loss
            with open(ckpt_path, "wb") as f:
                f.write(flax.serialization.to_bytes(state.params))
                
        if epoch % 1 == 0:
            print(f"Epoch {epoch:4d} | Train Rel L2: {avg_train_loss:.6f} | "
                  f"Test Rel L2: {avg_test_loss:.6f} | Best: {best_rel_l2:.6f} | Time: {time.time()-epoch_start:.2f}s")
                  
    print("Training Complete!")

if __name__ == "__main__":
    main()
