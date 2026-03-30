import os
import sys

# Force deterministic GPU operations for reproducibility
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# JAX Memory management
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    total_steps = config.opt.n_epochs * steps_per_epoch
    warmup_steps = min(310, total_steps // 10)
    
    if config.opt.scheduler == "cosine":
        cosine_decay_steps = config.opt.cosine_decay_epochs * steps_per_epoch - warmup_steps
        cosine_decay_steps = max(cosine_decay_steps, 1)
        
        cosine_schedule = optax.cosine_decay_schedule(
            init_value=config.opt.learning_rate,
            decay_steps=cosine_decay_steps,
            alpha=0.0
        )
        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, config.opt.learning_rate, warmup_steps),
                cosine_schedule
            ],
            boundaries=[warmup_steps]
        )
    else:
        schedule = optax.constant_schedule(config.opt.learning_rate)
    
    return schedule

def main():
    config = GINOCarCFDConfig()
    
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    
    print(f"Loading CarCFDDataset from {config.data.data_root}...")
    dataset = CarCFDDataset(
        root_dir=config.data.data_root,
        n_train=config.data.n_train,
        n_test=config.data.n_test,
        query_res=config.data.query_res,
        download=config.data.download,
        max_neighbors=config.model.max_neighbors,
        in_gno_radius=config.model.in_gno_radius,
        out_gno_radius=config.model.out_gno_radius,
        neighbor_cache_dir=config.data.neighbor_cache_dir,
        use_cache=True
    )
    
    print(f"Initializing GINO model...")
    model = GINO(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        gno_coord_dim=config.model.gno_coord_dim,
        in_gno_radius=config.model.in_gno_radius,
        out_gno_radius=config.model.out_gno_radius,
        in_gno_transform_type=config.model.in_gno_transform_type,
        out_gno_transform_type=config.model.out_gno_transform_type,
        fno_n_modes=config.model.fno_n_modes,
        fno_hidden_channels=config.model.fno_hidden_channels,
        fno_lifting_channel_ratio=config.model.fno_lifting_channel_ratio,
        fno_n_layers=config.model.fno_n_layers
    )
    
    try:
        dummy_dict = next(dataset.get_batch("train", batch_size=1))
    except StopIteration:
        print("Warning: Failed to fetch data. Dataset might be empty.")
        return

    input_geom = jnp.asarray(dummy_dict["vertices"])
    latent_queries = jnp.asarray(dummy_dict["query_points"])
    output_queries = jnp.asarray(dummy_dict["vertices"])
    latent_features = jnp.asarray(dummy_dict["distance"])
    
    in_nb = jax.tree_util.tree_map(jnp.asarray, dummy_dict["in_neighbors"])
    out_nb = jax.tree_util.tree_map(jnp.asarray, dummy_dict["out_neighbors"])
    
    # In some datasets, latent features lack the trailing channel block
    if latent_features.ndim == 4:
        latent_features = jnp.expand_dims(latent_features, -1)
        
    print("Init sizes | Geom:", input_geom.shape, "| Queries:", latent_queries.shape, "| Features:", latent_features.shape)
    
    variables = model.init(
        init_rng,
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        latent_features=latent_features,
        in_neighbors=in_nb,
        out_neighbors=out_nb
    )
    params = variables["params"]
    
    steps_per_epoch = config.get_steps_per_epoch()
    schedule = make_schedule(config, steps_per_epoch)
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=config.opt.weight_decay)
    )
    
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    @jax.jit
    def train_step(state, batch_geom, batch_queries, batch_out_queries, batch_lat_features, batch_y, in_nb, out_nb):
        def loss_fn(params):
            pred = state.apply_fn(
                {"params": params},
                input_geom=batch_geom,
                latent_queries=batch_queries,
                output_queries=batch_out_queries,
                latent_features=batch_lat_features,
                in_neighbors=in_nb,
                out_neighbors=out_nb
            )
            return lp_loss(pred, batch_y)
            
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, {"loss": loss}

    @jax.jit
    def eval_step(state, batch_geom, batch_queries, batch_out_queries, batch_lat_features, batch_y, in_nb, out_nb):
        pred = state.apply_fn(
            {"params": state.params},
            input_geom=batch_geom,
            latent_queries=batch_queries,
            output_queries=batch_out_queries,
            latent_features=batch_lat_features,
            in_neighbors=in_nb,
            out_neighbors=out_nb
        )
        return {"loss": lp_loss(pred, batch_y)}

    print(f"Starting training for {config.opt.n_epochs} epochs...")
    best_rel_l2 = float("inf")
    history = {"train_rel_l2": [], "test_rel_l2": []}
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(config.checkpoint_dir, config.model_name)
    
    for epoch in range(config.opt.n_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        for batch in dataset.get_batch("train", batch_size=config.data.batch_size):
            batch_geom = batch["vertices"]
            batch_queries = batch["query_points"]
            batch_out_queries = batch["vertices"]
            batch_lat_features = batch["distance"]
            batch_y = batch["press"]
            
            # Ensure proper channel dimensions for JAX ops: (batch, n_points, channels)
            if batch_y.ndim == 2:
                batch_y = batch_y[:, :, None]
            elif batch_y.ndim == 3 and batch_y.shape[1] == 1:
                # Handle (batch, 1, n_points) if it somehow appears
                batch_y = batch_y.transpose(0, 2, 1)
            
            if batch_lat_features.ndim == 4:
                batch_lat_features = batch_lat_features[:, :, :, :, None]
                
            batch_in_nb = batch["in_neighbors"]
            batch_out_nb = batch["out_neighbors"]
            
            state, metrics = train_step(
                state, batch_geom, batch_queries, batch_out_queries, batch_lat_features, batch_y,
                batch_in_nb, batch_out_nb
            )
            epoch_loss += float(metrics["loss"])
            
        avg_train_loss = epoch_loss / steps_per_epoch
        
        # Test
        test_loss = 0.0
        test_steps = 0
        for batch in dataset.get_batch("test", batch_size=config.data.batch_size):
            batch_geom = batch["vertices"]
            batch_queries = batch["query_points"]
            batch_out_queries = batch["vertices"]
            batch_lat_features = batch["distance"]
            batch_y = batch["press"]
            
            if batch_y.ndim == 2: batch_y = batch_y[:, :, None]
            if batch_lat_features.ndim == 4: batch_lat_features = batch_lat_features[:, :, :, :, None]
                
            batch_in_nb = batch["in_neighbors"]
            batch_out_nb = batch["out_neighbors"]

            metrics = eval_step(
                state, batch_geom, batch_queries, batch_out_queries, batch_lat_features, batch_y,
                batch_in_nb, batch_out_nb
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
