# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import unittest
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tabfm.src import model as tabfm_model
from tabfm.src.classifier_and_regressor import EnsembleGenerator, TabFMClassifier, TabFMRegressor, TransformToNumerical

class EnsembleGeneratorTest(absltest.TestCase):

  def test_permute_categorical_structure(self):
    # Create a simple dataset with one categorical column and one numerical
    X = pd.DataFrame({
        "cat": ["A", "B", "A", "B", "C"],
        "num": [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    y = np.array([0, 1, 0, 1, 0])

    # Pre-encode using TransformToNumerical as the Classifier does
    encoder = TransformToNumerical(min_cat_frequency=1)
    X_enc = encoder.fit_transform(X)
    # "cat" should be column 0 (value 0, 1, 2), "num" column 1.

    cat_features = [0]
    n_estimators = 10

    generator = EnsembleGenerator(
        n_estimators=n_estimators,
        norm_methods=["none"],  # Use none to minimize transformation noise
        cat_features=cat_features,
        permute_categorical=True,
        random_state=42
    )

    generator.fit(X_enc, y)

    # Check if permutations are generated
    self.assertLen(generator.cat_permutations_, 1) # One group for "none" norm
    perms = generator.cat_permutations_["none"]
    self.assertLen(perms, n_estimators)

    # Verify structure of permutations
    for perm in perms:
      self.assertIn(0, perm) # Column 0 should be in the permutation dict
      mapping = perm[0]
      # Original values are 0, 1, 2
      original_values = set(range(3))
      self.assertEqual(set(mapping.keys()), original_values)
      self.assertEqual(set(mapping.values()), original_values)

    # Verify that we actually have different permutations
    # It's statistically improbable that 10 estimators all have identity permutation
    identity_count = 0
    for perm in perms:
      mapping = perm[0]
      is_identity = all(k == v for k, v in mapping.items())
      if is_identity:
        identity_count += 1

    self.assertLess(identity_count, n_estimators, "Categorical permutations should vary.")

  def test_permute_categorical_application(self):
    # Test that transform actually changes the data
    # X has 2 samples, 1 cat feature with values 0 and 1.
    X_enc = np.array([[0, 10.0], [1, 20.0]])
    y = np.array([0, 1])
    cat_features = [0]

    # Force a specific permutation logic by mocking or just checking output
    # We rely on random_state for reproducibility
    generator = EnsembleGenerator(
        n_estimators=5,
        norm_methods=["none"],
        cat_features=cat_features,
        permute_categorical=True,
        feat_shuffle_method="none", # Disable feature shuffling to isolate value permutation
        random_state=42
    )

    generator.fit(X_enc, y)
    data = generator.transform(X_enc)

    # Extract the transformed batch for "none" normalization
    X_out, _ = data["none"]
    # X_out shape: (n_estimators, n_samples, n_features)

    # We expect some estimators to swap 0 and 1 in the first column
    # Since PreprocessingPipeline with "none" still does StandardScaler,
    # the values won't be exactly 0 and 1, but they will be distinct.
    # However, if 0->1 and 1->0 (swap), the resulting standardized values
    # should effectively flip signs (if mean centered) or at least change.

    # Let's inspect the raw permutations to know what to expect
    perms = generator.cat_permutations_["none"]

    # Find an estimator that swaps 0 and 1
    swap_idx = -1
    for i, perm in enumerate(perms):
      mapping = perm[0]
      if mapping.get(0) == 1 and mapping.get(1) == 0:
        swap_idx = i
        break

    if swap_idx != -1:
      # Compare with an estimator that (hopefully) didn't swap or at least is different
      # Actually, let's just compare X_out[swap_idx] vs input logic.
      # Input col 0: [0, 1]
      # Swapped col 0: [1, 0]
      # Standard Scaler on [0, 1] -> [-1, 1] (roughly)
      # Standard Scaler on [1, 0] -> [1, -1]
      # So the output values should be inverted relative to each other?
      # WAIT: The StandardScaler is fitted on the permuted TRAINING data.
      # If we permute train and test consistently, the distribution stats might
      # remain similar (since it's just relabeling), but the *instances* change values.
      # 0 becomes 1.

      # Let's verify that X_out[swap_idx, 0, 0] (Sample 0, Feat 0)
      # is different from X_out[identity_idx, 0, 0] if we find an identity one.

      # Easier check: In the swapped estimator, Sample 0 (was 0->1) should look like Sample 1 (was 1)
      # from a non-swapped estimator? No, Sample 1 in non-swapped is 1. Sample 0 in swapped is 1.
      # So yes, X_out[swap_idx, 0, 0] should be close to X_out[non_swapped, 1, 0].
      pass

    # Simply asserting that outputs are not all identical across estimators for col 0
    col0_values = X_out[:, 0, 0] # (n_estimators,)
    self.assertTrue(np.std(col0_values) > 1e-6, "Categorical values should vary across estimators due to permutation")

    # Verify col 1 (numerical) does NOT vary (feat_shuffle="none")
    # Actually PreprocessingPipeline adds noise? No, only RTDLQuantile does.
    # CustomStandardScaler is deterministic.
    # So col 1 should be identical across estimators.
    col1_values = X_out[:, 0, 1]
    self.assertTrue(np.std(col1_values) < 1e-6, "Numerical values should not vary if shuffling is off")

  def test_permute_categorical_false(self):
    X_enc = np.array([[0, 10.0], [1, 20.0], [2, 30.0]])
    y = np.array([0, 1, 0])
    cat_features = [0]

    generator = EnsembleGenerator(
        n_estimators=5,
        norm_methods=["none"],
        cat_features=cat_features,
        permute_categorical=False, # DISABLED
        feat_shuffle_method="none",
        random_state=42
    )

    generator.fit(X_enc, y)

    # Check permutations are None
    perms = generator.cat_permutations_["none"]
    for perm in perms:
      self.assertIsNone(perm)

    data = generator.transform(X_enc)
    X_out, _ = data["none"]

    # Check that outputs are identical across estimators for col 0
    col0_values = X_out[:, 0, 0]
    self.assertTrue(np.std(col0_values) < 1e-6, "Categorical values should be identical if permutation is disabled")


class BatchForwardTest(absltest.TestCase):

  def test_classifier_batch_forward(self):
    rngs = nnx.Rngs(0)
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=3,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=rngs,
    )
    config = argparse.Namespace()
    config.batch_size = 2

    classifier = TabFMClassifier(
        model=model, config=config, n_estimators=4, batch_size=2
    )
    classifier.n_classes_ = 3
    classifier.classes_ = np.array([0, 1, 2])

    # Generate dummy input arrays.
    # Xs shape: (n_estimators, max_seq_len, num_features)
    Xs = np.random.rand(4, 10, 5)
    # ys shape: (n_estimators, train_size)
    ys = np.random.randint(0, 3, size=(4, 6))

    # Run _batch_forward. Uses data-parallel JAX sharding internally.
    outputs = classifier._batch_forward(Xs, ys)

    # After concatenation, output shape should be (n_estimators, test_size, num_classes).
    # Since Xs length is 10 and ys length (train_size) is 6, test_size is 4.
    self.assertEqual(outputs.shape, (4, 4, 3))

  def test_regressor_batch_forward_rmse(self):
    rngs = nnx.Rngs(0)
    model = tabfm_model.TabFM(
        loss="rmse",
        max_classes=10,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=rngs,
    )
    config = argparse.Namespace()
    config.batch_size = 2
    config.loss = "rmse"

    regressor = TabFMRegressor(model=model, config=config, n_estimators=4)

    # Generate dummy input arrays.
    Xs = np.random.rand(4, 10, 5)
    # ys shape: (n_estimators, train_size)
    ys = np.random.rand(4, 6)

    # Run _batch_forward. Uses data-parallel JAX sharding internally.
    outputs = regressor._batch_forward(Xs, ys)

    # After concatenation, output shape should be (n_estimators, test_size, out_dim).
    # Since Xs length is 10 and ys length (train_size) is 6, test_size is 4.
    # TabFM with rmse outputs 1 value per prediction.
    self.assertEqual(outputs.shape, (4, 4, 1))

  def test_regressor_batch_forward_cross_entropy(self):
    rngs = nnx.Rngs(0)
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=10,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=rngs,
    )
    config = argparse.Namespace()
    config.batch_size = 2
    config.loss = "cross_entropy"

    regressor = TabFMRegressor(model=model, config=config, n_estimators=4)

    # Generate dummy input arrays.
    Xs = np.random.rand(4, 10, 5)
    # ys shape: (n_estimators, train_size)
    ys = np.random.rand(4, 6)

    # Run _batch_forward. Uses data-parallel JAX sharding internally.
    outputs = regressor._batch_forward(Xs, ys)

    # After concatenation, output shape should be (n_estimators, test_size, out_dim).
    # TabFM with cross_entropy outputs max_classes bins.
    self.assertEqual(outputs.shape, (4, 4, 10))


if __name__ == "__main__":
  absltest.main()
