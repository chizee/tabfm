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

import os
import threading
from typing import Optional
from absl import logging

from tabfm.src.pytorch.model import TabFM

HF_REPO_ID = "google/tabfm-1.0.0-pytorch"

_LOAD_CACHE_LOCK = threading.Lock()
_LOAD_CACHE = {}


def load(
    model_type: str = "classification",
    checkpoint_path: Optional[str] = None,
    *,
    device: Optional[str] = None,
    use_cache: bool = True,
) -> TabFM:
  """Loads the PyTorch TabFM v1.0.0 model with pre-trained weights.

  Args:
    model_type: 'classification' or 'regression'.
    checkpoint_path: Local directory or weights file. If None, downloads from
      Hugging Face (google/tabfm-1.0.0-pytorch).
    device: Target device (e.g. 'cuda', 'cpu'). Defaults to 'cpu'.
    use_cache: Reuse a process-wide cached model for identical settings.

  Returns:
    An eval-mode TabFM model with pre-trained weights loaded.
  """
  if model_type not in ("classification", "regression"):
    raise ValueError(
        f"Unsupported model_type: {model_type!r}. "
        "Must be 'classification' or 'regression'."
    )

  cache_key = (model_type, checkpoint_path, device)
  if use_cache:
    _LOAD_CACHE_LOCK.acquire()
  try:
    if use_cache and cache_key in _LOAD_CACHE:
      return _LOAD_CACHE[cache_key]

    if checkpoint_path is None:
      logging.info(
          "Downloading TabFM v1.0.0 PyTorch %s weights from Hugging Face...",
          model_type,
      )
      model = TabFM.from_pretrained(HF_REPO_ID, subfolder=model_type)
    else:
      local_dir = checkpoint_path
      if os.path.isdir(local_dir):
        sub = os.path.join(local_dir, model_type)
        if os.path.isdir(sub):
          local_dir = sub

      if os.path.isdir(local_dir) and os.path.exists(
          os.path.join(local_dir, "config.json")
      ):
        model = TabFM.from_pretrained(local_dir)
      else:
        # no config.json: pass is_classifier explicitly
        model = TabFM.from_pretrained(
            local_dir,
            is_classifier=(model_type == "classification"),
        )

    if device is not None:
      model = model.to(device)
    model.eval()

    if use_cache:
      _LOAD_CACHE[cache_key] = model
    return model
  finally:
    if use_cache:
      _LOAD_CACHE_LOCK.release()
