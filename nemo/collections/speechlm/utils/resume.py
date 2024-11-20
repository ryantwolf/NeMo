# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from nemo.lightning import io
from nemo.lightning.pytorch.callbacks.peft import ADAPTER_META_FILENAME
from nemo.lightning.resume import AdapterPath, AutoResume
from nemo.utils.app_state import AppState


@dataclass(kw_only=True)
class SLMAutoResume(AutoResume):
    """Dataclass for SLMAutoResume."""

    def get_trainer_ckpt_path(self, model: Optional[io.ConnectorMixin] = None) -> Optional[Path]:
        if self.resume_from_path:
            maybe_weights_path = self.get_weights_path(self.resume_from_path)
            return maybe_weights_path if maybe_weights_path.is_dir() else self.resume_from_path

        checkpoint = None
        app_state = AppState()
        app_state.restore = self.resume_if_exists
        if self.resume_if_exists:
            checkpoint = self._find_trainer_ckpt_path()

        if checkpoint:
            maybe_weights_path = self.get_weights_path(checkpoint)
            if maybe_weights_path.is_dir():
                checkpoint = maybe_weights_path

        if checkpoint:
            if self.adapter_path:
                return AdapterPath(Path(self.adapter_path), base_model_path=checkpoint)
            else:
                adapter_meta_path = checkpoint / ADAPTER_META_FILENAME
                if adapter_meta_path.exists():
                    return AdapterPath(checkpoint, base_model_path=checkpoint)
                else:
                    return Path(checkpoint)

        return None
