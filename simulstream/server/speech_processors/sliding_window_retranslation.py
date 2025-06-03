# Copyright 2025 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from types import SimpleNamespace
from typing import List

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.base import BaseSpeechProcessor, IncrementalOutput


class HFSlidingWindowRetranslator(BaseSpeechProcessor):

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        lang_tags = None
        if hasattr(config, "supported_langs") and config.supported_langs is not None:
            lang_tags = [config.lang_tag_template.format(lang) for lang in config.supported_langs]
        cls.processor = AutoProcessor.from_pretrained(
            config.hf_model_name,
            additional_special_tokens=lang_tags)
        cls.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            config.hf_model_name, trust_remote_code=True)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model.to(cls.device)

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.window_len = self.config.window_len * 100  # 10 ms for each frame

    def _generate(self, speech: torch.Tensor) -> List[str]:
        extra_kwargs = {}
        if self.lang_tag_id is not None:
            extra_kwargs["forced_bos_token_id"] = self.lang_tag_id
        generated_ids = self.model.generate(speech, **extra_kwargs)[0]
        return self.processor.tokenizer.convert_ids_to_tokens(
            generated_ids, skip_special_tokens=True)

    def _preprocess(self, waveform: np.float32) -> torch.Tensor:
        """
        Extracts the filter-bank features from the input waveform and appends them to the audio
        history. Returns the concatenated audio history and new frames, taking the last
        `self.window_len` frames, and returns it after storing it in the audio history.
        """
        # as we already have int16, while HF feature extractor assumes floats, we need to divide
        # by 2**15 the waveform
        new_speech = self.processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt")["input_features"]
        new_speech.to(self.device)
        if self.audio_history is not None:
            new_speech = torch.concat([self.audio_history, new_speech])
        new_speech_len = new_speech.shape[0]
        if new_speech_len > self.window_len:
            new_speech = new_speech[-self.window_len:]
        self.audio_history = new_speech
        return new_speech

    def _build_incremental_outputs(self, generated_tokens: List[str]) -> IncrementalOutput:
        if self.text_history is None:
            return IncrementalOutput(
                new_tokens=generated_tokens,
                new_string=self.processor.tokenizer.convert_tokens_to_string(generated_tokens),
                deleted_tokens=[],
                deleted_string=""
            )
        raise NotImplementedError("todo: dedup")

    def set_language(self, language: str) -> None:
        lang_tag_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.config.lang_tag_template.format(language))
        self.lang_tag_id = torch.tensor(lang_tag_id, dtype=torch.int, device=self.device)

    def _update_speech_history(
            self,
            new_speech: torch.Tensor,
            generated_tokens: List[str],
            new_output: IncrementalOutput) -> None:
        pass

    def _update_text_history(
            self,
            new_speech: torch.Tensor,
            generated_tokens: List[str],
            new_output: IncrementalOutput) -> None:
        self.text_history = generated_tokens

    def clear(self) -> None:
        self.text_history = None
        self.audio_history = None
        self.lang_tag_id = None
