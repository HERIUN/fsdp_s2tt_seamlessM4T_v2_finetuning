# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.


import logging
import os
from abc import abstractmethod
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from datasets import load_dataset

from .datatypes import LangPairSample, MultimodalSample
import json
import soundfile as sf
import torchaudio

logger = logging.getLogger(__name__)


class SpeechTokenizer:
    @abstractmethod
    def encode(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        ...


class Speech2SpeechFleursDatasetBuilder:
    """Assembles speech2speech dataset from google/fleurs on HuggingFace"""

    DATASET_NAME = "google/fleurs"

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        split: str = "test",
        skip_source_audio: bool = True,
        skip_target_audio: bool = True,
        audio_dtype: torch.dtype = torch.float32,
        dataset_cache_dir: Optional[str] = None,
        speech_tokenizer: Optional[SpeechTokenizer] = None,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.split = split
        self.dataset_cache_dir = dataset_cache_dir
        self.audio_dtype = audio_dtype
        self.skip_source_audio = skip_source_audio
        self.skip_target_audio = skip_target_audio
        self.speech_tokenizer = speech_tokenizer

    def _prepare_sample(
        self,
        sample_id: int,
        lang: str,
        text: str,
        audio_local_path: Optional[str] = None,
        waveform_npy: Optional[np.ndarray] = None,
        sampling_rate: Optional[int] = None,
    ) -> MultimodalSample:
        should_skip_audio = (
            lang == self.target_lang
            and self.skip_target_audio
            or lang == self.source_lang
            and self.skip_source_audio
            or waveform_npy is None
        )
        if not should_skip_audio:
            waveform = torch.from_numpy(waveform_npy).to(self.audio_dtype)
        else:
            waveform = None
        if self.speech_tokenizer is not None and not should_skip_audio:
            assert waveform is not None
            assert sampling_rate is not None
            units_tensor = self.speech_tokenizer.encode(
                waveform, sampling_rate
            ).reshape(-1)
            units = units_tensor.tolist()
        else:
            units = None
        return MultimodalSample(
            id=sample_id,
            lang=lang,
            text=text.strip(),
            audio_local_path=audio_local_path,
            waveform=waveform,
            sampling_rate=sampling_rate,
            units=units,
        )

    def iterate_lang_audio_samples(self, lang: str) -> Iterable[MultimodalSample]:
        ds = load_dataset(
            self.DATASET_NAME,
            lang,
            split=self.split,
            cache_dir=self.dataset_cache_dir,
            streaming=False,
            trust_remote_code=True,
        )
        for item in ds:
            audio_path = os.path.join(
                os.path.dirname(item["path"]), item["audio"]["path"]
            )
            (sample_id, audio_local_path, waveform, sampling_rate, text) = (
                item["id"],
                audio_path,
                item["audio"]["array"],
                item["audio"]["sampling_rate"],
                item["transcription"],
            )
            yield self._prepare_sample(
                sample_id=sample_id,
                audio_local_path=audio_local_path,
                waveform_npy=waveform,
                sampling_rate=sampling_rate,
                text=text,
                lang=lang,
            )

    def __iter__(self) -> Iterable[LangPairSample]:
        logger.info(f"Loading {self.target_lang} samples")
        target_samples: Dict[int, MultimodalSample] = {}
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(lang=self.target_lang)
        ):
            if idx and idx % 100 == 0:
                logger.info(f"..loaded {idx} target samples")
            target_samples[sample.id] = sample

        logger.info(f"Loading {self.source_lang} samples")
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(lang=self.source_lang)
        ):
            if idx and idx % 100 == 0:
                logger.info(f"..loaded {idx} source samples")
            if sample.id in target_samples:
                yield LangPairSample(source=sample, target=target_samples[sample.id])


class Speech2TextGigaspeechDatasetBuilder:
    """ Assembles speech2speech dataset from google/fleurs on HuggingFace.
        This dataset requires signing an license agreement and using an auth token.
    """

    DATASET_NAME = "speechcolab/gigaspeech"

    def __init__(
        self,
        auth_token: str,
        split: str = "test",
        skip_source_audio: bool = True,
        skip_target_audio: bool = True,
        audio_dtype: torch.dtype = torch.float32,
        dataset_cache_dir: Optional[str] = None,
        speech_tokenizer: Optional[SpeechTokenizer] = None,
    ):
        self.auth_token = auth_token
        self.split = split
        self.dataset_cache_dir = dataset_cache_dir
        self.audio_dtype = audio_dtype
        self.skip_source_audio = skip_source_audio
        self.skip_target_audio = skip_target_audio
        self.speech_tokenizer = speech_tokenizer

    def _prepare_sample(
        self,
        sample_id: int,
        lang: str,
        text: str,
        audio_local_path: Optional[str] = None,
        waveform_npy: Optional[np.ndarray] = None,
        sampling_rate: Optional[int] = None,
    ) -> MultimodalSample:
        if waveform_npy is not None:
            waveform = torch.from_numpy(waveform_npy).to(self.audio_dtype)
        else:
            waveform = None
        if self.speech_tokenizer is not None and waveform_npy is not None:
            assert waveform is not None
            assert sampling_rate is not None
            units_tensor = self.speech_tokenizer.encode(
                waveform, sampling_rate
            ).reshape(-1)
            units = units_tensor.tolist()
        else:
            units = None
        return MultimodalSample(
            id=sample_id,
            lang=lang,
            text=text.strip(),
            audio_local_path=audio_local_path,
            waveform=waveform,
            sampling_rate=sampling_rate,
            units=units,
        )

    def iterate_lang_audio_samples(self, lang: str) -> Iterable[MultimodalSample]:
        ds = load_dataset(
            self.DATASET_NAME,
            lang,
            split=self.split,
            cache_dir=self.dataset_cache_dir,
            streaming=False,
            trust_remote_code=True,
        )
        for item in ds:
            audio_path = os.path.join(
                os.path.dirname(item["path"]), item["audio"]["path"]
            )
            (sample_id, audio_local_path, waveform, sampling_rate, text) = (
                item["id"],
                audio_path,
                item["audio"]["array"],
                item["audio"]["sampling_rate"],
                item["transcription"],
            )
            yield self._prepare_sample(
                sample_id=sample_id,
                audio_local_path=audio_local_path,
                waveform_npy=waveform,
                sampling_rate=sampling_rate,
                text=text,
                lang=lang,
            )

    def __iter__(self) -> Iterable[LangPairSample]:
        logger.info(f"Loading {self.target_lang} samples")
        target_samples: Dict[int, MultimodalSample] = {}
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(lang=self.target_lang)
        ):
            if idx and idx % 100 == 0:
                logger.info(f"..loaded {idx} target samples")
            target_samples[sample.id] = sample

        logger.info(f"Loading {self.source_lang} samples")
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(lang=self.source_lang)
        ):
            if idx and idx % 100 == 0:
                logger.info(f"..loaded {idx} source samples")
            if sample.id in target_samples:
                yield LangPairSample(source=sample, target=target_samples[sample.id])


class Speech2TextDatasetBuilder:

    def __init__(
            self, 
            json_path, 
            source_lang="ko", 
            target_lang="en",
            skip_source_audio: bool = True,
            skip_target_audio: bool = True,
            speech_tokenizer: Optional[SpeechTokenizer] = None,
            audio_dtype: torch.dtype = torch.float32,
            ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.target_lang = target_lang
        self.source_lang = source_lang
        self.skip_source_audio = skip_source_audio
        self.skip_target_audio = skip_target_audio
        self.speech_tokenizer = speech_tokenizer
        self.audio_dtype = audio_dtype

    def _prepare_sample(
        self,
        sample_id: int,
        lang: str,
        text: str,
        audio_local_path: Optional[str] = None,
        waveform_npy: Optional[np.ndarray] = None,
        sampling_rate: Optional[int] = None,
    ) -> MultimodalSample:
        should_skip_audio = (
            lang == self.target_lang
            and self.skip_target_audio
            or lang == self.source_lang
            and self.skip_source_audio
            or waveform_npy is None
        )
        if not should_skip_audio:
            waveform = torch.from_numpy(waveform_npy).to(self.audio_dtype)
        else:
            waveform = None
        if self.speech_tokenizer is not None and not should_skip_audio:
            assert waveform is not None
            assert sampling_rate is not None
            units_tensor = self.speech_tokenizer.encode(
                waveform, sampling_rate
            ).reshape(-1)
            units = units_tensor.tolist()
        else:
            units = None
        return MultimodalSample(
            id=sample_id,
            lang=lang,
            text=text.strip(),
            audio_local_path=audio_local_path,
            waveform=waveform,
            sampling_rate=sampling_rate,
            units=units,
        )


    def iterate_lang_audio_samples(self, lang: str) -> Iterable[MultimodalSample]:
        for idx,d in enumerate(self.data):
            audio_array, sr =  sf.read(d["audio_path"])
            (sample_id, audio_local_path, waveform, sampling_rate, text) = (
                idx,
                d["audio_path"],
                audio_array,
                sr,
                d[lang],
            )
            yield self._prepare_sample(
                sample_id=sample_id,
                audio_local_path=audio_local_path,
                waveform_npy=waveform,
                sampling_rate=sampling_rate,
                text=text,
                lang=lang,
            )

    def __iter__(self) -> Iterable[LangPairSample]:
        target_samples: Dict[int, MultimodalSample] = {}
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(lang=self.target_lang)
        ):
            if idx and idx % 100 == 0:
                logger.info(f"..loaded {idx} target samples")
            target_samples[sample.id] = sample

        logger.info(f"Loading {self.source_lang} samples")
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(lang=self.source_lang)
        ):
            if idx and idx % 100 == 0:
                logger.info(f"..loaded {idx} source samples")
            if sample.id in target_samples:
                yield LangPairSample(source=sample, target=target_samples[sample.id])


class Speech2TextDatasetHFBuilder:
    """ Assembles speech2speech dataset from HuggingFace.
        This dataset may requires signing an license agreement and using an auth token.
        
        data format(example)
        {
        'audio': {'path': '0_161_ko-GS_DE_4524970_F_20_iPad8-11_TRIM.wav',
            'array': array([ 3.05175781e-04, -4.88281250e-04, -6.71386719e-04, ...,
                    -2.74658203e-04, -1.52587891e-04, -3.05175781e-05]),
        'sampling_rate': 16000},
        'transcription': '경복궁은 한국에서 가장 유명한 궁궐 중 하나입니다.', 
        'description': 'voice_event_182',
        'English': 'Gyeongbokgung Palace is one of the most famous palaces in Korea.' # target_text_column
        }

    """
    def __init__(
        self, 
        dataset_name:str,
        target_text_column:str, 
        source_lang:str,
        target_lang:str,
        save_directory:str,
        split:str=None,
        hf_token:str=None,
        audio_dtype: torch.dtype = torch.float32,
        ):
        self.dataset_name = dataset_name
        self.auth_token = hf_token
        self.split = split
        self.source_lang=source_lang
        self.target_lang=target_lang
        self.target_text_column = target_text_column
        self.audio_dtype = audio_dtype
        self.save_directory = save_directory # where to audio wav saved

        # where to save audio file local
        self.audio_dir = os.path.join(
                self.save_directory,
                self.dataset_name,
                self.split
            )
        os.makedirs(self.audio_dir, exist_ok=True)

    
    def _prepare_sample(
        self,
        sample_id: int,
        lang: str,
        text: str,
        audio_local_path: Optional[str] = None,
        waveform_npy: Optional[np.ndarray] = None,
        sampling_rate: Optional[int] = None,
    ) -> MultimodalSample:
        if waveform_npy is not None and lang==self.source_lang:
            waveform = torch.from_numpy(waveform_npy).to(self.audio_dtype)
            torchaudio.save(audio_local_path, waveform.unsqueeze(0), sampling_rate)
        else:
            waveform = None
        units = None
        return MultimodalSample(
            id=sample_id,
            lang=lang,
            text=text.strip(),
            audio_local_path=audio_local_path if lang==self.source_lang else None,
            waveform=waveform,
            sampling_rate=sampling_rate,
            units=units,
        )


    def iterate_lang_audio_samples(self, lang: str) -> Iterable[MultimodalSample]:
        ds = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=False,
            trust_remote_code=True,
            token = self.auth_token
        )

        for idx, item in enumerate(ds):
            audio_path = os.path.join(
                os.path.dirname(item["path"]), item["audio"]["path"]
            ) if item.get("path", None) is not None else os.path.join(
                self.audio_dir,
                item["audio"]["path"]
            )

            (sample_id, audio_local_path, waveform, sampling_rate, text) = (
                item.get("id",idx),
                audio_path,
                item["audio"]["array"] if lang==self.source_lang else None,
                item["audio"]["sampling_rate"] if lang==self.source_lang else None,
                item["transcription"] if lang==self.source_lang else item[self.target_text_column],
            )
            yield self._prepare_sample(
                sample_id=sample_id,
                audio_local_path=audio_local_path,
                waveform_npy=waveform,
                sampling_rate=sampling_rate,
                text=text,
                lang=lang,
            )

    def __iter__(self) -> Iterable[LangPairSample]:
        target_samples: Dict[int, MultimodalSample] = {}
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(lang=self.target_lang)
        ):
            if idx and idx % 100 == 0:
                logger.info(f"..loaded {idx} target samples")
            target_samples[sample.id] = sample

        logger.info(f"Loading {self.source_lang} samples")
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(lang=self.source_lang)
        ):
            if idx and idx % 100 == 0:
                logger.info(f"..loaded {idx} source samples")
            if sample.id in target_samples:
                yield LangPairSample(source=sample, target=target_samples[sample.id])
