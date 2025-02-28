seamlessM4Tv2 finetuning with custom data with FSDP

0.(optional). make sample s2tt data(n_samples=100) with chatgpt and google tts
```
python make_sample_data.py
```

1. prepare custom json data.
output.json : audio path is source language_speech. 
```json
[
    {
        "kor": "서울은 다양한 역사적 유산과 현대적인 매력을 함께 경험할 수 있는 도시입니다.",
        "eng": "Seoul is a city where you can experience both historical heritage and modern charm.",
        "audio_path": "/data/donggukang/seamless_test/s2tt_data/audio/output_0.wav"
    },
    {
        "kor": "경주는 한국에서 가장 오래된 도시 중 하나로서, 문화재와 유적지가 많이 있습니다.",
        "eng": "Gyeongju is one of the oldest cities in Korea with numerous cultural relics and historical sites.",
        "audio_path": "/data/donggukang/seamless_test/s2tt_data/audio/output_1.wav"
    },
    ...
]
```
or hugging face audio datasets. check src/seamless_communication/cli/m4t/finetune/dataset.py ```load_custom_s2tt_hf_dataset```

2. convert seamless format
```sh
python src/seamless_communication/cli/m4t/finetune/dataset.py \
                --name "custom" \
                --source_lang "kor" \
                --target_lang "eng" \
                --save_dir "/data/donggukang/data" \ # where to save manifest.json
                --json_path "out_json.json" \
```

3. edit finetune.py code
```python
sys.path.append("/data/donggukang/seamless_test/seamless_communication/src")
os.environ["CUDA_VISIBLE_DEVICES"]="0,3,4,7"
```

4. finetune s2tt with fsdp
```sh
python src/seamless_communication/cli/m4t/finetune/finetune.py \
   --mode SPEECH_TO_TEXT \
   --train_dataset /data/donggukang/data/custom_manifest.json  \
   --eval_dataset /data/donggukang/data/custom_manifest.json \
   --batch_size 1 \
   --learning_rate 1e-6 \
   --warmup_steps 100 \
   --max_epochs 10 \
   --patience 5 \
   --model_name seamlessM4T_v2_large \
   --save_model_to ./checkpoint.pt
```
