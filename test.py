from hubert import HubertForCTC
from datasets import load_dataset
from transformers import Wav2Vec2Processor , AutoFeatureExtractor , Wav2Vec2FeatureExtractor


dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True)
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

input_values= processor(dataset[0]["audio"]["array"], return_tensors="pt").input_values

 