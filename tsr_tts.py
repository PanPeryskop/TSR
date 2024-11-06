import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
import simpleaudio as sa


class TSR_TTS:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-multilingual").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-multilingual")
        self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.output_path = 'audio/audio.wav'
        self.description = ("Garry's is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.")
        self.temp_dir = 'audio/temp'
        os.makedirs(self.temp_dir, exist_ok=True)

    def tsr_tts(self, message: str):
        self.__get_file(message)
        self.__reader()

    def __get_file(self, message: str):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

        input_ids = self.description_tokenizer(self.description, return_tensors="pt").input_ids.to(self.device)
        attention_mask = self.description_tokenizer(self.description, return_tensors="pt").attention_mask.to(self.device)
        prompt_input_ids = self.tokenizer(message, return_tensors="pt").input_ids.to(self.device)
        prompt_attention_mask = self.tokenizer(message, return_tensors="pt").attention_mask.to(self.device)

        generation = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask
        )
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(self.output_path, audio_arr, self.model.config.sampling_rate)

    def __reader(self):
        if os.path.exists(self.output_path):
            wave_obj = sa.WaveObject.from_wave_file(self.output_path)
            play_obj = wave_obj.play()
            # play_obj.wait_done()