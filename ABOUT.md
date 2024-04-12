# HierSpeech++

## Введение
Во многих проектах, связанных с чат-ботами, может требоваться реализация воспроизведения текста конкретным голосом.
Более точно эта задача называется TTS (Text To Speech).
Для того, чтобы такая модель "заговорила" конкретным голосом, нам нужно либо найти обычную TTS модель и зафайнтюнить ее на датасете с этим голосом, либо прибегнуть к Zero-shot моделям. Их преимущество заключается в том, что нам не нужно дообучать модель, а можно просто подать небольшой аудиофрагмент с нужным голосом и текст, который необходимо озвучить. Собственно, здесь и далее мы будем рассматривать одну из лучших таких моделей, как она устроена и как ей пользоваться

## Архитектура
![Fig1_pipeline](https://github.com/sh-lee-prml/HierSpeechpp/assets/56749640/8f0b5f24-8491-4908-ae06-e0dfcc7d9e52)

## Система
- Python3.10
- Linux (tested on Ubuntu 22.04)
- CUDA 11.8, CuDNN

## Установка
```bash
git clone git@github.com:sh-lee-prml/HierSpeechpp.git
cd HierSpeechpp/
pip install -r requirements.txt
mkdir logs
cd logs
gdown --folder https://drive.google.com/drive/folders/1sFQP-8iS8z9ofCkE7szXNM_JEy4nKg41
gdown --folder https://drive.google.com/drive/folders/1QiFFdPhqhiLFo8VXc0x7cFHKXArx7Xza
cd ..
```

## Использование
```python
from inference import TTSModel

tts = TTSModel('launch.json')

text = 'Simple example for dubbing'
tts.invoke(text)
```

Пример конфигурационного файла для запуска инференса (комментариев быть не должно):
```json
{
    "input_prompt": "example/record.wav", # файл с записью оригинального голоса
    "input_txt": "example/reference_4.txt", # текст для озвучки в файле, хотя можно и из кода
    "output_dir": "output", # папка для сохранения синтезированного аудио
    "ckpt": "./logs/hierspeechpp_libritts960/hierspeechpp_lt960_ckpt.pth",
    "ckpt_text2w2v": "./logs/ttv_libritts_v1/ttv_lt960_ckpt.pth",
    "ckpt_sr": "./speechsr24k/G_340000.pth",
    "ckpt_sr48": "./speechsr48k/G_100000.pth",
    "denoiser_ckpt": "denoiser/g_best",
    "scale_norm": "max", # предположительно, нормализация громкости
    "output_sr": 48000, # 16k (без super resolution), 24k, 48k (Default)
    "noise_scale_ttv": 0.333, # параметр сохранения шума. Для надежности - 0.333, для выразительности - 0.666
    "noise_scale_vc": 0.333, # то же самое, но для voice conversion
    "denoise_ratio": 0.8 # 0 - не убирать шум, 1 - убирать. Баланс - между 0.6-0.8
}
```

## Заметки по использованию
- Модель пока только для английского, на другие языки нужно обучать отдельно
- Сколько ни пытайся говорить на ломаном английском, в итоге слова будут произноситься идеально => не факт, что точно передается манера речи
- Текст можно указать сколько угодной длины
- Аудиофайл больше ~10с роняет программу
- Время инференса (без инициализации) на 3070ti 8Gb - 4-10с (в зависимости от длины текста)