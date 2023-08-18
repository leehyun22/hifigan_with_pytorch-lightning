# Hifigan_with_Pytorch-lightning

이 레포는 pytorch-lightning 프레임 워크를 사용해서, hifigan 모델(vocoder model)을 훈련할 수 있는 코드입니다.

config 파일을 수정해서, 쉽게 훈련을 시킬 수 있습니다.

config 파일을 모듈로 세분화해서 관리하기 위해 [Hydra](https://hydra.cc/docs/intro/)를 사용했습니다.

모델 코드는 [hifi-gan](https://github.com/jik876/hifi-gan)의 코드를 참고해서 작성했습니다.

[hifi-gan 모델](https://arxiv.org/abs/2010.05646)은 기본적으로 Acoustic 모델에서 얻은 음성 feature를 받아서 waveform 으로 생성하는 Vocoder 모델입니다.

데이터는 12시간의 한국어 데이터 [KSS dataset(korean-single-speaker-speech-dataset)](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset)를 사용했습니다.



## 설치 & 훈련

1. 먼저 이 저장소를 클론합니다.

```bash
git clone https://github.com/leehyun22/hifigan_with_pytorch-lightning.git

```
2. python package를 설치합니다.

```bash
pip install -r requirements.txt
```
3. kss 데이터를 path에 준비합니다.

4. config값을 원하는 값으로 수정합니다.

5. train 코드 실행해서 훈련을 시작합니다.

```bash
python3 ./src/train.py
```

## 추론 & 결과



## 참조
https://github.com/jik876/hifi-gan

https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset
