# modu_2023
모두의 말뭉치 이야기 완성 과제

팀명 : 엘리

데이터 셋은 해당 github repository에 포함되어 있지 않습니다.

## 환경 셋팅
```
CUDA-12.0
conda create -n <env_name> python=3.9.18
conda activate <env_name>
```

## 라이브러리 설치
```
git clone <repository>
cd modu_2023

pip install -r requirement.txt

git clone https://github.com/huggingface/peft.git
cd peft
python setup.py install
cd ..
```

## Quick Start
```
sh train_run.sh
sh inference_run.sh
sh ensemble_run.sh
```
