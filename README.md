# Generative Adversarial Networks(GANs)기반의 헤어스타일 예측 소프트웨어 
## Project for graduation
### Pusan National University Computer Science Engineering
### Team No. : 43
### Team Name : 정컴 F3  
---------------------------------------------------------------------
## Contents
- [Architecture](@Architecture)  
  - [Face Detection](@Face-Detection)  
- [Recommending Hair Style](@Recommending-Hair-Style)  
  - [Pre-trained Models](@Pre-trained-Models)  
  - [Preview](@Preview)  
  
## Architecture  
![스크린샷 2021-01-15 오전 1 17 41](https://user-images.githubusercontent.com/48707020/104618927-a799e880-56d0-11eb-95df-05d84edb7a69.png)  

### Face Detection  
![그림1](https://user-images.githubusercontent.com/48707020/104619371-255df400-56d1-11eb-915d-9e8bb48fea4f.png) ![그림2](https://user-images.githubusercontent.com/48707020/104619386-2858e480-56d1-11eb-8318-ac8ee95a9e8c.png)  
- Target Image에 Face feature의 좌표를 구한 후 좌표를 바탕으로 Image를 crop한다.
- Crop Image를 적절한 가공(BGRA 채널로 변환, 투명 배경과 AND 연산 등)을 하여 feature를 담은 mask를 제외한 배경을 투명하게 만든다.
- RGB채널에 합성하기 위해 BGRA의 alpha 채널 수 만큼 loop를 돌려 합성해준다.

## Recommending Hair Style
### Pre-trained Models  
> [face_256x256](https://drive.google.com/open?id=1MTeDchdtcvTWWQAtYvFHKIJLiuvtq49k) In-domain GAN trained with [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset.  
### Preview
![ezgif com-gif-maker](https://user-images.githubusercontent.com/48707020/104618941-ab2d6f80-56d0-11eb-94ef-12dcfd3e865b.gif) ![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/48707020/104618951-acf73300-56d0-11eb-9273-753be65b392c.gif)
