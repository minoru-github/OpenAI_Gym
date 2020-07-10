# お題  
CartPole  
https://gym.openai.com/envs/CartPole-v1/  

img_states = env.render(mode='rgb_array')

## 観測空間、行動空間
| Observation<br>Space | Action<br>Space |
| ---- | ---- |
| numpy.ndarray<br>(400, 600, 3) |  Discrete(2)  |

    Observation:  
![image](https://user-images.githubusercontent.com/18751045/87165488-30406200-c305-11ea-9f92-2019df4bd0e1.png)  

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

https://github.com/openai/gym/wiki/Table-of-environments  
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

### 記録残し
画像で強化学習にトライ。  
CartPoleを画像で強化学習の事例みたことないからちゃんとできるか不明。
