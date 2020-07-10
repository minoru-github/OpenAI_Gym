# お題  
CartPole  
https://gym.openai.com/envs/CartPole-v1/  

## 観測空間、行動空間
| Observation<br>Space | Action<br>Space |
| ---- | ---- |
|  Box(4,)  |  Discrete(2)  |

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

https://github.com/openai/gym/wiki/Table-of-environments  
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

### 記録残し
2020.07.07 DQN動いたー！  
横軸：エピソード　縦軸：そのエピソードで得た累計報酬
![image](https://user-images.githubusercontent.com/18751045/86616280-91f48980-bff0-11ea-80c6-a0bcc7222c4a.png)
