# お題  
Breakout  
https://gym.openai.com/envs/Breakout-v0/  

## 観測空間、行動空間
| Observation<br>Space | Action<br>Space |
| ---- | ---- |
| numpy.ndarray<br>(210, 160, 3) |  Discrete(4)  |

    Observation:  
        Box(210, 160, 3)
![image](https://user-images.githubusercontent.com/18751045/89043441-724c4900-d383-11ea-8ea4-8bcfbbdb9335.png)

    Actions:
        0: "NOOP",
        1: "FIRE",
        2: "UP",
        3: "RIGHT",

https://github.com/openai/gym/wiki/Table-of-environments  
