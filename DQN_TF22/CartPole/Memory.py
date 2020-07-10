import random

# stored as (s(t), a(t), r(t), s(t+1))
class Memory(): 
    # Constructor
    def __init__(self, max_size):
        self.max_size = max_size
        self.samples = []
        self.is_full_flag = False # 外から参照できないようにしたい。。。
    # Add experience to memory
    def add(self, experience):
        self.samples.append(experience)
        if len(self.samples) > self.max_size:
            self.samples.pop(0) # メモリーの最大サイズを超えたら先頭を削除
            self.is_full_flag = True
    # Sample batch_size experiences from memory 
    def sample(self, batch_size):
        n = min(batch_size, len(self.samples))
        return(random.sample(self.samples, n))
    # Memoryが満タンかどうかの確認
    def is_full(self):
        return self.is_full_flag
