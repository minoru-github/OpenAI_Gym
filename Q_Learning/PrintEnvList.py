from gym import envs

env_list = envs.registry.all()
ids = [element.id for element in env_list]
[print(id) for id in ids]