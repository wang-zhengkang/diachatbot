# tmux
## 创建新会话

```tmux new -s session_name```

## 重新连接会话

```tmux ls```

```tmux a -t session_name```

## 创建会话中的新窗口

``` 第一步：按 Ctrl+B 组合键，然后松开。```

``` 第二步：再单独按一下 c 键。```

## 切换窗口

```假如要切换到 1：bash 这个窗口，步骤如下：```

```第一步：按 Ctrl-B 组合键，然后松开。```

```第二步：按数字 1 键。```

## 离开电脑

第一步：输入组合键 Ctrl+B，然后松开。

第二步：输入字母 d。

## 杀死某个会话：

```tmux kill-session -t {session-name}```

# Show_GPU_State

```watch -n 1 nvidia-smi```

# multiGPU_train

## 1. 在文件import部分之后加入
```python
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 设置主卡
    DEVICES = [0, 1]  # 设置多张GPU
```
## 2. 将模型放入GPU
```python
self.user = VHUS(config, voc_goal_size, voc_usr_size, voc_sys_size)
self.user = nn.DataParallel(self.user, device_ids=DEVICES).to(device)
```
## 3. 将数据放入GPU
```python
data = data.to(device=device)
```
