**环境配置**

0. 安装conda；运行miniconda3/bin中的activate；创建conda环境

    ```shell
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/miniconda3/bin/activate
    conda create -n vb python==3.11
    conda activate vb
    ```

<!-- 或者升级conda：

    ```shell
    conda activate base
    conda update conda
    conda activate vb
    ``` -->

1. 在conda环境中

    ```shell
    pip install uv
    ```

2. 解决 PyAV/av 构建报错（uv sync 时 av 从源码编译，需要以下依赖）

    ```shell
    conda install -c conda-forge 'ffmpeg>=7'
    conda install -c conda-forge pkg-config cython c-compiler
    ```

    如果没有repo：

        ```shell
        cd ~
        git clone https://github.com/chaoyiL/openpi_chaoyi.git
        ```

3. 配置环境

    ```shell
    cd ~/openpi_chaoyi
    uv sync
    uv pip install -e .
    ```

    注：实验室的电脑似乎没法直接在conda里跑 uv sync，需要强制使用系统头文件再跑 uv sync和uv pip install -e .

    ```shell
    CC=/usr/bin/gcc CFLAGS="-I/usr/include" uv sync
    CC=/usr/bin/gcc CFLAGS="-I/usr/include" uv pip install -e .
    ```

4. 下载文件
    ```shell
    gdown "https://drive.google.com/drive/folders/1YP4H8W_4Cp12oZy38Am3YABi0s4k98Ib" -O ./data --folder
    ```
    其中，网址替换成自己的数据

**Finetune 步骤**

0. 修改使用的数据集：config.py 第564行 data_name 改为需要用的数据集名称.

    如需后台运行，使用如下指令：

    a. 新建tmux终端：
    ```shell
    tmux new
    ```

    b. 连接到tmux终端：
    ```shell
    tmux attach
    ```

    c. 显示当前终端有哪些：
    ```shell
    tmux ls
    ```

1. 数据格式转换

```shell
bash scripts/run_convert.sh
```

注：_0118数据集现已完成转换，上传到了lerobot repo中。需要使用时只需将repo内容下载到 ~/.cache/huggingface/lerobot/chaoyi/_0118 即可。

2. 计算归一化统计量

```shell
bash scripts/compute_norm_stats.sh pi05_chaoyi_vitac
```
（用uv锁定所有库的版本，避免冲突）
其中，sh文件中pi05_chaoyi需要被修改为对应的config名

3. 配置wandb

```shell
wandb login
```

4. 训练

```shell
bash scripts/train.sh pi05_chaoyi_vitac
```

训练时使用的卡数在 config.py 中修改，由参数fsdp_devices决定。

    a. 如果直接运行后报错“没有从GCS下载的权限”，则直接下载文件 paligemma_tokenizer.lock 和 paligemma_tokenizer.model 到服务器路径 ~/.cache/openpi/big_vision 中去:

    ```shell
    rsync -avP -e "ssh -p [服务器的port] -i ~/.ssh/id_ed25519" ~/.cache/openpi/big_vision/paligemma_tokenizer.model root@[服务器的ip]:~/.cache/openpi/big_vision/
    ```
    port, ip都要改成服务器的。

    b. 训练时使用的卡数在 config.py 中修改，由参数fsdp_devices决定。

5. 下载ckpt

```shell
scp -P 58104 -i ~/.ssh/id_ed25519 -r root@195.26.233.55:~/openpi_chaoyi/checkpoints/pi05_chaoyi_vitac/my_experiment /home/rvsa/codehub/VB-VLA/checkpoints/pi05_chaoyi_vitac
# 或者下面的指令，需要在服务器上也下载 rsync
rsync -avz --progress -e "ssh -p 58104 -i ~/.ssh/id_ed25519" root@195.26.233.55:~/openpi_chaoyi/checkpoints/pi05_chaoyi_vitac/my_experiment /home/rvsa/codehub/VB-VLA/checkpoints/pi05_chaoyi_vitac/
```
-P替换端口，root@194.68.245.213替换SSH over exposed TCP对应的账户和ip，后面的地址替换成自己的目标文件夹

6. 单步推理测试，给出一个随机输入，输出action chunk

```shell
bash ./scripts/test_single_inf.sh --config pi05_chaoyi_vitac --ckpt-dir checkpoints/pi05_chaoyi_vitac/my_experiment/5000
```

action chunk 的长度可以在 config.py 中通过 action_horizon 参数修改

**Deploy 步骤**

1. JOG脚本

```shell
bash ./scripts/joint_jog.sh
```

2. deploy脚本

```shell
bash ./scripts/eval_real_bimanual_vb.sh
bash scripts/bimanual_vb_TE.sh
```

**TODO**

1. **数据集**（已完成）：包括state（observation.state, 自感知）和action（action），我们需要把state和action从joint改成tcp。

    state：1. 相对初始位置的位姿（pos + rot_vec, 6d * 2） 2. 夹爪距离（1d * 2） 3. 左夹爪相对右夹爪的位姿（pos + rot_vec, 6d），一共20d

    action：1. 末端执行器的位姿变化量（pos + 旋转矩阵前两列， 9d * 2）2. 夹爪距离**绝对量**（1d * 2）

2. **修改policy**（已完成）：写一个新的vb_policy，key和维度与我们自己的features符合

3. **修改网络**（已完成）：将基于图像的触觉信号加入

4. **单步inference**（已完成）：将训练好的ckpt存储，并尝试用其推断

5. **实机deploy**：

    a. 将env读取出来的实际obs与policy需要的dict项一一对应，生成obs_dict。*这一步容易出错，需要检查方法。*（已完成）

    b. 将obs_dict输入policy，获取policy的raw_action

    c. 将raw_action输入get_real_umi_action()函数，获得action（不用改）

*注：*

1. config中，state和action的维度默认相同（pi0_config.py第71行），但可以修改. 本库中已经将state_dim与action_dim解耦

2. train.py中，原先的action为32维，需要在_load_weights_and_validate函数中添加自动适配action_dim的代码（为何load出来的action dim会和load之前有不同？？）

3. model.py中，原先的tuple IMAGE_KEYs与vb_policy中设定的图像keys不同，需要修改为对应名称。**现已将不同模式的keys封装，只需在model文件中修改policy_type即可。**

4. 图像增强：在model.py第184行，原定的图像增强不对腕部相机进行（即wrist是否存在于key中），而只对外部相机进行。目前修改为对所有图像进行增强

5. 设置ckpt存储频率：在 config.py 中的 class TrainingConfig，有变量 save_interval，用来设置多少代存储一次ckpt

6. Git目前已经设置全局代理