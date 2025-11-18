> [!NOTE]
> This repo is about a set of commands which might be useful for coding.

<details>
<summary><h2>Docs</h2></summary>
  
[LLama Download](https://www.llama.com/llama-downloads/)

[이전 파이토치 버전, Previous Pytorch version](https://pytorch.kr/get-started/previous-versions/)

[Tensorflow compatiability with CUDA in Korean](https://www.tensorflow.org/install/source?hl=ko#gpu)

[CUDA Toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive)

[Compatiability between CUDA Toolkit and Driver](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)

[cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive)

[nvidia-smi 각 요소에 대한 설명](https://jjuke-brain.tistory.com/entry/GPU-%EC%84%9C%EB%B2%84-%EC%82%AC%EC%9A%A9%EB%B2%95-CUDA-PyTorch-%EB%B2%84%EC%A0%84-%EB%A7%9E%EC%B6%94%EA%B8%B0-%EC%B4%9D%EC%A0%95%EB%A6%AC)

Since this post is written in Korean, I used my native language

[vLLM supported model list](https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation)

[vLLM supported GPU](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#install-the-latest-code-using-pip)  
</details>

<details>
<summary><h2>Pytorch</h2></summary>

<strong>Initial Environmental Setup</strong>

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu<cuda no.>
e.g.,
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

<strong>Allow dynamic memory allocation to prevent from segmentation</strong>

```
# More details in https://pytorch.org/docs/stable/notes/cuda.html
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# alternatives in code
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
```

```
torch.no_grad()

# recommended
torch.inference_mode()
```

Remember that this technique cannot resolve fundamental OOM problems. i.e., You might resolve this issue by replacing the original with bigger gpu such as H200 !!
</details>

<details>
<summary><h2>screen</h2></summary>

0. <strong>Create a session</strong>
```
screen -S <session_name>
```

1. <strong>list screens</strong>
```
screen -ls
```

2. <strong>Re-enter screen</strong>
```
screen -r [session_name or ID]
```

3. <strong>toggle</strong>

| Toggle | explanation |
|----------|----------|
| Ctrl+A, C  | Create new window in a session  |
| Ctrl+A  | Move window |
| Ctrl+A, D or exit | Detach the session(move background)  |

4. <strong>Terminate session</strong>
```
exit
screen -X -S <session_name or PID> quit
pkill screen # stop all sessions

ps aux | grep screen
kill -9 <session id>
```

5. <strong>Activate scroll mode</strong>
```
Ctrl+A, [
```


</details>

<details>
<summary><h2><strong>tmux</strong></h2></summary>

tmux is useful when you have to utilize multiple terminals concurrently without termination.(i.e., efficient terminal usage!)

Multiple sessions can be created via tmux, and it results in more efficient terminal mangement than single one.

0. <strong>Prerequisite</strong>
```
sudo apt-get install tmux
```
1. <strong>Create a session</strong>
```
tmux new -s <sessions_name>
```
2. <strong>Print out information on session</strong>

Information includes in session name, the number of windows, and current attached session
```
tmux ls
```
3. <strong>Enter a session</strong>
```
tmux attach -t <session_name>
```
4. <strong>Detach from a session</strong>

If using this command, you need to the former and the latter separately not concurrently(i.e., 1) ctrl+b, 2) d
```
ctrl+b -> d
```
5. <strong>Create a new window</strong>
```
ctrl+b -> c
```

6. <strong>Prerequisite for previous input command</strong>
```
bash
```

<strong>add scroll in terminal</strong>
```
vi ~/.tmux.conf
set -g mouse on
:wq!
tmux source-file ~/.tmux.conf
```

<strong>panel split</strong>
```
# vertical split (좌우 분할)
ctrl + b => shift 5
# horizontal split (위아래 분할)
ctrl + b => shift + '(between return and : key)
```

<strong>close panel</strong>
```
# close current panel
ctrl + b => x
# close all panels
ctrl + b => &
```
</details>

<details>
<summary><h2><strong>Multi-GPUs</strong></h2></summary>

<strong> Monitor GPU status in regular </strong>
```
watch -n 1 nvidia-smi 
```

1. <strong>Check current storage capacity</strong>
```
df -h
```
2. <strong>print current working directory</strong>
```
pwd
```
3. <strong>Check your CUDA version</strong>
```
nvcc -V
```
4. <strong>Output current directory's capacity</strong>
```
du -sh .
du -sh *
```

Sort by hidden files/directories
```
du -sh .[!.]* * | sort -h
du -sh .[!.]* * | sort -hr # sorted by descending order
```

4-1. <strong>Identify specific directory's capacity</strong>
```
du -h --max-depth=1 <dir_path> | sort -hr
```
5. <strong>Check all running processes' id and command related to python</strong>
```
nvidia-smi | grep python | awk '{print $5}' | xargs -I{} ps -p {} -o pid,cmd
```
```
ps aux | grep python
```
6. <strong>Print list of currently running prcoesses on background</strong>
```
jobs
```
7. <strong>Monitor resource usage and running processs</strong>
```
top
```
8. <strong>Similar to top, but better in terms of visualization</strong>
```
nvtop # recommended
htop
```
9. <strong>Monitor certain process for GPU utilization</strong>
```
nvidia-smi pmon -i <GPU_NUM>
e.g., nvidia-smi pmon -i 0
```

| GPU | PID | Type | SM(%) == Volatile GPU-Util| Mem(MB) == Memory Usage| Enc | Dec | Command |
|----------|----------|----------|----------|----------|----------|----------|----------|
| 0  |   | C:Compute, G:Graphic | GPU utilization | allocated GPU mem amount | | | executed cmd |

Type C is for computation such as CUDA or pytorch, and Type G is for graphic rendering job such as OpenGL.

10. <strong>Nohup for background processing</strong>
```
CUDA_VISIBLE_DEVICES=0 nohup python <your_cmd> > output.log 2>&1 &
```

If you want to save output log with a different filename, then add the following command to the above. (default filename: `nohup.out`)

```
> file_name.log
```
11. <strong>More details in process</strong>
```
ps -fp <PID>
```
12. <strong>Terminate a process</strong>
```
kill <PID>
```
13. <strong>Select process using current port</strong>
```
netstat -tulnp | grep <port_num>
```
14. <strong>(System) RAM usage monitoring</strong>
```
watch -n 1 'free -h && echo "=== GPU ===" && nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```
</details>

<details>
<summary><h2><strong>Conda</strong></h2></summary>
<strong>1. Install pytorch library with cuda</strong>

You might as well check whether cuda version is aligned with pytorch one.

```
conda install pytorch=='your version' torchvision=='version' torchaudio=='version' pytorch-cuda='cuda-version' -c pytorch -c nvidia
```
You can refer to the following regarding to the version; [Pytorch previous version](https://pytorch.org/get-started/previous-versions/)

<strong>2. Check whether to be ready for running GPU or installed cuda on your OS </strong>
```
cuda_is_available() module in pytorch
```
<strong>3. List of conda virtual environments</strong>
```
conda env list
```
<strong>4. Create conda virtual environment python version is specified</strong>
```
conda create -n (env name) python='version'
```
<strong>5. Remove your conda virtual environment</strong>
```
conda env remove --name (env name) --all
```
<strong>6. Activate/Deactivate conda virtual environment</strong>
```
conda activate/deactivate
```
<strong>7. Create a new conda environment with old library</strong>
```
conda create --name <new_name> --clone <old_env_name>
e.g., conda create --name new_nev --clone llara
```
<strong>8. Create current environment.yml</strong>
```
conda env export > environment.yml
```
<storng>9. Install library dependency via environment.yaml</strong>
```
conda env update --name <env-name> --file environment.yaml
# In the activated env
conda env update --file environment.yaml
```
<strong>Clean conda cache</strong>
```
conda clean --all -y
```
</details>

<details>
<summary><h2><strong>VS Code</strong></h2></summary>

<strong>Manage kinds of hidden files</strong>
```
ctrl+shift+p
files.exclude
```

<strong>1. Cwd path setting</strong>
```
ctrl+shift+p
```
<strong>2. KeyInterrupt during code execution</strong>
```
ctrl+c
```
or you can insert process termination call(i.e., exit()) into your code snippet 
```
exit()
```
<strong>3. list of hidden extensions</strong>
```
ctrl + , -> search files.exclude
```

<strong>4. automatic formatting for json file</strong>
```
shift+alt+F # If you try to it with large-size(>20MB) json, then it cannot be executed.

shift+option+F # for macOS
```
</details>

<details>
<summary><h2><strong>Unzip</strong></h2></summary>

<strong>Unzip your file in specified directory</strong>
```
# -q: quiet mode, -qq: without any output
unzip file_name.zip -d /path/to/directory
unzip -qq (your zip file name)
```

<strong>Download files stored in google drive</strong>
```
pip install gdown
gdown --fuzzy (google drive link)
```

```
file_name.zip -d /path/to/directory
```

<strong>tar file</strong>
```
tar -xvzf file_name.tar.gz 

# -v: verbose
tar -czvf file_name.tar.gz --exclude='desired exclude path' path/to/destination

tar -xvzf fine_name.tar.gz -C path/to/destination/
```

<strong>unzip multiple files via 7z</strong>

If having multiple segments of a file(e.g., split.z01, split.z02  ..., and split.zip), first of all you should check existence of zip file (i.e., split.zip).
Then, you can get a merged file by executing ``7z``.
Finally, it will be able to result in intended file by unzipping newly generated file.

You can test it via [Uground in GUI-Actor-Data](https://huggingface.co/datasets/cckevinn/GUI-Actor-Data/tree/main)

```
(sudo) apt-get install p7zip-full
7z x Uground_images_split.zip
unzip <newly generated file_name>
```
</details>

<details>
<summary><h2><strong>Git</strong></h2></summary>
  
<strong>Print all branches</strong>
```
git branch
```

<strong>move other branch</strong>
```
git checkout <branch_name>
```

<strong>Create a branch and move to it</strong>
```
git checkout -b <branch_name>
```

<strong>Setting user name and email</strong>
```
git config user.name "Your Name"
git config user.email "you@example.com"
```

<strong>Check git's global setting</strong>
```
git config --global --list
```

<strong>Print commit logs</strong>
```
git log
```

<strong>Clone not all but certain directories for big size repository(you can test it via [transformers](https://github.com/huggingface/transformers))</strong>
```
git clone --no-checkout <repo_url>
```
```
cd repo_dir
```
```
git sparse-checkout init --cone
```
```
git sparse-checkout set dir1 dir2 ...
```
```
git checkout main
```

<strong>Fetch modified remote repo without merge</strong>
```
git fetch origin
git reset --hard origin/main
```

<strong>8. Git lfs installation</strong>

You can download [the specific version of lfs](https://github.com/git-lfs/git-lfs/releases) depending on your OS if you are not a root manager.
```
$ tar -xvzf <git-fls-tar.gz file_name>
$ cd <generated_tar.gz file_name>
 
$ ./install.sh

# You can check whether to succeed to install git lfs via the below command.
$ git lfs install
```

<strong>Exclude added file to be commited</strong>
```
# delete all added files
git reset HEAD

# delete certain added file
git reset HEAD path/to/file
```

<strong>Download model from huggingface</strong>
```
huggingface-cli download <repo_name> --cache-dir <destination_path>

huggingface-cli download Qwen/Qwen2.5-7B --cache-dir /opt/utl/hg_cache
```

</details>

<details>
<summary><h2><strong>pip</strong></h2></summary>

<strong>Itemize installed libraries</strong>
```
pip freeze > requirements.txt
```

<strong>Check Pytorch and flash-attention version</strong>
```
python -c "import torch; print(torch.__version__)"
python -c "import flash_attn; print(flash_attn.__version__)"
```

```
pip install flash-attn --no-build-isolation # prevent from the dependency problem.
pip install flash-attn --no-cache-dir # don't refer to cache for possible mismatch library.
```

<strong>Install pytorch aligned with cuda version</strong>
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/your_cuda_version
e.g., pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 flash-attn==2.6.3
```

<strong>Check version of library installed with pip</strong>
```
pip show <library_name>
e.g., pip show smolagents
```

<strong>Show state-of-the-art library installed with PyPI</strong>
```
pip index versions <library_name>
e.g.,
pip index versions flash-attn
```

Upgrade
```
pip install -U <library_name>
```
</details>

<details>
  <summary><h2><strong>csv, xlsx extension</strong></h2></summary>
If you change the file which extension is csv, then you might as well save xlsx extension to ensure that changes are applied in terms of visualization.
(csv 파일의 세팅을 변경했다면, 시각화 측면에서 변경사항이 반영되도록 xlsx 확장자로 저장하는 게 낫다.)
</details>

<details>
<summary><h2><strong>bash</strong></h2></summary>

<strong>mkdir</strong>
```
mkdir -p data/{pdf,html,youtube,docx,ppt,txt,output,generated,cleaned,final}
```

<strong>Search for such keywords in file</strong>

```
grep -r "variable" path/to/directory
e.g., grep -r "TASK_MAPPING" ./LIBERO
```
The results above the example are as follows:
```
./LIBERO/libero/libero/envs/__init__.py:from .bddl_base_domain import TASK_MAPPING
./LIBERO/libero/libero/envs/env_wrapper.py:        self.env = TASK_MAPPING[self.problem_name](
./LIBERO/libero/libero/envs/bddl_base_domain.py:TASK_MAPPING = {}
./LIBERO/libero/libero/envs/bddl_base_domain.py:    TASK_MAPPING[target_class.__name__.lower()] = target_class
./LIBERO/scripts/create_dataset.py:    env = TASK_MAPPING[problem_name](
./LIBERO/scripts/collect_demonstration.py:    env = TASK_MAPPING[problem_name](
./LIBERO/scripts/libero_100_collect_demonstrations.py:    env = TASK_MAPPING[problem_name](
```

<strong>1. Open file</strong>
```
vi file-name
```
<strong>2. Access via insert mode</strong>
```
i
```
<strong>3. Print and set environment variables</strong>
```
echo $YOUR_ENV_VARIABLE_NAME
e.g., echo $CUDA_VISIBLE_DEVICES
```

If you want to set `environment variable` permanently, then you can use the following command:

```
echo "export <variable_name>=value" >>  ~/.bashrc
source ~/.bashrc

e.g.,
echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
source ~/.bashrc
```

However, I prefer to modifying `~/.bashrc` via `vi ~/.bashrc` rather than using the above in terms of readability.
In details, the environment variable will be written at the very bottom after executing the above command.

When you press ESC, then linux will be changed to command mode.

<h3><strong>3.Save</strong></h3>

<strong>3-1. Save changes and exit</strong>
```
:wq
```
<strong>3-2. Save changes without exit</strong>
```
:w
```
<strong>3-3. Exit without saving</strong>
```
:qa!
```

<h3><strong>4. Delete</strong></h3>

<strong>4-1. Delete one character</strong>
```
x
```
<strong>4-2. Delete a word</strong>
```
dw
```
<strong>4-3. Delete a line after cursor</strong>
```
d$
```
<strong>4-4. Create an empty file which called in file-name</strong>
```
touch
```
<strong>5. Output result or log</strong>
```
filename > result.txt
```
<strong>6. Check owner, group, and others permission on [r]ead, [w]rite, and e[x]ecution</strong>
```
ls -l (file_name)
```
<strong>7. Modify the file called in file-name. Before executing this command, you might as well check your permission on read, write, and execution.</strong>
```
vi (file-name)
```
Although you chanage your file's mode via chmod command, it cannot be changed because of parent directory's permission. Therefore, you could check the upper directory's permission.

<strong>8. printk is used in kernel mode instead of printf</strong>
```
printk
```
<strong>9. Copy target file</strong>
```
cp <target-file> <target-directory>
```
<strong>10. Clear kernel log</strong>
```
dmesg -C
```
<strong>11. Transfer file or directory of virtualbox(guestOS) to local computer(HostOS)</strong>
```
$scp -r source-path HostOS's username@host_ip:destination-path
e.g., scp -r /hw js@192.x.x.x:/Users/JS/Desktop/
```
<strong>(Recommended) 12. Another methods of 11.</strong>
```
$ssh serverA_username@ip # access serverA via ssh

$sftp -P port_num serverB_username@ip # Access serverA to serverB via sftp
$put -r <source_path> <destination_path>

$scp -i ~/.ssh/id_rsa -P 30024 -r <data_path> root@<serverB_ip>:/home/root/
```
<strong>13. Log last access time, modified time and last change mode time</strong>
```
stat file-name
```
<strong>14. remove contents in file but preserve file itself</strong>
```
> file-name
```
<strong>15. ls -l </strong>
```
 |m| g| o|      owner         group             last_edit   dir name
drwxr-x--- 22 junseoklee   junseoklee    4096 Feb 24 11:13 junseoklee
```
In case of directory, x means execution. You can access it via cd cmd.
| Case | r | x | 설명 |
|----------|----------|----------|----------|
| ls dir  | Yes  | No  | 디렉토리 목록 보려면 r 필요 |
| cd dir  | No  | Yes  | 디렉토리 진입하려면 x 필요 |
| ls dir/file  | No  | Yes  | 디렉토리 내 파일 정보 보려면 x 필요 |

<strong>wget</strong>
```
wget -c <your_url> -O <output_path> -o output.log &
e.g.,
wget -c https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1.zip -O /home/dataset/AVSD/Charades_v1.zip -o /home/dataset/AVSD/output.log &

-c: continue(이어받기)
-o: output log to file
-O: output file path
&: background execution
```

<strong>Identify process occupied with certain port</strong>

```
torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address. port: 29500, useIpv6: false, code: -98, name: EADDRINUSE, message: address already in use
```

If you are struggling with the above problem, occupied process in certain port, you can resolve it by selecting port_no and then killing process.

```
lsof -i :<port_no>
e.g., lsof -i :29500
```

<strong>mv></strong>

You can move certain files or directories to destination via `mv`.
```
mv <file_name> <destination_path>
e.g., mv file.txt /home/junseoklee/~
```

</details>

<details>
<summary><h2><strong>Docker</strong></h2></summary>

<strong>Check docker's executing containers/images</strong>
```
docker ps
# list out including stopped containers
docker ps -a
```

```
docker images
docker rmi <image_id>
```

<strong>Pull(Download) predefined image</strong>
```
docker pull <dockerhub_name>
e.g., docker pull pytorch/pytorch
```

<strong>Enter pulled image</strong>
```
'''
--rm: temporal
docker run -it --rm <image_name> <list of cmds>
'''
docker run -it --rm pytorch/pytorch /bin/bash
```

<strong>Save the container as image</strong>
```
docker commit <container_id> <intended_repo_name>:<tag>
e.g.,
docker commit 7e8e pytorch/example:jslee
```

<strong>Create and Execute a container</strong>
```
docker compose up --build
# background
docker compose up --build -d
```

<strong>Stop the container</strong>
```
docker compose down
```

<strong>3. Essential files deploying containers</strong>

Dockerfile, compose.yaml, .dockerignore

<strong>4. Update running compose as coders edit and save codes</strong>
```
docker watch
```
</details>

<details>
<summary><h2>vLLM</h2></summary>
  
<strong>Serve model without mentioned supported list</strong>
```
vllm serve [huggingface_repo or local_path]
```

</details>
